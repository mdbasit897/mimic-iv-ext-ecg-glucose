-- =============================================================================
-- SENSITIVITY ANALYSIS VARIANT B — Fast-Acting Insulin Window: ±150 min
-- =============================================================================
-- Original pipeline: mimiciv_ecg_glucose_pipeline.sql
-- Change from original: Line 169  120 → 150  (insulin_fast CTE only)
-- Purpose: sensitivity analysis — test broader PK window
-- Run in BigQuery, export result as: sensitivity_B_150min.csv
-- After export: feed into q3_sensitivity_analysis.py alongside
--               sensitivity_A_90min.csv and original dataset CSV
-- DO NOT use this CSV as your published dataset — original SQL is canonical
-- =============================================================================


-- =============================================================================
-- SOURCE DATASET VERSIONS
-- =============================================================================
-- MIMIC-IV v3.1
--   PhysioNet DOI : https://doi.org/10.13026/kpb9-mt58
--   Published     : October 11, 2024
--   BigQuery      : physionet-data.mimiciv_3_1_hosp
--                   physionet-data.mimiciv_3_1_icu
--
-- MIMIC-IV-ECG v1.0
--   PhysioNet DOI : https://doi.org/10.13026/4nqg-sb35
--   Published     : September 15, 2023
--   BigQuery      : physionet-data.mimiciv_ecg
-- =============================================================================
-- MIMIC-IV × MIMIC-IV-ECG: Temporal Glucose–ECG Alignment Pipeline
-- Enhancements:
--   [1] Exogenous insulin / IV dextrose intervention flagging (inputevents)
--   [2] ECG Signal Quality Index (SQI) heuristics from machine_measurements
--   [3] Multi-formula QTc: Bazett · Fridericia · Framingham · Hodges
--   [4] Lagged Glucose Dynamics: dG/dt rate-of-change (mg/dL/hr), delta,
--       inter-measurement gap, glycaemic trajectory classification,
--       intervention-propagation guard, and first-measurement NULL audit
-- Patches (v3.1):
--   [P1] Minimum gap lower gate (< 15 min) added to rate computation and
--        gap_exceeds_threshold flag — eliminates rapid-repeat artefact rates
--        observed as low as -3000 mg/dL/hr in validation results
--   [P2] Bradycardia QTc divergence hard exclusion — BRADYCARDIA rows with
--        qtc_formula_range_ms > 200 ms promoted to EXCLUDE (erroneous RR
--        intervals producing physiologically impossible HR values, e.g. 2 bpm)
-- Patches (v3.2):
--   [P3] Trajectory threshold recalibration — prior thresholds (±1/±3 mg/dL/hr)
--        were too tight for retrospective lab glucose data, misclassifying
--        noise as RAPID change (only 1% STABLE in v3.1 results). Corrected to
--        ADA/ISO 15197 CGM standards converted to mg/dL/hr:
--          RAPID     : ±60 mg/dL/hr  (was ±3)   ← ±1.0 mg/dL/min
--          MODERATE  : ±20 mg/dL/hr  (was ±1)   ← ±0.33 mg/dL/min
--          STABLE    : < ±20 mg/dL/hr (was ±1)
-- Datasets: physionet-data.mimiciv_3_1_hosp   (Hospital)
--           physionet-data.mimiciv_3_1_icu     (ICU / inputevents)
--           physionet-data.mimiciv_ecg    (ECG)
-- =============================================================================

-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 1 ▸ Blood Glucose Ground Truth
--   itemid 50809 → Glucose (Blood Gas) | 50931 → Glucose (Chemistry)
--   itemid 52027 → no matching rows in MIMIC-IV v3.1 (retained in filter; contributes 0 records)
--   Plausibility gate: 30 – 700 mg/dL
-- ─────────────────────────────────────────────────────────────────────────────
WITH glucose_labs AS (
  SELECT
    le.subject_id,
    le.hadm_id,
    le.labevent_id,
    le.specimen_id,
    le.itemid,
    dl.label                               AS glucose_label,
    le.charttime                           AS glucose_time,
    le.valuenum                            AS glucose_mg_dl,
    le.valueuom,
    le.flag                                AS lab_flag,
    le.ref_range_lower,
    le.ref_range_upper,
    le.priority,
    CASE
      WHEN le.valuenum <  54  THEN 'SEVERE_HYPO'
      WHEN le.valuenum <  70  THEN 'HYPO'
      WHEN le.valuenum <= 180 THEN 'EUGLYCEMIC'
      WHEN le.valuenum <= 250 THEN 'HYPER'
      ELSE                         'SEVERE_HYPER'
    END                                    AS glycemic_class
  FROM `physionet-data.mimiciv_3_1_hosp.labevents`       AS le
  INNER JOIN `physionet-data.mimiciv_3_1_hosp.d_labitems` AS dl
    ON le.itemid = dl.itemid
  WHERE le.itemid IN (50809, 50931, 52027)
    AND le.valuenum  BETWEEN 30 AND 700
    AND le.charttime IS NOT NULL
),

-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 2 ▸ Patient Demographics & Admission Context
-- ─────────────────────────────────────────────────────────────────────────────
glucose_with_demographics AS (
  SELECT
    gl.*,
    pt.gender,
    pt.anchor_age                          AS age,
    pt.anchor_year_group,
    adm.admittime,
    adm.dischtime,
    adm.admission_type,
    adm.insurance,
    adm.race,
    adm.hospital_expire_flag,
    DATETIME_DIFF(gl.glucose_time, adm.admittime, MINUTE) / 60.0
                                           AS hours_since_admission
  FROM glucose_labs AS gl
  INNER JOIN `physionet-data.mimiciv_3_1_hosp.patients`   AS pt  ON gl.subject_id = pt.subject_id
  LEFT  JOIN `physionet-data.mimiciv_3_1_hosp.admissions` AS adm ON gl.hadm_id    = adm.hadm_id
),

-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 3 ▸ ICU Stay Context
-- ─────────────────────────────────────────────────────────────────────────────
glucose_with_icu AS (
  SELECT
    gd.*,
    icu.stay_id,
    icu.first_careunit,
    icu.last_careunit,
    icu.intime                             AS icu_intime,
    icu.outtime                            AS icu_outtime,
    icu.los                                AS icu_los_days,
    (gd.glucose_time BETWEEN icu.intime AND icu.outtime)
                                           AS during_icu_stay,
    DATETIME_DIFF(gd.glucose_time, icu.intime, MINUTE) / 60.0
                                           AS hours_since_icu_admission
  FROM glucose_with_demographics AS gd
  LEFT JOIN `physionet-data.mimiciv_3_1_icu.icustays` AS icu
    ON  gd.subject_id = icu.subject_id
    AND gd.hadm_id    = icu.hadm_id
    AND gd.glucose_time BETWEEN icu.intime AND icu.outtime
),

-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 4 ▸ [ENHANCEMENT 1] Exogenous Intervention Flagging
--
--   Rationale: Exogenous insulin drives rapid glucose ↓ while IV dextrose
--   drives rapid glucose ↑, both decoupling the autonomic ECG–glucose
--   relationship assumed by non-invasive prediction models. Pairs recorded
--   during active administration must be flagged and optionally excluded.
--
--   Insulin itemids (MIMIC-IV ICU d_items — regular + infusion types):
--     Regular Insulin    : 223258
--     Insulin - Humalog  : 229299  (Lispro, fast-acting)
--     Insulin - Novolog  : 228959  (Aspart, fast-acting)
--     Insulin - NPH      : 223262  (intermediate)
--     Insulin - Glargine : 223260  (basal — slower effect, flag conservatively)
--     Insulin - 70/30    : 223259
--
--   IV Dextrose itemids:
--     Dextrose 5%  (D5W)        : 220949
--     Dextrose 10% (D10W)       : 220950
--     Dextrose 50% (D50W, bolus): 228142
--     Dextrose 5% in 0.9% NaCl : 220952
--     Dextrose 5% in 0.45% NaCl: 220955
--     Dextrose 5% in LR        : 220953
--
--   Pharmacokinetic windows applied:
--     Fast-acting insulin  : effect window  = ±120 min (onset ~15 min, peak ~60-90 min)
--     Insulin infusion     : window = entire infusion + 60 min washout
--     IV Dextrose          : effect window  = ±60  min (rapid onset)
--     Basal insulin        : conservative flag window ±240 min
-- ─────────────────────────────────────────────────────────────────────────────

-- 4a. Fast-acting & Regular Insulin administrations (bolus + infusion)
insulin_fast AS (
  SELECT
    subject_id,
    stay_id,
    starttime,
    endtime,
    itemid,
    amount,
    amountuom,
    rate,
    rateuom,
    statusdescription,
    'INSULIN_FAST' AS intervention_type,
    150            AS effect_window_minutes   -- ±150 min pharmacokinetic window  [SENSITIVITY-B]
  FROM `physionet-data.mimiciv_3_1_icu.inputevents`
  WHERE itemid IN (223258, 229299, 228959, 223259)  -- Regular, Lispro, Aspart, 70/30
    AND statusdescription != 'Rewritten'             -- Exclude amended/cancelled orders
    AND amount > 0
),

-- 4b. Basal insulin (slower PK — wider conservative window)
insulin_basal AS (
  SELECT
    subject_id,
    stay_id,
    starttime,
    endtime,
    itemid,
    amount,
    amountuom,
    rate,
    rateuom,
    statusdescription,
    'INSULIN_BASAL' AS intervention_type,
    240             AS effect_window_minutes
  FROM `physionet-data.mimiciv_3_1_icu.inputevents`
  WHERE itemid IN (223260, 223262)   -- Glargine, NPH
    AND statusdescription != 'Rewritten'
    AND amount > 0
),

-- 4c. IV Dextrose administrations
dextrose_iv AS (
  SELECT
    subject_id,
    stay_id,
    starttime,
    endtime,
    itemid,
    amount,
    amountuom,
    rate,
    rateuom,
    statusdescription,
    'DEXTROSE_IV' AS intervention_type,
    60            AS effect_window_minutes    -- ±60 min for rapid glucose rise
  FROM `physionet-data.mimiciv_3_1_icu.inputevents`
  WHERE itemid IN (220949, 220950, 228142, 220952, 220955, 220953)
    AND statusdescription != 'Rewritten'
    AND amount > 0
),

-- 4d. Unified intervention events
all_interventions AS (
  SELECT * FROM insulin_fast
  UNION ALL
  SELECT * FROM insulin_basal
  UNION ALL
  SELECT * FROM dextrose_iv
),

-- 4e. For each glucose measurement × each intervention, determine if the
--     glucose draw falls within the pharmacokinetic effect window.
--     Window logic: glucose_time within [starttime - window, endtime + window]
--     For infusions (endtime > starttime + 5 min), the active window spans
--     the full infusion duration plus a washout tail.
glucose_intervention_flags AS (
  SELECT
    gi.labevent_id,
    gi.subject_id,
    gi.glucose_time,
    -- Insulin flags
    LOGICAL_OR(
      iv.intervention_type IN ('INSULIN_FAST', 'INSULIN_BASAL')
      AND gi.glucose_time BETWEEN
          DATETIME_SUB(iv.starttime, INTERVAL iv.effect_window_minutes MINUTE)
          AND DATETIME_ADD(iv.endtime,   INTERVAL iv.effect_window_minutes MINUTE)
    )                                                AS insulin_active,
    -- Dextrose flags
    LOGICAL_OR(
      iv.intervention_type = 'DEXTROSE_IV'
      AND gi.glucose_time BETWEEN
          DATETIME_SUB(iv.starttime, INTERVAL iv.effect_window_minutes MINUTE)
          AND DATETIME_ADD(iv.endtime,   INTERVAL iv.effect_window_minutes MINUTE)
    )                                                AS dextrose_active,
    -- Fast-acting insulin specifically (tighter coupling concern)
    LOGICAL_OR(
      iv.intervention_type = 'INSULIN_FAST'
      AND gi.glucose_time BETWEEN
          DATETIME_SUB(iv.starttime, INTERVAL iv.effect_window_minutes MINUTE)
          AND DATETIME_ADD(iv.endtime,   INTERVAL iv.effect_window_minutes MINUTE)
    )                                                AS fast_insulin_active,
    -- Concurrent dual intervention (insulin + dextrose → sliding scale, most confounded)
    LOGICAL_OR(
      iv.intervention_type IN ('INSULIN_FAST', 'INSULIN_BASAL')
      AND gi.glucose_time BETWEEN
          DATETIME_SUB(iv.starttime, INTERVAL iv.effect_window_minutes MINUTE)
          AND DATETIME_ADD(iv.endtime,   INTERVAL iv.effect_window_minutes MINUTE)
    )
    AND
    LOGICAL_OR(
      iv.intervention_type = 'DEXTROSE_IV'
      AND gi.glucose_time BETWEEN
          DATETIME_SUB(iv.starttime, INTERVAL iv.effect_window_minutes MINUTE)
          AND DATETIME_ADD(iv.endtime,   INTERVAL iv.effect_window_minutes MINUTE)
    )                                                AS dual_intervention_active,
    -- Closest insulin dose amount within window (for severity weighting)
    MAX(
      CASE WHEN iv.intervention_type IN ('INSULIN_FAST', 'INSULIN_BASAL')
            AND gi.glucose_time BETWEEN
                DATETIME_SUB(iv.starttime, INTERVAL iv.effect_window_minutes MINUTE)
                AND DATETIME_ADD(iv.endtime, INTERVAL iv.effect_window_minutes MINUTE)
           THEN iv.amount ELSE NULL END
    )                                                AS max_insulin_dose_units,
    MAX(
      CASE WHEN iv.intervention_type = 'DEXTROSE_IV'
            AND gi.glucose_time BETWEEN
                DATETIME_SUB(iv.starttime, INTERVAL iv.effect_window_minutes MINUTE)
                AND DATETIME_ADD(iv.endtime, INTERVAL iv.effect_window_minutes MINUTE)
           THEN iv.amount ELSE NULL END
    )                                                AS max_dextrose_amount_ml
  FROM glucose_with_icu  AS gi
  LEFT JOIN all_interventions AS iv
    ON  gi.subject_id = iv.subject_id
  GROUP BY gi.labevent_id, gi.subject_id, gi.glucose_time
),

-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 5 ▸ Merge Intervention Flags Back Into Glucose Records
-- ─────────────────────────────────────────────────────────────────────────────
glucose_flagged AS (
  SELECT
    gi.*,
    COALESCE(inf.insulin_active,         FALSE) AS insulin_active,
    COALESCE(inf.dextrose_active,        FALSE) AS dextrose_active,
    COALESCE(inf.fast_insulin_active,    FALSE) AS fast_insulin_active,
    COALESCE(inf.dual_intervention_active, FALSE) AS dual_intervention_active,
    inf.max_insulin_dose_units,
    inf.max_dextrose_amount_ml,
    -- Composite decoupling risk score: observed values 0, 1, 2, 4 (3 cannot occur)
    (CASE WHEN COALESCE(inf.fast_insulin_active, FALSE)     THEN 2 ELSE 0 END
   + CASE WHEN COALESCE(inf.dextrose_active, FALSE)         THEN 1 ELSE 0 END
   + CASE WHEN COALESCE(inf.dual_intervention_active, FALSE) THEN 1 ELSE 0 END)
                                               AS decoupling_risk_score,
    -- Recommended usage flag for DL training
    CASE
      WHEN COALESCE(inf.dual_intervention_active, FALSE) THEN 'EXCLUDE'
      WHEN COALESCE(inf.fast_insulin_active, FALSE)      THEN 'FLAG_HIGH_RISK'
      WHEN COALESCE(inf.insulin_active, FALSE)
        OR COALESCE(inf.dextrose_active, FALSE)          THEN 'FLAG_MODERATE_RISK'
      ELSE                                                    'CLEAN'
    END                                        AS intervention_status
  FROM glucose_with_icu    AS gi
  LEFT JOIN glucose_intervention_flags AS inf
    ON gi.labevent_id = inf.labevent_id
),

-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 6 ▸ [ENHANCEMENT 4] Lagged Glucose Dynamics — dG/dt Rate of Change
--
--   CLINICAL RATIONALE
--   ──────────────────
--   The autonomic nervous system (ANS) responds to the *velocity* of glycaemic
--   change more acutely than to the static glucose level. A rapid glucose fall
--   triggers sympathetic counter-regulatory activation (catecholamine surge,
--   HR elevation, QTc prolongation) even before the absolute value reaches a
--   hypoglycaemic threshold. This is the primary physiological mechanism
--   linking ECG morphology to glucose dynamics, and is the signal GlucoFormer
--   must learn to exploit.
--
--   IMPLEMENTATION DECISIONS (addressing known methodological risks)
--   ────────────────────────────────────────────────────────────────
--   (A) PARTITION BOUNDARY: PARTITION BY subject_id, stay_id
--       NOT subject_id alone — prevents cross-admission deltas where the gap
--       between discharge and re-admission would produce a meaningless Δ.
--
--   (B) TIME-NORMALISED RATE: glucose_delta_mg_dl / gap_hours → mg/dL/hr
--       A raw LAG difference is dimensionally inconsistent across unequal
--       sampling intervals. The normalised rate is the clinically comparable
--       quantity. Raw delta and gap are retained as separate columns so
--       GlucoFormer can model the relationship explicitly rather than
--       absorbing a collapsed single feature.
--
--   (C) GAP VALIDITY GATE: gap > 6 hours → rate set to NULL
--       Gaps > 6 hrs make dG/dt clinically uninterpretable (multiple
--       physiological states may have intervened). Rate is nullified;
--       gap and delta are retained for the model to learn from.
--       A secondary flag (gap_exceeds_threshold) marks these rows explicitly.
--
--   (D) INTERVENTION PROPAGATION GUARD
--       A CLEAN glucose reading whose *predecessor* was drawn during active
--       insulin/dextrose infusion will inherit a pharmacologically driven
--       delta — importing confound silently into what appears clean data.
--       lag_intervention_status carries the predecessor's intervention_status
--       so GlucoFormer can learn to discount these deltas, and
--       delta_is_intervention_confounded flags the most dangerous cases.
--
--   (E) FIRST-MEASUREMENT NULL HANDLING
--       Every patient's first observation within a stay has NULL lag values.
--       is_first_glucose_in_stay marks these rows. In GlucoFormer, treat
--       glucose_rate_mg_dl_per_hr as a MASKED input (learned mask embedding)
--       when NULL — do NOT impute zero, which would falsely signal stability.
--
--   OUTPUT COLUMNS (6a → glucose_with_dynamics)
--   ─────────────────────────────────────────────
--   lag_glucose_mg_dl          : previous glucose value (mg/dL)
--   lag_glucose_time           : timestamp of previous measurement
--   lag_intervention_status    : intervention_status of the preceding row
--   glucose_delta_mg_dl        : current − previous (signed, mg/dL)
--                                positive = rising, negative = falling
--   inter_measurement_gap_min  : time between consecutive measurements (min)
--   inter_measurement_gap_hr   : same in hours (for rate calculation)
--   gap_exceeds_threshold      : TRUE if gap > 360 min (6 hr) OR gap < 15 min (rapid-repeat)
--   glucose_rate_mg_dl_per_hr  : dG/dt in mg/dL/hr; NULL if gap > 6 hr
--   delta_is_intervention_confounded : TRUE if lag row was EXCLUDE/HIGH_RISK
--   is_first_glucose_in_stay   : TRUE if no prior observation in this stay
--   glucose_trajectory         : clinical classification of rate direction
-- ─────────────────────────────────────────────────────────────────────────────

-- 6a. Ordered glucose sequence per patient per ICU stay with LAG values
glucose_ordered AS (
  SELECT
    *,
    -- ── Lag values (previous measurement within same subject + stay) ──────────
    LAG(glucose_mg_dl)      OVER (
      PARTITION BY subject_id, stay_id
      ORDER     BY glucose_time
    )                                          AS lag_glucose_mg_dl,

    LAG(glucose_time)       OVER (
      PARTITION BY subject_id, stay_id
      ORDER     BY glucose_time
    )                                          AS lag_glucose_time,

    -- Carry forward intervention status of the PRIOR measurement
    -- so downstream logic can detect confounded deltas
    LAG(intervention_status) OVER (
      PARTITION BY subject_id, stay_id
      ORDER     BY glucose_time
    )                                          AS lag_intervention_status,

    -- Row number within stay — row 1 = first measurement (LAG will be NULL)
    ROW_NUMBER() OVER (
      PARTITION BY subject_id, stay_id
      ORDER     BY glucose_time
    )                                          AS glucose_seq_in_stay

  FROM glucose_flagged   -- source: includes intervention_status for propagation guard
),

-- 6b. Compute derived dynamics columns
glucose_with_dynamics AS (
  SELECT
    go.*,

    -- ── Is this the first glucose draw in this ICU stay? ─────────────────────
    (go.glucose_seq_in_stay = 1)               AS is_first_glucose_in_stay,

    -- ── Raw signed delta (mg/dL) ─────────────────────────────────────────────
    (go.glucose_mg_dl - go.lag_glucose_mg_dl)  AS glucose_delta_mg_dl,

    -- ── Inter-measurement gap ────────────────────────────────────────────────
    DATETIME_DIFF(go.glucose_time, go.lag_glucose_time, MINUTE)
                                               AS inter_measurement_gap_min,
    DATETIME_DIFF(go.glucose_time, go.lag_glucose_time, MINUTE) / 60.0
                                               AS inter_measurement_gap_hr,

    -- ── Gap validity flag ────────────────────────────────────────────────────
    -- Upper gate: gap > 360 min (6 hrs) → rate clinically uninterpretable
    -- Lower gate: gap < 15 min          → near-duplicate / rapid-repeat draw;
    --   a small delta over a tiny interval produces physiologically impossible
    --   rates (observed as low as -3000 mg/dL/hr in validation results).
    --   15 min = one clinical measurement cycle (minimum meaningful interval).
    (DATETIME_DIFF(go.glucose_time, go.lag_glucose_time, MINUTE) > 360
     OR DATETIME_DIFF(go.glucose_time, go.lag_glucose_time, MINUTE) < 15)
                                               AS gap_exceeds_threshold,

    -- ── Time-normalised rate of change: dG/dt (mg/dL/hr) ─────────────────────
    -- Set to NULL when:
    --   • gap > 360 min (6 hrs) — clinically uninterpretable
    --   • gap = 0 (duplicate timestamp — division guard)
    --   • lag values are NULL (first measurement in stay)
    CASE
      WHEN go.lag_glucose_time IS NULL THEN NULL
      WHEN DATETIME_DIFF(go.glucose_time, go.lag_glucose_time, MINUTE) > 360 THEN NULL
      WHEN DATETIME_DIFF(go.glucose_time, go.lag_glucose_time, MINUTE) < 15  THEN NULL  -- lower gate: rapid-repeat artefact
      WHEN DATETIME_DIFF(go.glucose_time, go.lag_glucose_time, MINUTE) = 0   THEN NULL
      ELSE SAFE_DIVIDE(
        (go.glucose_mg_dl - go.lag_glucose_mg_dl),
         DATETIME_DIFF(go.glucose_time, go.lag_glucose_time, MINUTE) / 60.0
      )
    END                                        AS glucose_rate_mg_dl_per_hr,

    -- ── Intervention propagation guard ───────────────────────────────────────
    -- TRUE when the prior measurement was drawn under active pharmacological
    -- influence — the delta is mechanistically confounded even if the CURRENT
    -- measurement appears CLEAN
    (go.lag_intervention_status IN ('EXCLUDE', 'FLAG_HIGH_RISK'))
                                               AS delta_is_intervention_confounded,

    -- ── Glycaemic trajectory classification ──────────────────────────────────
    --
    -- THRESHOLD RATIONALE (v3.2 recalibration):
    -- ─────────────────────────────────────────
    -- Prior thresholds (±1 / ±3 mg/dL/hr) were too tight for retrospective
    -- lab glucose data, misclassifying routine measurement noise as RAPID
    -- change and producing only 1% STABLE records (validated in v3.1 results).
    --
    -- Corrected thresholds are derived from clinical CGM standards converted
    -- to mg/dL/hr, consistent with ADA and ISO 15197 rate-of-change criteria:
    --
    --   CGM standard       mg/dL/min    →   mg/dL/hr (this pipeline)
    --   ─────────────────────────────────────────────────────────────
    --   Rapid change       ±1.0         →   ±60
    --   Moderate change    ±0.33        →   ±20
    --   Stable             < ±0.33      →   < ±20
    --
    -- Clinical interpretation:
    --   RAPID_FALL   (< −60 mg/dL/hr): strong sympathetic counter-regulatory
    --                activation; catecholamine surge → HR↑, QTc prolongation
    --   MODERATE_FALL(−60 to −20):     measurable autonomic response expected
    --   STABLE       (±20 mg/dL/hr):   negligible ANS perturbation from rate;
    --                                  static glucose level is primary driver
    --   MODERATE_RISE(+20 to +60):     insulin demand signal building
    --   RAPID_RISE   (> +60 mg/dL/hr): acute hyperglycaemic trajectory;
    --                                  sympathetic tone may be elevated
    --
    -- Note: retrospective lab glucose (MIMIC-IV) has lower time resolution
    -- than CGM (~1-4 hr vs 5 min), so even the corrected thresholds will
    -- produce more STABLE classifications than CGM-based studies. This is
    -- expected and should be stated in the Methods section.
    -- ─────────────────────────────────────────────────────────────────────────
    CASE
      -- Cannot classify: first measurement, gap too wide, or gap too narrow
      WHEN go.lag_glucose_time IS NULL                                        THEN 'UNKNOWN_FIRST'
      WHEN DATETIME_DIFF(go.glucose_time, go.lag_glucose_time, MINUTE) > 360 THEN 'UNKNOWN_GAP'
      WHEN DATETIME_DIFF(go.glucose_time, go.lag_glucose_time, MINUTE) < 15  THEN 'UNKNOWN_GAP'

      -- RAPID_FALL: < −60 mg/dL/hr (equivalent to < −1 mg/dL/min)
      -- Strong sympathetic counter-regulatory activation; primary ANS driver
      WHEN SAFE_DIVIDE(
             (go.glucose_mg_dl - go.lag_glucose_mg_dl),
              DATETIME_DIFF(go.glucose_time, go.lag_glucose_time, MINUTE) / 60.0
           ) < -60.0                                                           THEN 'RAPID_FALL'

      -- MODERATE_FALL: −60 to −20 mg/dL/hr (−1.0 to −0.33 mg/dL/min)
      WHEN SAFE_DIVIDE(
             (go.glucose_mg_dl - go.lag_glucose_mg_dl),
              DATETIME_DIFF(go.glucose_time, go.lag_glucose_time, MINUTE) / 60.0
           ) < -20.0                                                           THEN 'MODERATE_FALL'

      -- STABLE: within ±20 mg/dL/hr (±0.33 mg/dL/min)
      -- Negligible rate-driven ANS perturbation; static level is primary signal
      WHEN ABS(SAFE_DIVIDE(
             (go.glucose_mg_dl - go.lag_glucose_mg_dl),
              DATETIME_DIFF(go.glucose_time, go.lag_glucose_time, MINUTE) / 60.0
           )) <= 20.0                                                          THEN 'STABLE'

      -- MODERATE_RISE: +20 to +60 mg/dL/hr (+0.33 to +1.0 mg/dL/min)
      WHEN SAFE_DIVIDE(
             (go.glucose_mg_dl - go.lag_glucose_mg_dl),
              DATETIME_DIFF(go.glucose_time, go.lag_glucose_time, MINUTE) / 60.0
           ) <= 60.0                                                           THEN 'MODERATE_RISE'

      -- RAPID_RISE: > +60 mg/dL/hr (> +1.0 mg/dL/min)
      ELSE                                                                          'RAPID_RISE'
    END                                        AS glucose_trajectory

  FROM glucose_ordered AS go
),

-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 6c ▸ NULL Audit CTE — for Methods Section Reporting
--   Computes the per-stay and per-careunit first-measurement null rate.
--   Reference this in your paper's preprocessing section to document the
--   fraction of records with missing dG/dt and any demographic skew.
--   (Commented out from main query — run separately for reporting.)
-- ─────────────────────────────────────────────────────────────────────────────
/*
glucose_dynamics_null_audit AS (
  SELECT
    first_careunit,
    COUNT(*)                                              AS total_records,
    COUNTIF(is_first_glucose_in_stay)                     AS n_first_measurement_nulls,
    COUNTIF(gap_exceeds_threshold)                        AS n_gap_threshold_nulls,
    COUNTIF(glucose_rate_mg_dl_per_hr IS NULL)            AS n_rate_null_total,
    COUNTIF(glucose_rate_mg_dl_per_hr IS NULL) * 100.0
      / COUNT(*)                                          AS pct_rate_null,
    COUNTIF(delta_is_intervention_confounded)             AS n_confounded_deltas,
    ROUND(AVG(inter_measurement_gap_hr), 2)               AS mean_gap_hr,
    ROUND(STDDEV(inter_measurement_gap_hr), 2)            AS std_gap_hr,
    APPROX_QUANTILES(inter_measurement_gap_hr, 4)[OFFSET(1)] AS p25_gap_hr,
    APPROX_QUANTILES(inter_measurement_gap_hr, 4)[OFFSET(2)] AS median_gap_hr,
    APPROX_QUANTILES(inter_measurement_gap_hr, 4)[OFFSET(3)] AS p75_gap_hr
  FROM glucose_with_dynamics
  GROUP BY first_careunit
  ORDER BY first_careunit
),
*/

-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 7 ▸ ECG Records from record_list
-- ─────────────────────────────────────────────────────────────────────────────
ecg_records AS (
  SELECT
    subject_id,
    study_id,
    file_name,
    ecg_time,
    path AS waveform_path
  FROM `physionet-data.mimiciv_ecg.record_list`
),

-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 8 ▸ [ENHANCEMENT 2] Signal Quality Index (SQI) Computation
--
--   The MIMIC-IV-ECG machine_measurements table does not store raw per-lead
--   amplitude data, but it exposes derived interval measurements and free-text
--   report fields (report_0 … report_17) that encode cart-detected artifacts.
--   We implement a multi-criterion SQI heuristic using:
--
--   (A) INTERVAL PLAUSIBILITY CHECKS — physiologically impossible values
--       indicate failed lead contact, baseline wander, or motion artifacts.
--         RR interval  : valid 300–2000 ms (HR 30–200 bpm)
--         QRS duration : valid  60–200 ms
--         QT interval  : valid 200–600 ms
--         P-onset      : must precede QRS onset
--         T-end        : must follow QRS end
--         Axis ranges  : P/QRS/T axes valid -180° to +180°
--
--   (B) INTERNAL CONSISTENCY CHECKS
--         QRS duration = qrs_end - qrs_onset  (must match cart values ±10 ms)
--         PR interval  = qrs_onset - p_onset  (expected 120–300 ms)
--         JT interval  = t_end - qrs_end      (expected 150–400 ms)
--
--   (C) REPORT-FIELD ARTIFACT KEYWORD SCAN
--       The cart auto-generates interpretive text in report_0–report_17.
--       Keywords: 'artifact', 'noise', 'cannot interpret', 'poor quality',
--                 'baseline wander', 'lead off', 'uninterpretable'
--       Any match → automatic low-quality flag.
--
--   SQI Score: 0–10 (10 = pristine signal)
--     Each failed criterion subtracts points as specified below.
--   SQI Category thresholds:
--     8–10 → GOOD    (use for training)
--     5–7  → FAIR    (use for pre-training / augmentation only)
--     0–4  → POOR    (exclude)
-- ─────────────────────────────────────────────────────────────────────────────
ecg_with_sqi AS (
  SELECT
    mm.subject_id,
    mm.study_id,
    mm.ecg_time,
    mm.rr_interval,
    mm.p_onset,
    mm.p_end,
    mm.qrs_onset,
    mm.qrs_end,
    mm.t_end,
    mm.p_axis,
    mm.qrs_axis,
    mm.t_axis,
    mm.bandwidth,
    mm.filtering,
    -- ── Derived interval values (ms) ────────────────────────────────────────
    (mm.qrs_end   - mm.qrs_onset)              AS qrs_duration_ms,
    (mm.t_end     - mm.qrs_onset)              AS qt_raw_ms,
    (mm.qrs_onset - mm.p_onset)               AS pr_interval_ms,
    (mm.t_end     - mm.qrs_end)               AS jt_interval_ms,
    SAFE_DIVIDE(60000.0, mm.rr_interval)       AS heart_rate_bpm,

    -- ── (A) Interval plausibility — each violation: -1 pt ───────────────────
    (CASE WHEN mm.rr_interval  BETWEEN 300  AND 2000 THEN 1 ELSE 0 END) AS sqi_rr_ok,
    (CASE WHEN (mm.qrs_end - mm.qrs_onset) BETWEEN 60 AND 200 THEN 1 ELSE 0 END)
                                                                         AS sqi_qrs_ok,
    (CASE WHEN (mm.t_end - mm.qrs_onset)   BETWEEN 200 AND 600 THEN 1 ELSE 0 END)
                                                                         AS sqi_qt_ok,
    (CASE WHEN mm.p_onset IS NOT NULL
           AND mm.qrs_onset IS NOT NULL
           AND mm.p_onset < mm.qrs_onset                THEN 1 ELSE 0 END) AS sqi_p_before_qrs,
    (CASE WHEN mm.t_end IS NOT NULL
           AND mm.qrs_end IS NOT NULL
           AND mm.t_end > mm.qrs_end                    THEN 1 ELSE 0 END) AS sqi_t_after_qrs,

    -- ── (B) Internal consistency checks — each violation: -1 pt ─────────────
    (CASE WHEN (mm.qrs_onset - mm.p_onset) BETWEEN 120 AND 300 THEN 1 ELSE 0 END)
                                                                         AS sqi_pr_ok,
    (CASE WHEN (mm.t_end - mm.qrs_end)     BETWEEN 150 AND 400 THEN 1 ELSE 0 END)
                                                                         AS sqi_jt_ok,
    (CASE WHEN mm.p_axis   BETWEEN -90  AND 90  THEN 1 ELSE 0 END)      AS sqi_p_axis_ok,
    (CASE WHEN mm.qrs_axis BETWEEN -180 AND 180 THEN 1 ELSE 0 END)      AS sqi_qrs_axis_ok,
    (CASE WHEN mm.t_axis   BETWEEN -180 AND 180 THEN 1 ELSE 0 END)      AS sqi_t_axis_ok,

    -- ── (C) Report artifact keyword scan — any match: -3 pts ────────────────
    (CASE WHEN
         LOWER(COALESCE(mm.report_0,  '')) LIKE ANY ('%artifact%','%noise%','%poor quality%','%lead off%','%cannot interpret%','%uninterpretable%','%baseline wander%')
      OR LOWER(COALESCE(mm.report_1,  '')) LIKE ANY ('%artifact%','%noise%','%poor quality%','%lead off%','%cannot interpret%','%uninterpretable%','%baseline wander%')
      OR LOWER(COALESCE(mm.report_2,  '')) LIKE ANY ('%artifact%','%noise%','%poor quality%','%lead off%','%cannot interpret%','%uninterpretable%','%baseline wander%')
      OR LOWER(COALESCE(mm.report_3,  '')) LIKE ANY ('%artifact%','%noise%','%poor quality%','%lead off%','%cannot interpret%','%uninterpretable%','%baseline wander%')
      OR LOWER(COALESCE(mm.report_4,  '')) LIKE ANY ('%artifact%','%noise%','%poor quality%','%lead off%','%cannot interpret%','%uninterpretable%','%baseline wander%')
      OR LOWER(COALESCE(mm.report_5,  '')) LIKE ANY ('%artifact%','%noise%','%poor quality%','%lead off%','%cannot interpret%','%uninterpretable%','%baseline wander%')
      OR LOWER(COALESCE(mm.report_6,  '')) LIKE ANY ('%artifact%','%noise%','%poor quality%','%lead off%','%cannot interpret%','%uninterpretable%','%baseline wander%')
      OR LOWER(COALESCE(mm.report_7,  '')) LIKE ANY ('%artifact%','%noise%','%poor quality%','%lead off%','%cannot interpret%','%uninterpretable%','%baseline wander%')
      OR LOWER(COALESCE(mm.report_8,  '')) LIKE ANY ('%artifact%','%noise%','%poor quality%','%lead off%','%cannot interpret%','%uninterpretable%','%baseline wander%')
      OR LOWER(COALESCE(mm.report_9,  '')) LIKE ANY ('%artifact%','%noise%','%poor quality%','%lead off%','%cannot interpret%','%uninterpretable%','%baseline wander%')
      OR LOWER(COALESCE(mm.report_10, '')) LIKE ANY ('%artifact%','%noise%','%poor quality%','%lead off%','%cannot interpret%','%uninterpretable%','%baseline wander%')
      OR LOWER(COALESCE(mm.report_11, '')) LIKE ANY ('%artifact%','%noise%','%poor quality%','%lead off%','%cannot interpret%','%uninterpretable%','%baseline wander%')
      OR LOWER(COALESCE(mm.report_12, '')) LIKE ANY ('%artifact%','%noise%','%poor quality%','%lead off%','%cannot interpret%','%uninterpretable%','%baseline wander%')
      OR LOWER(COALESCE(mm.report_13, '')) LIKE ANY ('%artifact%','%noise%','%poor quality%','%lead off%','%cannot interpret%','%uninterpretable%','%baseline wander%')
      OR LOWER(COALESCE(mm.report_14, '')) LIKE ANY ('%artifact%','%noise%','%poor quality%','%lead off%','%cannot interpret%','%uninterpretable%','%baseline wander%')
      OR LOWER(COALESCE(mm.report_15, '')) LIKE ANY ('%artifact%','%noise%','%poor quality%','%lead off%','%cannot interpret%','%uninterpretable%','%baseline wander%')
      OR LOWER(COALESCE(mm.report_16, '')) LIKE ANY ('%artifact%','%noise%','%poor quality%','%lead off%','%cannot interpret%','%uninterpretable%','%baseline wander%')
      OR LOWER(COALESCE(mm.report_17, '')) LIKE ANY ('%artifact%','%noise%','%poor quality%','%lead off%','%cannot interpret%','%uninterpretable%','%baseline wander%')
     THEN 1 ELSE 0 END)                                                  AS report_artifact_flag

  FROM `physionet-data.mimiciv_ecg.machine_measurements` AS mm
),

-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 9 ▸ [ENHANCEMENT 3] Multi-Formula QTc + Composite SQI Score
--
--   QT CORRECTION FORMULAS (QT and RR in seconds):
--   ─────────────────────────────────────────────────────────────────────────
--   Bazett    : QTcB = QT / √(RR)
--     • Most widely used; OVER-corrects at high HR (>90 bpm) — common in ICU
--     • UNDER-corrects at low HR (<60 bpm)
--     • Recommended use: reference comparison only; NOT primary in ICU cohort
--
--   Fridericia: QTcF = QT / ∛(RR)          [cube root]
--     • Preferable across the wide HR range of ICU patients (40–150 bpm)
--     • Less rate-dependent bias; recommended as PRIMARY QTc in this pipeline
--
--   Framingham: QTcFram = QT + 0.154 × (1 – RR)    [linear correction]
--     • Good for normal sinus rhythm; less validated at extreme HRs
--     • Useful for cross-validation against Fridericia
--
--   Hodges    : QTcH = QT + 1.75 × (HR − 60)       [linear HR-based]
--     • Avoids overestimation in tachycardia better than Bazett
--     • Expressed in ms (HR in bpm, QT in ms)
--
--   Normal QTc thresholds (ms): prolonged > 450 (male) / 460 (female)
--   Critically prolonged: > 500 ms (Torsades risk)
-- ─────────────────────────────────────────────────────────────────────────────
ecg_with_qtc AS (
  SELECT
    esqi.*,
    -- Intermediate: QT and RR in seconds for formula application
    esqi.qt_raw_ms       / 1000.0 AS qt_sec,
    esqi.rr_interval     / 1000.0 AS rr_sec,

    -- ── Bazett QTc (primary citation formula) ────────────────────────────────
    SAFE_DIVIDE(
      esqi.qt_raw_ms / 1000.0,
      SQRT(esqi.rr_interval / 1000.0)
    ) * 1000.0                                     AS qtc_bazett_ms,

    -- ── Fridericia QTc (recommended for ICU tachycardia/bradycardia) ─────────
    SAFE_DIVIDE(
      esqi.qt_raw_ms / 1000.0,
      POW(esqi.rr_interval / 1000.0, 1.0/3.0)
    ) * 1000.0                                     AS qtc_fridericia_ms,

    -- ── Framingham QTc (linear additive) ─────────────────────────────────────
    ( (esqi.qt_raw_ms / 1000.0)
      + 0.154 * (1.0 - (esqi.rr_interval / 1000.0))
    ) * 1000.0                                     AS qtc_framingham_ms,

    -- ── Hodges QTc (linear HR-based, in ms) ──────────────────────────────────
    ( esqi.qt_raw_ms
      + 1.75 * (SAFE_DIVIDE(60000.0, esqi.rr_interval) - 60.0)
    )                                              AS qtc_hodges_ms,

    -- ── Composite SQI Score (0–10) ───────────────────────────────────────────
    -- Interval checks (5 criteria × 1 pt each  = max 5 pts)
    -- Consistency checks (5 criteria × 1 pt each = max 5 pts)
    -- Report artifact flag (deducts 3 pts)
    GREATEST(0,
        esqi.sqi_rr_ok
      + esqi.sqi_qrs_ok
      + esqi.sqi_qt_ok
      + esqi.sqi_p_before_qrs
      + esqi.sqi_t_after_qrs
      + esqi.sqi_pr_ok
      + esqi.sqi_jt_ok
      + esqi.sqi_p_axis_ok
      + esqi.sqi_qrs_axis_ok
      + esqi.sqi_t_axis_ok
      - (esqi.report_artifact_flag * 3)
    )                                              AS sqi_score,

    -- ── SQI Category ──────────────────────────────────────────────────────────
    CASE
      WHEN GREATEST(0,
             esqi.sqi_rr_ok + esqi.sqi_qrs_ok + esqi.sqi_qt_ok +
             esqi.sqi_p_before_qrs + esqi.sqi_t_after_qrs +
             esqi.sqi_pr_ok + esqi.sqi_jt_ok +
             esqi.sqi_p_axis_ok + esqi.sqi_qrs_axis_ok + esqi.sqi_t_axis_ok
             - (esqi.report_artifact_flag * 3)) >= 8  THEN 'GOOD'
      WHEN GREATEST(0,
             esqi.sqi_rr_ok + esqi.sqi_qrs_ok + esqi.sqi_qt_ok +
             esqi.sqi_p_before_qrs + esqi.sqi_t_after_qrs +
             esqi.sqi_pr_ok + esqi.sqi_jt_ok +
             esqi.sqi_p_axis_ok + esqi.sqi_qrs_axis_ok + esqi.sqi_t_axis_ok
             - (esqi.report_artifact_flag * 3)) >= 5  THEN 'FAIR'
      ELSE                                             'POOR'
    END                                              AS sqi_category

  FROM ecg_with_sqi AS esqi
),

-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 10 ▸ QTc Prolongation & Inter-Formula Divergence Assessment
--
--   QTc formulas diverge most at extreme HRs — this CTE computes:
--   (a) prolongation flags per formula (clinical safety threshold 500 ms)
--   (b) inter-formula range (max – min across all 4 formulas)
--       High divergence (>30 ms) indicates heart-rate extremes where
--       Bazett is most unreliable — use Fridericia as authoritative value.
-- ─────────────────────────────────────────────────────────────────────────────
ecg_final AS (
  SELECT
    eqtc.*,
    -- Prolongation flags (gender-agnostic threshold: 500 ms = universal danger)
    (eqtc.qtc_bazett_ms     > 500) AS qtc_bazett_prolonged,
    (eqtc.qtc_fridericia_ms > 500) AS qtc_fridericia_prolonged,
    (eqtc.qtc_framingham_ms > 500) AS qtc_framingham_prolonged,
    (eqtc.qtc_hodges_ms     > 500) AS qtc_hodges_prolonged,
    -- Conservative: prolonged if ANY formula agrees
    (eqtc.qtc_bazett_ms     > 500
     OR eqtc.qtc_fridericia_ms > 500
     OR eqtc.qtc_framingham_ms > 500
     OR eqtc.qtc_hodges_ms  > 500) AS qtc_any_prolonged,
    -- Strict: prolonged only if ALL formulas agree
    (eqtc.qtc_bazett_ms     > 500
     AND eqtc.qtc_fridericia_ms > 500
     AND eqtc.qtc_framingham_ms > 500
     AND eqtc.qtc_hodges_ms  > 500) AS qtc_all_prolonged,
    -- Inter-formula divergence (ms) — high = Bazett unreliable
    GREATEST(
      eqtc.qtc_bazett_ms,
      eqtc.qtc_fridericia_ms,
      eqtc.qtc_framingham_ms,
      eqtc.qtc_hodges_ms
    ) - LEAST(
      eqtc.qtc_bazett_ms,
      eqtc.qtc_fridericia_ms,
      eqtc.qtc_framingham_ms,
      eqtc.qtc_hodges_ms
    )                                                  AS qtc_formula_range_ms,
    -- Recommended QTc value: Fridericia for ICU (wide HR), Bazett for reference
    eqtc.qtc_fridericia_ms                             AS qtc_recommended_ms,
    -- HR regime classification (determines which formula is most appropriate)
    CASE
      WHEN eqtc.heart_rate_bpm < 50  THEN 'BRADYCARDIA'
      WHEN eqtc.heart_rate_bpm < 60  THEN 'LOW_NORMAL'
      WHEN eqtc.heart_rate_bpm <= 90 THEN 'NORMAL'
      WHEN eqtc.heart_rate_bpm <= 120 THEN 'MILD_TACHY'
      ELSE                                 'TACHYCARDIA'
    END                                                AS hr_regime
  FROM ecg_with_qtc AS eqtc
),

-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 11 ▸ Join ECG records with enriched machine measurements
-- ─────────────────────────────────────────────────────────────────────────────
ecg_enriched AS (
  SELECT
    er.subject_id,
    er.study_id,
    er.file_name,
    er.ecg_time,
    er.waveform_path,
    ef.*  EXCEPT (subject_id, study_id, ecg_time)
  FROM ecg_records AS er
  LEFT JOIN ecg_final AS ef
    ON  er.subject_id = ef.subject_id
    AND er.study_id   = ef.study_id
),

-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 12 ▸ Nearest-Neighbor Temporal Alignment
--   ±240 min outer window; nearest ECG per glucose observation
-- ─────────────────────────────────────────────────────────────────────────────
glucose_ecg_candidates AS (
  SELECT
    gf.*,
    ee.study_id,
    ee.file_name                     AS ecg_file_name,
    ee.ecg_time,
    ee.waveform_path,
    -- SQI
    ee.sqi_score,
    ee.sqi_category,
    ee.report_artifact_flag,
    ee.sqi_rr_ok, ee.sqi_qrs_ok, ee.sqi_qt_ok,
    ee.sqi_p_before_qrs, ee.sqi_t_after_qrs,
    ee.sqi_pr_ok, ee.sqi_jt_ok,
    ee.sqi_p_axis_ok, ee.sqi_qrs_axis_ok, ee.sqi_t_axis_ok,
    -- ECG intervals (ms)
    ee.rr_interval,
    ee.qrs_duration_ms,
    ee.qt_raw_ms,
    ee.pr_interval_ms,
    ee.jt_interval_ms,
    ee.heart_rate_bpm,
    ee.hr_regime,
    ee.p_axis, ee.qrs_axis, ee.t_axis,
    -- Multi-formula QTc
    ee.qtc_bazett_ms,
    ee.qtc_fridericia_ms,
    ee.qtc_framingham_ms,
    ee.qtc_hodges_ms,
    ee.qtc_recommended_ms,
    ee.qtc_formula_range_ms,
    -- Prolongation flags
    ee.qtc_any_prolonged,
    ee.qtc_all_prolonged,
    ee.qtc_bazett_prolonged,
    ee.qtc_fridericia_prolonged,
    ee.qtc_framingham_prolonged,
    ee.qtc_hodges_prolonged,
    -- Temporal offset
    DATETIME_DIFF(gf.glucose_time, ee.ecg_time, MINUTE)
                                     AS glucose_ecg_offset_minutes,
    ABS(DATETIME_DIFF(gf.glucose_time, ee.ecg_time, MINUTE))
                                     AS abs_offset_minutes,
    ROW_NUMBER() OVER (
      PARTITION BY gf.subject_id, gf.labevent_id
      ORDER BY ABS(DATETIME_DIFF(gf.glucose_time, ee.ecg_time, MINUTE)) ASC
    )                                AS ecg_proximity_rank
  FROM glucose_with_dynamics  AS gf
  INNER JOIN ecg_enriched AS ee
    ON  gf.subject_id = ee.subject_id
    AND ABS(DATETIME_DIFF(gf.glucose_time, ee.ecg_time, MINUTE)) <= 240
),

nearest_pairs AS (
  SELECT * FROM glucose_ecg_candidates WHERE ecg_proximity_rank = 1
),

-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 13 ▸ Temporal Relationship & Alignment Quality Labels
-- ─────────────────────────────────────────────────────────────────────────────
labeled_pairs AS (
  SELECT
    *,
    CASE
      WHEN glucose_ecg_offset_minutes > 0 THEN 'ECG_BEFORE_GLUCOSE'
      WHEN glucose_ecg_offset_minutes < 0 THEN 'ECG_AFTER_GLUCOSE'
      ELSE                                     'SIMULTANEOUS'
    END                              AS temporal_relationship,
    CASE
      WHEN abs_offset_minutes <= 30  THEN 'TIGHT'
      WHEN abs_offset_minutes <= 60  THEN 'MODERATE'
      WHEN abs_offset_minutes <= 120 THEN 'LOOSE'
      ELSE                                'EXTENDED'
    END                              AS alignment_quality
  FROM nearest_pairs
),

-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 14 ▸ Patient-Level Normalization Anchors
-- ─────────────────────────────────────────────────────────────────────────────
patient_glucose_stats AS (
  SELECT
    subject_id,
    COUNT(*)                           AS n_glucose_ecg_pairs,
    AVG(glucose_mg_dl)                 AS patient_mean_glucose,
    STDDEV(glucose_mg_dl)              AS patient_std_glucose,
    MIN(glucose_mg_dl)                 AS patient_min_glucose,
    MAX(glucose_mg_dl)                 AS patient_max_glucose,
    SAFE_DIVIDE(STDDEV(glucose_mg_dl), AVG(glucose_mg_dl)) * 100
                                       AS glucose_cv_percent,
    COUNTIF(glucose_mg_dl BETWEEN 70 AND 180) * 100.0 / COUNT(*)
                                       AS time_in_range_pct,
    COUNTIF(glucose_mg_dl < 70)  * 100.0 / COUNT(*)
                                       AS time_below_range_pct,
    COUNTIF(glucose_mg_dl > 180) * 100.0 / COUNT(*)
                                       AS time_above_range_pct
  FROM labeled_pairs
  GROUP BY subject_id
)

-- =============================================================================
-- STEP 15 ▸ FINAL ASSEMBLY — Fully Enriched D1NAMO-style Multimodal Record
-- =============================================================================
SELECT
  -- ── Identifiers ─────────────────────────────────────────────────────────────
  lp.subject_id,
  lp.hadm_id,
  lp.stay_id,
  lp.labevent_id,
  lp.study_id                          AS ecg_study_id,

  -- ── Patient Demographics ────────────────────────────────────────────────────
  lp.gender,
  lp.age,
  lp.anchor_year_group,
  lp.race,
  lp.insurance,

  -- ── Clinical Admission Context ──────────────────────────────────────────────
  lp.admission_type,
  lp.first_careunit,
  lp.last_careunit,
  lp.icu_los_days,
  lp.during_icu_stay,
  lp.hours_since_admission,
  lp.hours_since_icu_admission,
  lp.hospital_expire_flag,

  -- ── Glucose Ground Truth (TARGET) ──────────────────────────────────────────
  lp.glucose_time,
  lp.glucose_mg_dl,
  lp.glucose_label,
  lp.glycemic_class,
  lp.lab_flag,
  lp.ref_range_lower,
  lp.ref_range_upper,
  lp.priority                          AS lab_priority,

  -- ── [ENHANCEMENT 1] Intervention Flags ─────────────────────────────────────
  lp.insulin_active,
  lp.fast_insulin_active,
  lp.dextrose_active,
  lp.dual_intervention_active,
  lp.max_insulin_dose_units,
  lp.max_dextrose_amount_ml,
  lp.decoupling_risk_score,
  lp.intervention_status,              -- 'CLEAN' | 'FLAG_MODERATE_RISK' | 'FLAG_HIGH_RISK' | 'EXCLUDE'

  -- ── [ENHANCEMENT 4] Lagged Glucose Dynamics ────────────────────────────────
  -- Predecessor measurement
  lp.lag_glucose_mg_dl,                -- Previous glucose value (mg/dL)
  lp.lag_glucose_time,                 -- Timestamp of previous measurement
  lp.lag_intervention_status,          -- Intervention status of the prior row
  lp.glucose_seq_in_stay,              -- Sequence number within this ICU stay

  -- Raw delta and temporal gap
  lp.glucose_delta_mg_dl,              -- Signed difference: current − prior (mg/dL)
  lp.inter_measurement_gap_min,        -- Time since prior measurement (minutes)
  lp.inter_measurement_gap_hr,         -- Time since prior measurement (hours)

  -- Validity and confound guards
  lp.is_first_glucose_in_stay,         -- TRUE → no prior obs; lag cols are NULL
  lp.gap_exceeds_threshold,            -- TRUE → gap > 6 hr OR gap < 15 min; rate is NULL
  lp.delta_is_intervention_confounded, -- TRUE → prior row was under active intervention

  -- Primary dynamic feature for GlucoFormer (NULL when unreliable — use mask embedding)
  lp.glucose_rate_mg_dl_per_hr,        -- dG/dt in mg/dL/hr

  -- Clinical trajectory classification
  lp.glucose_trajectory,               -- RAPID_FALL | MODERATE_FALL | STABLE |
                                       -- MODERATE_RISE | RAPID_RISE | UNKNOWN_*

  -- ── ECG Recording Metadata ──────────────────────────────────────────────────
  lp.ecg_time,
  lp.ecg_file_name,
  lp.waveform_path,
  lp.temporal_relationship,
  lp.alignment_quality,
  lp.glucose_ecg_offset_minutes,
  lp.abs_offset_minutes,

  -- ── [ENHANCEMENT 2] Signal Quality Index ───────────────────────────────────
  lp.sqi_score,
  lp.sqi_category,                     -- 'GOOD' | 'FAIR' | 'POOR'
  lp.report_artifact_flag,
  lp.sqi_rr_ok,
  lp.sqi_qrs_ok,
  lp.sqi_qt_ok,
  lp.sqi_p_before_qrs,
  lp.sqi_t_after_qrs,
  lp.sqi_pr_ok,
  lp.sqi_jt_ok,
  lp.sqi_p_axis_ok,
  lp.sqi_qrs_axis_ok,
  lp.sqi_t_axis_ok,

  -- ── ECG Interval Features ───────────────────────────────────────────────────
  lp.heart_rate_bpm,
  lp.hr_regime,
  lp.rr_interval,
  lp.qrs_duration_ms,
  lp.qt_raw_ms,
  lp.pr_interval_ms,
  lp.jt_interval_ms,
  lp.p_axis,
  lp.qrs_axis,
  lp.t_axis,

  -- ── [ENHANCEMENT 3] Multi-Formula QTc ──────────────────────────────────────
  lp.qtc_bazett_ms,
  lp.qtc_fridericia_ms,
  lp.qtc_framingham_ms,
  lp.qtc_hodges_ms,
  lp.qtc_recommended_ms,               -- = Fridericia (preferred for ICU HR range)
  lp.qtc_formula_range_ms,             -- Inter-formula divergence (high = use Fridericia)
  lp.qtc_any_prolonged,
  lp.qtc_all_prolonged,
  lp.qtc_bazett_prolonged,
  lp.qtc_fridericia_prolonged,
  lp.qtc_framingham_prolonged,
  lp.qtc_hodges_prolonged,

  -- ── Patient-Level Normalization Anchors ────────────────────────────────────
  pgs.n_glucose_ecg_pairs,
  pgs.patient_mean_glucose,
  pgs.patient_std_glucose,
  pgs.patient_min_glucose,
  pgs.patient_max_glucose,
  pgs.glucose_cv_percent,
  pgs.time_in_range_pct,
  pgs.time_below_range_pct,
  pgs.time_above_range_pct,

  -- ── Normalized Glucose Values ───────────────────────────────────────────────
  SAFE_DIVIDE(
    lp.glucose_mg_dl - pgs.patient_mean_glucose,
    NULLIF(pgs.patient_std_glucose, 0)
  )                                    AS glucose_z_score,
  (lp.glucose_mg_dl - 30.0) / (700.0 - 30.0)
                                       AS glucose_minmax_norm,

  -- ── Composite Usability Flag ────────────────────────────────────────────────
  -- Integrates all four enhancement layers into one filterable column.
  -- IDEAL   : all quality gates pass AND ECG precedes glucose (causal direction)
  -- USABLE  : minor quality concern — include with awareness
  -- CAUTION : significant quality concern — use only for pre-training
  -- EXCLUDE : hard exclusion — do not use for any DL training
  CASE
    -- Hard exclusions (order matters — most severe first)
    WHEN lp.intervention_status = 'EXCLUDE'                        THEN 'EXCLUDE'
    WHEN lp.sqi_category        = 'POOR'                           THEN 'EXCLUDE'
    -- Bradycardia QTc divergence exclusion:
    --   In BRADYCARDIA (HR < 50 bpm), inter-formula QTc range > 200 ms indicates
    --   an erroneous RR interval from the ECG cart (e.g. HR = 2 bpm observed in
    --   validation results — physiologically impossible, caused by lead-off or
    --   hardware error). Bazett catastrophically overestimates QTc at extreme
    --   bradycardia, producing formula divergence that signals measurement failure
    --   rather than true cardiac pathology. These records must be excluded.
    WHEN lp.hr_regime = 'BRADYCARDIA'
     AND lp.qtc_formula_range_ms > 200                             THEN 'EXCLUDE'
    -- Caution cases
    WHEN lp.intervention_status = 'FLAG_HIGH_RISK'                 THEN 'CAUTION'
    WHEN lp.sqi_category        = 'FAIR'                           THEN 'CAUTION'
    -- Confounded delta: current row is CLEAN but prior was intervened
    -- Rate feature is unreliable; flag for ablation study
    WHEN lp.delta_is_intervention_confounded                        THEN 'CAUTION'
    -- Gap too wide: trajectory unknown, rate is NULL
    -- Record itself may be valid but dynamic feature is absent
    WHEN lp.gap_exceeds_threshold                                   THEN 'CAUTION'
    -- Ideal: causal direction + tight temporal window + full quality gates
    WHEN lp.temporal_relationship  = 'ECG_BEFORE_GLUCOSE'
      AND lp.alignment_quality    IN ('TIGHT', 'MODERATE')
      AND lp.intervention_status   = 'CLEAN'
      AND lp.sqi_category          = 'GOOD'
      AND NOT lp.is_first_glucose_in_stay          -- dynamic features available
      AND NOT lp.delta_is_intervention_confounded  -- delta is trustworthy
      AND NOT lp.gap_exceeds_threshold             -- rate is computable
                                                                     THEN 'IDEAL'
    ELSE                                                                  'USABLE'
  END                                  AS record_usability,

  -- ── Train/Val/Test Split (subject-level, no patient leakage) ───────────────
  CASE
    WHEN MOD(ABS(FARM_FINGERPRINT(CAST(lp.subject_id AS STRING))), 10) < 8 THEN 'TRAIN'
    WHEN MOD(ABS(FARM_FINGERPRINT(CAST(lp.subject_id AS STRING))), 10) = 8 THEN 'VALIDATION'
    ELSE                                                                         'TEST'
  END                                  AS split

FROM labeled_pairs         AS lp
INNER JOIN patient_glucose_stats AS pgs ON lp.subject_id = pgs.subject_id
ORDER BY lp.subject_id ASC, lp.glucose_time ASC;


-- =============================================================================
-- APPENDIX A: RECOMMENDED DL TRAINING SUBSET
--   ECG before glucose | GOOD SQI | CLEAN intervention | TIGHT/MODERATE window
--   Dynamic features fully available (not first measurement, gap within 6 hr)
-- =============================================================================
/*
SELECT * FROM <above_query_as_view_or_subquery>
WHERE record_usability IN ('IDEAL', 'USABLE')
  AND temporal_relationship = 'ECG_BEFORE_GLUCOSE'
  AND split = 'TRAIN'
ORDER BY subject_id, glucose_time;
*/

-- =============================================================================
-- APPENDIX B: QTc FORMULA DIVERGENCE AUDIT
--   Identify records where Bazett is unreliable (high inter-formula range)
-- =============================================================================
/*
SELECT
  hr_regime,
  COUNT(*)                                    AS n,
  ROUND(AVG(qtc_formula_range_ms), 1)         AS mean_formula_divergence_ms,
  ROUND(AVG(qtc_bazett_ms), 1)                AS mean_qtc_bazett,
  ROUND(AVG(qtc_fridericia_ms), 1)            AS mean_qtc_fridericia,
  ROUND(AVG(qtc_framingham_ms), 1)            AS mean_qtc_framingham,
  ROUND(AVG(qtc_hodges_ms), 1)                AS mean_qtc_hodges,
  COUNTIF(qtc_formula_range_ms > 30) * 100.0 / COUNT(*) AS pct_high_divergence
FROM <view>
GROUP BY hr_regime ORDER BY hr_regime;
*/

-- =============================================================================
-- APPENDIX C: INTERVENTION IMPACT ANALYSIS
--   Compare glucose distribution across intervention categories
-- =============================================================================
/*
SELECT
  intervention_status,
  glycemic_class,
  COUNT(*)                             AS n,
  ROUND(AVG(glucose_mg_dl), 1)         AS mean_glucose,
  ROUND(STDDEV(glucose_mg_dl), 1)      AS std_glucose,
  ROUND(AVG(abs_offset_minutes), 1)    AS mean_ecg_offset_min
FROM <view>
GROUP BY intervention_status, glycemic_class
ORDER BY intervention_status, glycemic_class;
*/

-- =============================================================================
-- APPENDIX D: SQI DISTRIBUTION BY CARE UNIT
--   Understand signal quality by clinical environment
-- =============================================================================
/*
SELECT
  first_careunit,
  sqi_category,
  COUNT(*)                             AS n,
  ROUND(AVG(sqi_score), 2)             AS mean_sqi,
  COUNTIF(report_artifact_flag = 1) * 100.0 / COUNT(*) AS pct_artifact_flagged
FROM <view>
GROUP BY first_careunit, sqi_category
ORDER BY first_careunit, sqi_category;
*/

-- =============================================================================
-- APPENDIX E: [ENHANCEMENT 4] GLUCOSE DYNAMICS NULL AUDIT
--   Run this standalone to quantify dG/dt data availability for Methods section.
--   Reports the fraction of records with NULL rates by care unit, gap cause,
--   and demographic group — required for publication transparency.
-- =============================================================================
/*
SELECT
  first_careunit,
  COUNT(*)                                                   AS total_records,
  COUNTIF(is_first_glucose_in_stay)                          AS n_first_obs_null,
  COUNTIF(gap_exceeds_threshold AND NOT is_first_glucose_in_stay)
                                                             AS n_gap_null,
  COUNTIF(glucose_rate_mg_dl_per_hr IS NULL)                 AS n_rate_null_total,
  ROUND(COUNTIF(glucose_rate_mg_dl_per_hr IS NULL) * 100.0
        / COUNT(*), 1)                                       AS pct_rate_null,
  COUNTIF(delta_is_intervention_confounded)                  AS n_confounded_deltas,
  ROUND(COUNTIF(delta_is_intervention_confounded) * 100.0
        / COUNT(*), 1)                                       AS pct_confounded,
  ROUND(AVG(inter_measurement_gap_hr), 2)                    AS mean_gap_hr,
  ROUND(APPROX_QUANTILES(inter_measurement_gap_hr, 4)[OFFSET(2)], 2)
                                                             AS median_gap_hr,
  COUNTIF(glucose_trajectory = 'RAPID_FALL')  * 100.0 / COUNT(*) AS pct_rapid_fall,
  COUNTIF(glucose_trajectory = 'STABLE')      * 100.0 / COUNT(*) AS pct_stable,
  COUNTIF(glucose_trajectory = 'RAPID_RISE')  * 100.0 / COUNT(*) AS pct_rapid_rise
FROM <view>
GROUP BY first_careunit
ORDER BY first_careunit;
*/

-- =============================================================================
-- APPENDIX F: [ENHANCEMENT 4] TRAJECTORY × GLYCAEMIC CLASS CROSS-TABULATION
--   Validates that trajectory classifications are clinically coherent:
--   RAPID_FALL should be enriched for HYPO/SEVERE_HYPO glycaemic class.
-- =============================================================================
/*
SELECT
  glucose_trajectory,
  glycemic_class,
  COUNT(*)                                AS n,
  ROUND(AVG(glucose_mg_dl), 1)            AS mean_glucose,
  ROUND(AVG(glucose_rate_mg_dl_per_hr), 2) AS mean_rate_mg_dl_hr,
  ROUND(AVG(heart_rate_bpm), 1)           AS mean_hr_bpm,
  ROUND(AVG(qtc_fridericia_ms), 1)        AS mean_qtcf_ms,
  ROUND(AVG(rr_interval), 0)             AS mean_rr_ms
FROM <view>
WHERE glucose_rate_mg_dl_per_hr IS NOT NULL
GROUP BY glucose_trajectory, glycemic_class
ORDER BY glucose_trajectory, glycemic_class;
*/

-- =============================================================================
-- APPENDIX G: GLUCOFORMER ABLATION SPLIT
--   Two training subsets for ablation study:
--   FULL   — includes glucose_rate_mg_dl_per_hr (when available, masked when NULL)
--   STATIC — excludes all dynamic features (glucose_delta, rate, trajectory)
--   Compare performance to quantify the contribution of dG/dt to model accuracy.
--   Report MAE / RMSE / hypoglycaemia sensitivity separately for each subset.
-- =============================================================================
/*
-- Full feature set (with dynamics — use mask embedding for NULL rate)
SELECT *, 'FULL_FEATURES' AS ablation_group
FROM <view>
WHERE record_usability IN ('IDEAL', 'USABLE')
  AND split = 'TRAIN'

UNION ALL

-- Static-only subset (same records, dynamic columns set to NULL for ablation)
SELECT
  * EXCEPT (
    lag_glucose_mg_dl, lag_glucose_time, lag_intervention_status,
    glucose_delta_mg_dl, inter_measurement_gap_min, inter_measurement_gap_hr,
    is_first_glucose_in_stay, gap_exceeds_threshold,
    delta_is_intervention_confounded, glucose_rate_mg_dl_per_hr,
    glucose_trajectory
  ),
  NULL AS lag_glucose_mg_dl,
  NULL AS lag_glucose_time,
  NULL AS lag_intervention_status,
  NULL AS glucose_delta_mg_dl,
  NULL AS inter_measurement_gap_min,
  NULL AS inter_measurement_gap_hr,
  NULL AS is_first_glucose_in_stay,
  NULL AS gap_exceeds_threshold,
  NULL AS delta_is_intervention_confounded,
  NULL AS glucose_rate_mg_dl_per_hr,
  NULL AS glucose_trajectory,
  'STATIC_ONLY' AS ablation_group
FROM <view>
WHERE record_usability IN ('IDEAL', 'USABLE')
  AND split = 'TRAIN';
*/