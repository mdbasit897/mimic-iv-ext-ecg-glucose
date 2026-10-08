# Query guide

`mimiciv_ecg_glucose_pipeline.sql` is the BigQuery SQL that produced
`mimiciv_ecg_glucose_aligned.csv` in **MIMIC-IV-Ext-ECG-Glucose v1.0.0** on PhysioNet.
This file is byte-identical to the copy published on PhysioNet. Do not edit it to
fix analysis issues; handle them in code (see `glucoecg/features.py`).

## Running it

1. Get credentialed PhysioNet access to MIMIC-IV v3.1 and MIMIC-IV-ECG v1.0, and link
   PhysioNet to Google BigQuery (`physionet-data` project).
2. Open the BigQuery console in your own Google Cloud project and run the main query
   (everything up to `ORDER BY lp.subject_id ASC, lp.glucose_time ASC;`).
3. Export the result as CSV. You should get 437,671 rows and 101 columns. The published
   file has SHA-256 `eaea161137acbf4cd913a431e0757206f50f87f99a82d0b6a2c67c1fcd9fe552`.
   Row order and float formatting of a fresh export can differ, so compare counts
   (`pytest tests/` with `GLUCOECG_CSV` set) rather than the hash.

Source tables: `mimiciv_3_1_hosp.labevents`, `d_labitems`, `patients`, `admissions`;
`mimiciv_3_1_icu.icustays`, `inputevents`; `mimiciv_ecg.record_list`, `machine_measurements`.

## Pipeline steps

| Step | CTE | What it does |
| --- | --- | --- |
| 1 | `glucose_labs` | Glucose results, itemids 50931 (chemistry) and 50809 (blood gas), 30-700 mg/dL. Itemid 52027 is listed but returns no rows in v3.1 |
| 2-3 | `glucose_with_demographics`, `glucose_with_icu` | Patient, admission (when `hadm_id` exists) and ICU-stay context |
| 4-5 | `insulin_*`, `dextrose_iv`, `glucose_flagged` | Insulin / dextrose exposure within `[starttime - W, endtime + W]` (W = 120 / 240 / 60 min) |
| 6 | `glucose_ordered`, `glucose_with_dynamics` | Previous glucose within `(subject_id, stay_id)`, gap, delta, rate, trajectory |
| 7-10 | `ecg_records` ... `ecg_final` | ECG measurements, 10-check SQI, four QTc formulas, prolongation flags |
| 11-13 | `glucose_ecg_candidates` ... `labeled_pairs` | Nearest ECG within +/-240 min; direction and alignment tier |
| 14 | `patient_glucose_stats` | Whole-record patient statistics (see warning below) |
| 15 | final `SELECT` | Usability tier and `FARM_FINGERPRINT` patient-level split |

## Columns that must not be model inputs

Several columns are computed from the target glucose or from information that only
exists after it. They are kept for description and stratification:

| Group | Columns | Why |
| --- | --- | --- |
| Target-derived | `glycemic_class`, `lab_flag`, `glucose_delta_mg_dl`, `glucose_rate_mg_dl_per_hr`, `glucose_trajectory`, `glucose_z_score`, `glucose_minmax_norm` | Functions of `glucose_mg_dl` |
| Whole-record patient statistics | `patient_mean_glucose`, `patient_std_glucose`, `patient_min_glucose`, `patient_max_glucose`, `glucose_cv_percent`, `time_in_range_pct`, `time_below_range_pct`, `time_above_range_pct` | `GROUP BY subject_id` over all of a patient's rows, including the current and later ones |
| Known only later | `n_glucose_ecg_pairs`, `icu_los_days`, `hospital_expire_flag`, `last_careunit` | Discharge or whole-record information |
| Intervention annotations | `decoupling_risk_score`, `intervention_status`, `*_active`, `max_*` | The symmetric window includes doses given after the draw, often in response to the measured glucose. Use to filter or stratify |

`glucoecg.features.FORBIDDEN_INPUTS` lists all of them, and `assert_no_leak()` refuses
them. Use `glucoecg.features.add_causal_history()` for history features built only from
earlier records.

## Known behaviours of the SQL (documented, not changed)

- For records with no ICU stay, `stay_id` is NULL, so the lag partition is the patient's
  whole non-ICU sequence; the previous value can come from another encounter.
- Two assays of one draw can share a timestamp (1,023 records share patient and glucose
  time with another record). Their lag gap is 0; the code masks lags with gaps under 15 min.
- `decoupling_risk_score` takes values 0, 1, 2, 4; 2 means fast-acting insulin alone or
  basal insulin + dextrose.
- The nearest-ECG `ROW_NUMBER()` has no tie-breaker. When two ECGs are equally close to a
  glucose result, BigQuery may pick either, so a fresh run can pair some records with a
  different ECG. Comparing the sensitivity exports with the published CSV, 439 of 437,671
  records (0.10%) had a different ECG, all with an identical time offset; alignment counts
  are unaffected, SQI counts change slightly. A future version should add `ee.study_id`
  to the `ORDER BY`.
- `age` is `anchor_age`; ages above 89 appear as 91.
- Machine-measurement placeholders (29999, 32767, 65535) are not cleaned in SQL; the code
  treats `|value| >= 10000` as missing.

## Appendices (commented out at the end of the file)

| Appendix | Purpose |
| --- | --- |
| A | Recommended training subset (IDEAL+USABLE, ECG before glucose, TRAIN) |
| B | QTc formula divergence audit |
| C | Glucose distribution by intervention category |
| D | SQI distribution by care unit |
| E | Glucose-dynamics NULL audit |
| F | Trajectory x glycaemic class cross-tabulation |
| G | Ablation subsets with and without the rate feature (the rate is target-derived) |

## Sensitivity variants

`sql/pipeline_sensitivity_a_90min.sql` and `..._b_150min.sql`
differ from the main query only in the fast-acting insulin window (90 or 150 min instead
of 120). Export each to CSV and pass them to `validation/q3_sensitivity_analysis.py`.
