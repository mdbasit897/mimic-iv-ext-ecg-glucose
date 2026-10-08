"""
Leak-free feature construction for MIMIC-IV-Ext-ECG-Glucose v1.0.0.

The released CSV keeps several columns that are computed from the target
glucose value or from information that only exists after it (see the PhysioNet
README, "Columns that must not be used as model inputs"). This module:

  * lists those columns in ``FORBIDDEN_INPUTS`` and refuses them as inputs;
  * cleans machine-measurement placeholder codes (29999, 32767, 65535) and RR = 0;
  * rebuilds patient glucose history from strictly earlier records only
    (``prior_*`` columns), so a row never sees its own or a later glucose value;
  * masks ``lag_glucose_mg_dl`` when the previous draw is < 15 min earlier
    (mostly the other assay of the same blood draw).

History is built from the records in the CSV (glucose values that had an ECG
within +/-240 min) plus the all-labs ``lag_glucose_mg_dl`` column. Users with
MIMIC-IV access can extend it from ``labevents`` through ``labevent_id``.
"""

import numpy as np
import pandas as pd

TARGET = "glucose_mg_dl"

# ── Allowed model inputs ──────────────────────────────────────────────────────
# ECG machine measurements and values derived only from them
ECG_FEATURES = [
    "heart_rate_bpm", "rr_interval", "qrs_duration_ms", "qt_raw_ms", "pr_interval_ms",
    "jt_interval_ms", "p_axis", "qrs_axis", "t_axis", "qtc_bazett_ms", "qtc_fridericia_ms",
    "qtc_framingham_ms", "qtc_hodges_ms", "qtc_recommended_ms", "qtc_formula_range_ms",
    "sqi_score", "sqi_rr_ok", "sqi_qrs_ok", "sqi_qt_ok", "sqi_p_before_qrs",
    "sqi_t_after_qrs", "sqi_pr_ok", "hr_regime_enc",
]
# Continuous ECG columns that can hold placeholder codes
ECG_NUMERIC = ECG_FEATURES[:15]
QTC_COLS = ["qtc_bazett_ms", "qtc_fridericia_ms", "qtc_framingham_ms", "qtc_hodges_ms",
            "qtc_recommended_ms", "qtc_formula_range_ms"]
# Binary ECG flags (used by the MLP and the PyTorch loader)
ECG_BINARY_FEATURES = [
    "sqi_rr_ok", "sqi_qrs_ok", "sqi_qt_ok", "sqi_p_before_qrs", "sqi_t_after_qrs",
    "sqi_pr_ok", "sqi_jt_ok", "sqi_p_axis_ok", "sqi_qrs_axis_ok", "sqi_t_axis_ok",
    "report_artifact_flag", "qtc_any_prolonged", "qtc_all_prolonged",
]
# Demographics, timing of the pair and admission context known at measurement time
CONTEXT_FEATURES = [
    "age", "gender_m", "abs_offset_minutes", "glucose_ecg_offset_minutes", "ecg_before",
    "alignment_quality_enc", "during_icu", "glucose_seq_in_stay",
    "hours_since_admission", "hours_since_icu_admission",
]
# Glucose history from strictly earlier measurements (built by add_causal_history)
HISTORY_FEATURES = [
    "lag_ok", "gap_ok_hr", "prior_n", "prior_mean", "prior_std", "prior_tir_pct",
    "prior_min", "prior_max", "prior_last", "prior_last_hr", "prior_change",
]

FEATURE_SETS = {
    "ecg_only": ECG_FEATURES,
    "history_only": CONTEXT_FEATURES + HISTORY_FEATURES,
    "ecg_history": ECG_FEATURES + CONTEXT_FEATURES + HISTORY_FEATURES,
}
FEATURE_SET_LABELS = {
    "ecg_only": "XGBoost, ECG only",
    "history_only": "XGBoost, causal history + context",
    "ecg_history": "XGBoost, ECG + causal history + context",
}

# ── Columns that must never be model inputs ───────────────────────────────────
IDENTIFIERS = {"subject_id", "hadm_id", "stay_id", "labevent_id", "ecg_study_id",
               "ecg_file_name", "waveform_path", "glucose_time", "ecg_time", "lag_glucose_time"}
TARGET_DERIVED = {
    TARGET, "glycemic_class", "lab_flag", "glucose_delta_mg_dl", "glucose_rate_mg_dl_per_hr",
    "glucose_trajectory", "glucose_z_score", "glucose_minmax_norm",
    "patient_mean_glucose", "patient_std_glucose", "patient_min_glucose", "patient_max_glucose",
    "glucose_cv_percent", "time_in_range_pct", "time_below_range_pct", "time_above_range_pct",
}
# Known only after the measurement (discharge or whole-record information)
RETROSPECTIVE = {"n_glucose_ecg_pairs", "icu_los_days", "hospital_expire_flag", "last_careunit"}
# Symmetric pharmacokinetic windows include doses started after the draw, often in
# response to the measured glucose, so these annotate records but are not predictors.
INTERVENTION = {
    "insulin_active", "fast_insulin_active", "dextrose_active", "dual_intervention_active",
    "max_insulin_dose_units", "max_dextrose_amount_ml", "decoupling_risk_score",
    "intervention_status", "lag_intervention_status", "delta_is_intervention_confounded",
}
# Tier and split labels are selection variables, not predictors
SELECTION = {"record_usability", "split"}

FORBIDDEN_INPUTS = IDENTIFIERS | TARGET_DERIVED | RETROSPECTIVE | INTERVENTION | SELECTION


def assert_no_leak(columns) -> None:
    """Raise if any requested model input is target-derived, retrospective or an identifier."""
    bad = sorted(FORBIDDEN_INPUTS.intersection(columns))
    if bad:
        raise ValueError(f"Forbidden model inputs (leak target or future information): {bad}")


for _name, _cols in FEATURE_SETS.items():
    assert_no_leak(_cols)
assert_no_leak(ECG_BINARY_FEATURES)


# ── Loading and cleaning ──────────────────────────────────────────────────────

def _to_bool_float(s: pd.Series) -> pd.Series:
    """BigQuery booleans ('true'/'false'/empty) to 1.0 / 0.0 / NaN."""
    if s.dtype == bool:
        return s.astype(float)
    m = s.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=s.index)
    out[m.isin(["true", "1", "1.0"])] = 1.0
    out[m.isin(["false", "0", "0.0"])] = 0.0
    return out


INTERVAL_COLS = ["qrs_duration_ms", "qt_raw_ms", "pr_interval_ms", "jt_interval_ms"]


def clean_ecg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Treat MIMIC-IV-ECG 'not measured' codes and the values derived from them as missing.

    * any ECG value with |value| >= 10000 (codes 29999, 32767, 65535);
    * RR <= 0, which also invalidates heart rate and every QTc;
    * intervals (QRS, QT, PR, JT) <= 0 ms or >= 2000 ms. The CSV stores them as
      differences of two time points, so two placeholder codes give 0 (29999 - 29999)
      or a mid-range value (32767 - 29999 = 2768) that the first rule misses;
    * QTc values (and the formula range) whenever QT is invalid.
    """
    for c in ECG_NUMERIC:
        df.loc[df[c].abs() >= 10000, c] = np.nan
    for c in INTERVAL_COLS:
        df.loc[(df[c] <= 0) | (df[c] >= 2000), c] = np.nan
    rr_bad = ~(df["rr_interval"] > 0)
    df.loc[rr_bad, ["rr_interval", "heart_rate_bpm", *QTC_COLS]] = np.nan
    df.loc[df["qt_raw_ms"].isna(), QTC_COLS] = np.nan
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric encodings used as model inputs."""
    df["alignment_quality_enc"] = df["alignment_quality"].map(
        {"TIGHT": 3, "MODERATE": 2, "LOOSE": 1, "EXTENDED": 0})
    df["hr_regime_enc"] = df["hr_regime"].map(
        {"BRADYCARDIA": 0, "LOW_NORMAL": 1, "NORMAL": 2, "MILD_TACHY": 3, "TACHYCARDIA": 4})
    df.loc[df["heart_rate_bpm"].isna(), "hr_regime_enc"] = np.nan  # RR = 0 is labelled TACHYCARDIA
    df["gender_m"] = (df["gender"] == "M").astype(float)
    df["ecg_before"] = (df["temporal_relationship"] == "ECG_BEFORE_GLUCOSE").astype(float)
    df["during_icu"] = df["during_icu_stay"].fillna(0.0)
    return df


def add_causal_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``prior_*`` history features from strictly earlier glucose times of the same
    patient, plus ``lag_ok`` / ``gap_ok_hr`` (lag masked when the gap is < 15 min).

    Records that share a timestamp (for example chemistry and blood-gas assays of
    one draw) never see each other. Returns the frame sorted by patient and time.
    """
    df["_t"] = pd.to_datetime(df["glucose_time"])
    df = df.sort_values(["subject_id", "_t", "labevent_id"]).reset_index(drop=True)
    g = df[TARGET].astype(float)
    per = (df.assign(_g=g, _g2=g ** 2, _inr=g.between(70, 180).astype(float))
             .groupby(["subject_id", "_t"], sort=True)
             .agg(_s=("_g", "sum"), _s2=("_g2", "sum"), _n=("_g", "size"), _in=("_inr", "sum"),
                  _mn=("_g", "min"), _mx=("_g", "max"), _last=("_g", "last"))
             .reset_index())
    gb = per.groupby("subject_id", sort=False)
    per["prior_n"] = gb["_n"].cumsum() - per["_n"]
    n = per["prior_n"].replace(0, np.nan)
    per["prior_mean"] = (gb["_s"].cumsum() - per["_s"]) / n
    var = (gb["_s2"].cumsum() - per["_s2"]) / n - per["prior_mean"] ** 2
    per["prior_std"] = np.sqrt(var.clip(lower=0))
    per["prior_tir_pct"] = (gb["_in"].cumsum() - per["_in"]) / n * 100
    per["prior_min"] = gb["_mn"].cummin().groupby(per["subject_id"]).shift(1)
    per["prior_max"] = gb["_mx"].cummax().groupby(per["subject_id"]).shift(1)
    per["prior_last"] = gb["_last"].shift(1)
    per["prior_last_hr"] = (per["_t"] - gb["_t"].shift(1)).dt.total_seconds() / 3600
    per["prior_change"] = per["prior_last"] - gb["_last"].shift(2)
    keep = ["subject_id", "_t", *[c for c in HISTORY_FEATURES if c.startswith("prior_")]]
    df = df.merge(per[keep], on=["subject_id", "_t"], how="left").drop(columns="_t")

    ok = df["inter_measurement_gap_min"] >= 15
    df["lag_ok"] = df["lag_glucose_mg_dl"].where(ok)
    df["gap_ok_hr"] = df["inter_measurement_gap_hr"].where(ok)
    return df


def load_dataset(csv_path, history: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Read the released CSV, fix booleans, clean ECG placeholders, encode, add causal history."""
    df = pd.read_csv(csv_path, low_memory=False)
    for c in df.columns:
        if df[c].dtype == object:
            vals = set(df[c].dropna().astype(str).str.lower().unique()[:5])
            if vals and vals <= {"true", "false"}:
                df[c] = _to_bool_float(df[c])
    df = clean_ecg(df)
    df = encode(df)
    if history:
        df = add_causal_history(df)
    if verbose:
        print(f"  Loaded {len(df):,} records | {df['subject_id'].nunique():,} patients")
    return df


# ── Standard row selections ───────────────────────────────────────────────────

def is_iu(df):
    return df["record_usability"].isin(["IDEAL", "USABLE"]).to_numpy()


def train_mask(df, subset: str = "iu"):
    """TRAIN rows: 'iu' = IDEAL+USABLE, 'all' = every non-EXCLUDE tier."""
    base = (df["split"] == "TRAIN").to_numpy()
    return base & (is_iu(df) if subset == "iu" else (df["record_usability"] != "EXCLUDE").to_numpy())


def val_mask(df, subset: str = "iu"):
    base = (df["split"] == "VALIDATION").to_numpy()
    return base & (is_iu(df) if subset == "iu" else (df["record_usability"] != "EXCLUDE").to_numpy())


def test_sets(df) -> dict:
    """
    Fixed TEST row selections; every model is scored on the same rows.

    iu           : IDEAL+USABLE (the manuscript's primary set)
    conservative : IDEAL+USABLE, ECG before glucose, CLEAN (PhysioNet README filter)
    all          : every non-EXCLUDE tier
    iu_history   : IDEAL+USABLE rows with at least one earlier record
    iu_no_history: IDEAL+USABLE rows without any earlier record
    """
    te = (df["split"] == "TEST").to_numpy()
    iu = is_iu(df)
    cons = (iu & (df["temporal_relationship"] == "ECG_BEFORE_GLUCOSE").to_numpy()
            & (df["intervention_status"] == "CLEAN").to_numpy())
    hist = (df["prior_n"].fillna(0) > 0).to_numpy() if "prior_n" in df else np.zeros(len(df), bool)
    return {
        "iu": te & iu,
        "conservative": te & cons,
        "all": te & (df["record_usability"] != "EXCLUDE").to_numpy(),
        "iu_history": te & iu & hist,
        "iu_no_history": te & iu & ~hist,
    }


TEST_SET_LABELS = {
    "iu": "IDEAL+USABLE test",
    "conservative": "Conservative test (I+U, ECG before glucose, CLEAN)",
    "all": "All non-EXCLUDE test",
    "iu_history": "I+U test, with earlier history",
    "iu_no_history": "I+U test, no earlier history",
}

USABILITY_WEIGHTS = {"IDEAL": 4.0, "USABLE": 2.0, "CAUTION": 1.0}


def usability_weights(df) -> np.ndarray:
    return df["record_usability"].map(USABILITY_WEIGHTS).fillna(1.0).to_numpy()
