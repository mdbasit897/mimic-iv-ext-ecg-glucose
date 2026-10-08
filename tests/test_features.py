import numpy as np
import pandas as pd
import pytest

from glucoecg.features import (FEATURE_SETS, FORBIDDEN_INPUTS, add_causal_history, assert_no_leak,
                               clean_ecg, ECG_NUMERIC)


def _toy():
    # patient 1: four draws, two of them at the same timestamp (two assays of one draw)
    return pd.DataFrame({
        "subject_id": [1, 1, 1, 1, 2],
        "labevent_id": [10, 11, 12, 13, 20],
        "glucose_time": ["2150-01-01 08:00", "2150-01-01 10:00", "2150-01-01 10:00",
                         "2150-01-01 14:00", "2150-01-01 09:00"],
        "glucose_mg_dl": [100.0, 200.0, 210.0, 50.0, 300.0],
        "lag_glucose_mg_dl": [np.nan, 100.0, 200.0, 210.0, np.nan],
        "inter_measurement_gap_min": [np.nan, 120.0, 0.0, 240.0, np.nan],
        "inter_measurement_gap_hr": [np.nan, 2.0, 0.0, 4.0, np.nan],
    })


def test_prior_features_use_only_strictly_earlier_values():
    df = add_causal_history(_toy())
    p1 = df[df.subject_id == 1].set_index("labevent_id")
    assert np.isnan(p1.loc[10, "prior_mean"]) and p1.loc[10, "prior_n"] == 0
    # both 10:00 rows see only the 08:00 value, never each other
    for lid in (11, 12):
        assert p1.loc[lid, "prior_n"] == 1
        assert p1.loc[lid, "prior_mean"] == 100
        assert p1.loc[lid, "prior_last"] == 100
    # 14:00 sees 100, 200, 210 but not its own 50
    assert p1.loc[13, "prior_n"] == 3
    assert p1.loc[13, "prior_mean"] == pytest.approx(170)
    assert p1.loc[13, "prior_min"] == 100 and p1.loc[13, "prior_max"] == 210
    assert p1.loc[13, "prior_last"] == 210            # last value of the 10:00 group
    assert p1.loc[13, "prior_change"] == pytest.approx(110)
    assert p1.loc[13, "prior_last_hr"] == pytest.approx(4)
    # the other patient is unaffected
    assert df[df.subject_id == 2]["prior_n"].iloc[0] == 0


def test_lag_masked_for_same_draw():
    df = add_causal_history(_toy()).set_index("labevent_id")
    assert np.isnan(df.loc[12, "lag_ok"])             # gap 0 -> masked
    assert df.loc[11, "lag_ok"] == 100


def test_no_feature_set_contains_forbidden_columns():
    for cols in FEATURE_SETS.values():
        assert not FORBIDDEN_INPUTS.intersection(cols)


def test_assert_no_leak_rejects_target_derived():
    for col in ["patient_mean_glucose", "glucose_delta_mg_dl", "decoupling_risk_score",
                "glycemic_class", "n_glucose_ecg_pairs", "glucose_mg_dl"]:
        with pytest.raises(ValueError):
            assert_no_leak(["heart_rate_bpm", col])


def test_clean_ecg_placeholders():
    df = pd.DataFrame({c: [100.0, 29999.0] for c in ECG_NUMERIC})
    df["rr_interval"] = [800.0, 0.0]
    out = clean_ecg(df)
    assert out.loc[1, "p_axis"] != out.loc[1, "p_axis"]          # NaN
    assert np.isnan(out.loc[1, "heart_rate_bpm"]) and np.isnan(out.loc[1, "qtc_bazett_ms"])
    assert out.loc[0, "p_axis"] == 100


def test_clean_ecg_intervals_derived_from_two_placeholders():
    # QT stored as t_end - qrs_onset: 29999 - 29999 = 0 and 32767 - 29999 = 2768
    df = pd.DataFrame({c: [400.0, 400.0, 400.0] for c in ECG_NUMERIC})
    df["rr_interval"] = [800.0, 800.0, 800.0]
    df["qt_raw_ms"] = [400.0, 0.0, 2768.0]
    df["pr_interval_ms"] = [160.0, -29799.0, 2768.0]
    out = clean_ecg(df)
    assert out.loc[0, "qt_raw_ms"] == 400 and out.loc[0, "qtc_fridericia_ms"] == 400
    for i in (1, 2):
        assert np.isnan(out.loc[i, "qt_raw_ms"]) and np.isnan(out.loc[i, "pr_interval_ms"])
        assert np.isnan(out.loc[i, "qtc_fridericia_ms"]) and np.isnan(out.loc[i, "qtc_formula_range_ms"])
    assert out.loc[1, "heart_rate_bpm"] == 400          # heart rate itself is unaffected
