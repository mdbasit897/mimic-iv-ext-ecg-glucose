"""
Checks against the published CSV. Skipped unless GLUCOECG_CSV points to the file.

    GLUCOECG_CSV=/path/to/mimiciv_ecg_glucose_aligned.csv pytest tests/
"""
import os

import numpy as np
import pandas as pd
import pytest

from glucoecg.config import ENV_VAR, PHYSIONET_COUNTS

CSV = os.environ.get(ENV_VAR, "")
pytestmark = pytest.mark.skipif(not CSV or not os.path.exists(CSV), reason=f"{ENV_VAR} not set")


@pytest.fixture(scope="module")
def df():
    from glucoecg.features import load_dataset
    return load_dataset(CSV, verbose=False)


def test_published_counts(df):
    c = PHYSIONET_COUNTS
    assert len(df) == c["records"]
    assert df["subject_id"].nunique() == c["patients"]
    assert df["ecg_study_id"].nunique() == c["ecg_studies"]
    for col in ("record_usability", "sqi_category", "intervention_status"):
        assert df[col].value_counts().to_dict() == c[col]
    assert df["split"].value_counts().to_dict() == c["split_records"]
    kept = df[df["record_usability"] != "EXCLUDE"]
    assert kept["split"].value_counts().to_dict() == c["split_records_no_exclude"]
    assert kept.groupby("split")["subject_id"].nunique().to_dict() == c["split_patients_no_exclude"]


def test_split_is_patient_level(df):
    assert (df.groupby("subject_id")["split"].nunique() == 1).all()


def test_prior_features_match_recomputation(df):
    rng = np.random.default_rng(0)
    rows = df.iloc[rng.choice(len(df), 1000, replace=False)]
    sub = df[df["subject_id"].isin(rows["subject_id"])]
    t = pd.to_datetime(sub["glucose_time"])
    for _, r in rows.iterrows():
        mine = sub[(sub["subject_id"] == r["subject_id"]) & (t < pd.Timestamp(r["glucose_time"]))]
        assert r["prior_n"] == len(mine)
        if len(mine):
            assert r["prior_mean"] == pytest.approx(mine["glucose_mg_dl"].mean())
            assert r["prior_max"] == pytest.approx(mine["glucose_mg_dl"].max())
