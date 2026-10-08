"""
glucoecg_dataset.py
════════════════════════════════════════════════════════════════════════════════
PyTorch Dataset for MIMIC-IV-Ext-ECG-Glucose v1.0.0 (leak-free inputs).

Every input tensor is built from glucoecg.features, so no target-derived,
whole-record, discharge-time or intervention column reaches a model:

  ecg_static       [15]  ECG intervals, axes, QTc (placeholder codes -> missing), z-scored
  ecg_binary       [13]  ECG plausibility and QTc-prolongation flags
  ecg_cat          [3]   hr_regime, temporal_relationship, alignment_quality (index, 0 = unknown)
  dynamics_feats   [3]   previous glucose, gap to it, previous change (z-scored, 0 when masked)
  dynamics_mask    [3]   1 = available, 0 = use a learned MASK embedding
  history_stats    [9]   statistics of strictly earlier glucose records + has-history flag
  clinical_scalar  [6]   age, offsets, sequence number, hours since admission / ICU admission
  clinical_cat     [3]   sqi_category, gender, during_icu_stay (index)
  target           [1]   glucose_mg_dl
  target_class     [1]   glycaemic class of the target (a label only, never an input)
  sample_weight    [1]   usability-tier x capped inverse-frequency class weight
  subject_id, labevent_id, waveform ([12, 5000] when use_waveform=True, else [0])

Changes from v1.0 (all removed inputs leaked the target or future information):
  * patient_stats (whole-record patient_mean_glucose etc.) -> history_stats from earlier records
  * glucose_delta / glucose_rate (computed from the target) -> previous-change dynamics
  * clinical_cat no longer contains glycemic_class, glucose_trajectory or intervention_status
  * clinical_scalar no longer contains icu_los_days, decoupling_risk_score, n_glucose_ecg_pairs
  * the only allowed target is glucose_mg_dl (glucose_z_score is built from whole-record stats)
  * combined sample weights are capped (v1.0 reached 228.9)

Normalisation statistics come from TRAIN only and are shared with VALIDATION / TEST.
Causal history is computed on the full CSV before split filtering, so a patient's
earlier records are always visible to later ones (all of a patient's records share a split).

Usage
    from glucoecg_dataset import build_dataloaders
    train_dl, val_dl, test_dl, train_ds = build_dataloaders("mimiciv_ecg_glucose_aligned.csv")
════════════════════════════════════════════════════════════════════════════════
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glucoecg.config import ENV_VAR, default_csv  # noqa: E402
from glucoecg.features import (ECG_BINARY_FEATURES, ECG_NUMERIC, TARGET, assert_no_leak,  # noqa: E402
                               load_dataset)

try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# COLUMN DEFINITIONS (all checked against glucoecg.features.FORBIDDEN_INPUTS)
# ─────────────────────────────────────────────────────────────────────────────
ECG_STATIC_COLS = list(ECG_NUMERIC)
ECG_BINARY_COLS = list(ECG_BINARY_FEATURES)
DYNAMICS_COLS = ["lag_ok", "gap_ok_hr", "prior_change"]
HISTORY_COLS = ["prior_n", "prior_mean", "prior_std", "prior_tir_pct", "prior_min", "prior_max",
                "prior_last", "prior_last_hr"]
CLINICAL_SCALAR_COLS = ["age", "abs_offset_minutes", "glucose_ecg_offset_minutes",
                        "glucose_seq_in_stay", "hours_since_admission", "hours_since_icu_admission"]
LOG_COLS = {"prior_n", "prior_last_hr", "glucose_seq_in_stay", "gap_ok_hr"}

ECG_CAT_COLS = {
    "hr_regime": ["BRADYCARDIA", "LOW_NORMAL", "NORMAL", "MILD_TACHY", "TACHYCARDIA"],
    "temporal_relationship": ["ECG_BEFORE_GLUCOSE", "ECG_AFTER_GLUCOSE", "SIMULTANEOUS"],
    "alignment_quality": ["TIGHT", "MODERATE", "LOOSE", "EXTENDED"],
}
CLINICAL_CAT_COLS = {
    "sqi_category": ["GOOD", "FAIR", "POOR"],
    "gender": ["M", "F"],
    "during_icu": [0.0, 1.0],
}
GLYCEMIC_CLASSES = ["EUGLYCEMIC", "HYPO", "HYPER", "SEVERE_HYPO", "SEVERE_HYPER"]

for _cols in (ECG_STATIC_COLS, ECG_BINARY_COLS, DYNAMICS_COLS, HISTORY_COLS, CLINICAL_SCALAR_COLS,
              list(ECG_CAT_COLS), list(CLINICAL_CAT_COLS)):
    assert_no_leak(_cols)

USABILITY_WEIGHTS: Dict[str, float] = {"IDEAL": 4.0, "USABLE": 2.0, "CAUTION": 0.5, "EXCLUDE": 0.0}
MAX_SAMPLE_WEIGHT = 20.0

ECG_LEADS, ECG_SAMPLING_RATE, ECG_DURATION_SEC = 12, 500, 10
ECG_NUM_SAMPLES = ECG_SAMPLING_RATE * ECG_DURATION_SEC


def _log(a: np.ndarray, cols) -> np.ndarray:
    a = a.copy()
    for j, c in enumerate(cols):
        if c in LOG_COLS:
            a[:, j] = np.sign(a[:, j]) * np.log1p(np.abs(a[:, j]))
    return a


class GlucoECGDataset(Dataset):
    """
    Parameters
    ----------
    csv_path : path to mimiciv_ecg_glucose_aligned.csv (ignored when ``frame`` is given)
    split : 'TRAIN', 'VALIDATION' or 'TEST'
    frame : optional DataFrame already returned by glucoecg.features.load_dataset
            (avoids re-reading the CSV for each split)
    physionet_root : parent of 'files/' in a local MIMIC-IV-ECG mirror (waveforms only)
    use_dynamics, use_history : include the masked dynamics / history groups
    use_waveform : load raw 12-lead waveforms (needs physionet_root and wfdb)
    exclude_caution : keep only IDEAL and USABLE records
    norm_stats : statistics from a TRAIN dataset (set automatically by from_train_stats)
    """

    def __init__(self, csv_path: Union[str, Path, None] = None, split: str = "TRAIN",
                 frame: Optional[pd.DataFrame] = None,
                 physionet_root: Optional[Union[str, Path]] = None, target: str = TARGET,
                 use_dynamics: bool = True, use_history: bool = True, use_waveform: bool = False,
                 exclude_caution: bool = False, cache_waveforms: bool = False,
                 norm_stats: Optional[dict] = None):
        super().__init__()
        if target != TARGET:
            raise ValueError(f"target must be '{TARGET}'; other glucose columns are derived from "
                             "whole-record patient statistics and leak the target")
        self.split = split.upper()
        assert self.split in ("TRAIN", "VALIDATION", "TEST")
        self.physionet_root = Path(physionet_root) if physionet_root else None
        self.use_dynamics, self.use_history = use_dynamics, use_history
        self.use_waveform = bool(use_waveform and self.physionet_root and WFDB_AVAILABLE)
        if use_waveform and not self.use_waveform:
            warnings.warn("use_waveform=True needs physionet_root and the wfdb package; "
                          "falling back to tabular-only mode.", stacklevel=2)
        self.cache_waveforms = cache_waveforms
        self._waveform_cache: Dict[str, np.ndarray] = {}

        full = frame if frame is not None else load_dataset(csv_path or default_csv(), verbose=False)
        keep = (full["split"] == self.split) & (full["record_usability"] != "EXCLUDE")
        if exclude_caution:
            keep &= full["record_usability"].isin(["IDEAL", "USABLE"])
        self.df = full[keep].reset_index(drop=True)

        self.norm_stats = norm_stats if norm_stats is not None else self._fit_norm_stats()
        self._build_arrays()
        self.sample_weights = self._compute_sample_weights()
        print(f"[GlucoECGDataset] {self.split} | {len(self.df):,} records | "
              f"{self.df['subject_id'].nunique():,} patients | "
              f"waveform={'ON' if self.use_waveform else 'OFF'}")

    # ── normalisation (TRAIN only) ────────────────────────────────────────────
    def _fit_norm_stats(self) -> dict:
        stats = {}
        for name, cols in [("ecg_static", ECG_STATIC_COLS), ("history", HISTORY_COLS),
                           ("clinical", CLINICAL_SCALAR_COLS), ("dynamics", DYNAMICS_COLS)]:
            a = _log(self.df[cols].astype(float).to_numpy(), cols)
            lo, hi = np.nanquantile(a, 0.005, axis=0), np.nanquantile(a, 0.995, axis=0)
            c = np.clip(a, lo, hi)
            mu, sd = np.nan_to_num(np.nanmean(c, axis=0)), np.nanstd(c, axis=0)
            sd[~(sd > 1e-6)] = 1.0
            stats[name] = (lo, hi, mu, sd)
        return stats

    def _scaled(self, name, cols) -> np.ndarray:
        lo, hi, mu, sd = self.norm_stats[name]
        a = _log(self.df[cols].astype(float).to_numpy(), cols)
        return (np.clip(a, lo, hi) - mu) / sd

    @staticmethod
    def _index(series: pd.Series, categories) -> np.ndarray:
        lookup = {c: i + 1 for i, c in enumerate(categories)}  # 0 = unknown
        return series.map(lookup).fillna(0).astype(np.int64).to_numpy()

    def _build_arrays(self):
        d = self.df
        f32 = lambda a: torch.from_numpy(np.nan_to_num(a, nan=0.0).astype(np.float32))  # noqa: E731
        self._t = {
            "ecg_static": f32(self._scaled("ecg_static", ECG_STATIC_COLS)),
            "ecg_binary": f32(d[ECG_BINARY_COLS].astype(float).to_numpy()),
            "ecg_cat": torch.from_numpy(np.column_stack(
                [self._index(d[c], cats) for c, cats in ECG_CAT_COLS.items()])),
            "clinical_scalar": f32(self._scaled("clinical", CLINICAL_SCALAR_COLS)),
            "clinical_cat": torch.from_numpy(np.column_stack(
                [self._index(d[c], cats) for c, cats in CLINICAL_CAT_COLS.items()])),
            "target": f32(d[[TARGET]].to_numpy(float)),
            "target_class": torch.from_numpy(self._index(d["glycemic_class"], GLYCEMIC_CLASSES)[:, None]),
            "subject_id": torch.from_numpy(d[["subject_id"]].to_numpy(np.int64)),
            "labevent_id": torch.from_numpy(d[["labevent_id"]].to_numpy(np.int64)),
        }
        hist = self._scaled("history", HISTORY_COLS)
        has = (d["prior_n"].fillna(0) > 0).to_numpy(float)[:, None]
        self._t["history_stats"] = f32(np.hstack([hist, has])) if self.use_history \
            else torch.zeros(len(d), len(HISTORY_COLS) + 1)
        dyn = self._scaled("dynamics", DYNAMICS_COLS)
        mask = (~np.isnan(d[DYNAMICS_COLS].astype(float).to_numpy())).astype(np.float32)
        if not self.use_dynamics:
            mask[:] = 0
        self._t["dynamics_feats"] = f32(np.clip(dyn, -5, 5) * mask)
        self._t["dynamics_mask"] = torch.from_numpy(mask)

    @classmethod
    def from_train_stats(cls, csv_path, split: str, train_dataset: "GlucoECGDataset", **kwargs):
        """Build VALIDATION / TEST with the TRAIN dataset's normalisation statistics."""
        return cls(csv_path=csv_path, split=split, norm_stats=train_dataset.norm_stats, **kwargs)

    def _compute_sample_weights(self) -> np.ndarray:
        """Usability weight x inverse-frequency glycaemic-class weight, capped and normalised to mean 1."""
        n = len(self.df)
        use_w = self.df["record_usability"].map(USABILITY_WEIGHTS).fillna(1.0).to_numpy(float)
        counts = self.df["glycemic_class"].value_counts()
        cls_w = self.df["glycemic_class"].map(lambda c: n / counts.get(c, 1)).to_numpy(float)
        cls_w /= cls_w.mean()
        w = np.minimum(use_w * cls_w, MAX_SAMPLE_WEIGHT)
        return (w / w.mean()).astype(np.float32)

    # ── waveforms ─────────────────────────────────────────────────────────────
    def _load_waveform(self, waveform_path: str) -> Tensor:
        if self.cache_waveforms and waveform_path in self._waveform_cache:
            return torch.from_numpy(self._waveform_cache[waveform_path])
        zero = torch.zeros(ECG_LEADS, ECG_NUM_SAMPLES)
        try:
            signal = wfdb.rdsamp(str(self.physionet_root / waveform_path))[0]
            if signal.shape[1] < ECG_LEADS:
                signal = np.pad(signal, ((0, 0), (0, ECG_LEADS - signal.shape[1])))
            signal = signal[:, :ECG_LEADS].T
            if signal.shape[1] != ECG_NUM_SAMPLES:
                from scipy.signal import resample
                signal = resample(signal, ECG_NUM_SAMPLES, axis=1)
            signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
            sd = signal.std(axis=1, keepdims=True)
            signal = ((signal - signal.mean(axis=1, keepdims=True)) / np.where(sd < 1e-8, 1, sd)).astype(np.float32)
            if self.cache_waveforms:
                self._waveform_cache[waveform_path] = signal
            return torch.from_numpy(signal)
        except Exception as e:  # missing or corrupt record
            warnings.warn(f"Failed to load waveform '{waveform_path}': {e}", stacklevel=2)
            return zero

    # ── Dataset interface ─────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        item = {k: v[idx] for k, v in self._t.items()}
        item["sample_weight"] = torch.tensor([self.sample_weights[idx]])
        item["waveform"] = (self._load_waveform(str(self.df.at[idx, "waveform_path"]))
                            if self.use_waveform else torch.zeros(0))
        return item

    @property
    def feature_dim(self) -> Dict[str, int]:
        return {k: int(self._t[k].shape[1]) for k in
                ["ecg_static", "ecg_binary", "ecg_cat", "dynamics_feats", "dynamics_mask",
                 "history_stats", "clinical_scalar", "clinical_cat"]} | {
                "waveform_leads": ECG_LEADS, "waveform_samples": ECG_NUM_SAMPLES}

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        return WeightedRandomSampler(torch.from_numpy(self.sample_weights).double(),
                                     num_samples=len(self.sample_weights), replacement=True)

    def class_distribution(self) -> pd.Series:
        return self.df["glycemic_class"].value_counts()

    def usability_distribution(self) -> pd.Series:
        return self.df["record_usability"].value_counts()

    def dynamics_availability(self) -> Dict[str, float]:
        return {c: float(self.df[c].notna().mean()) for c in DYNAMICS_COLS}


def build_dataloaders(csv_path: Union[str, Path, None] = None, batch_size: int = 32, num_workers: int = 4,
                      physionet_root: Optional[Union[str, Path]] = None, use_dynamics: bool = True,
                      use_history: bool = True, use_waveform: bool = False, exclude_caution: bool = False,
                      pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader, GlucoECGDataset]:
    """TRAIN / VALIDATION / TEST loaders sharing TRAIN normalisation; the CSV is read once."""
    frame = load_dataset(csv_path or default_csv())
    kw = dict(frame=frame, physionet_root=physionet_root, use_dynamics=use_dynamics,
              use_history=use_history, use_waveform=use_waveform, exclude_caution=exclude_caution)
    train_ds = GlucoECGDataset(split="TRAIN", **kw)
    val_ds = GlucoECGDataset(split="VALIDATION", norm_stats=train_ds.norm_stats, **kw)
    test_ds = GlucoECGDataset(split="TEST", norm_stats=train_ds.norm_stats, **kw)
    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=train_ds.get_weighted_sampler(),
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_dl, val_dl, test_dl, train_ds


def sanity_check(csv_path: Union[str, Path, None] = None) -> None:
    """Structural check of the dataset and loaders (no GPU or waveforms needed)."""
    print("\n" + "=" * 65 + "\nGLUCOECG DATASET: SANITY CHECK\n" + "=" * 65)
    train_dl, val_dl, test_dl, train_ds = build_dataloaders(csv_path, batch_size=32, num_workers=0,
                                                            pin_memory=False)
    print("\nFeature dimensions:")
    for k, v in train_ds.feature_dim.items():
        print(f"  {k:<20}: {v}")
    print("\nUsability (TRAIN):\n" + train_ds.usability_distribution().to_string())
    print("\nDynamics availability (TRAIN):")
    for c, p in train_ds.dynamics_availability().items():
        print(f"  {c:<14}: {p * 100:.1f}%")
    batch = next(iter(train_dl))
    print("\nOne TRAIN batch:")
    for k, v in batch.items():
        print(f"  {k:<16} shape={str(tuple(v.shape)):<12} dtype={v.dtype}")
    sw = train_ds.sample_weights
    print(f"\nSample weights: mean {sw.mean():.2f}  min {sw.min():.3f}  max {sw.max():.2f} "
          f"(cap {MAX_SAMPLE_WEIGHT} before normalisation)")
    print(f"Batches: TRAIN {len(train_dl):,} | VALIDATION {len(val_dl):,} | TEST {len(test_dl):,}")
    print("\nSanity check passed.\n")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else default_csv()
    if not path:
        sys.exit(f"Usage: python glucoecg_dataset.py /path/to/csv   (or set {ENV_VAR})")
    sanity_check(path)
