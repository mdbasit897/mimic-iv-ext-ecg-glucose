"""
Shared configuration: dataset location, expected checksum and published counts.

The dataset is MIMIC-IV-Ext-ECG-Glucose v1.0.0 on PhysioNet. The CSV is never
modified by this repository; every script reads it read-only.

Set the CSV location once per shell:

    export GLUCOECG_CSV=/path/to/mimiciv_ecg_glucose_aligned.csv

or pass ``--csv`` to any script.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

ENV_VAR = "GLUCOECG_CSV"
REPO_ROOT = Path(__file__).resolve().parents[1]

# SHA-256 of mimiciv_ecg_glucose_aligned.csv in PhysioNet v1.0.0
EXPECTED_SHA256 = "eaea161137acbf4cd913a431e0757206f50f87f99a82d0b6a2c67c1fcd9fe552"

# Counts stated in the PhysioNet v1.0.0 record; used by tests and sanity checks
PHYSIONET_COUNTS = {
    "records": 437_671,
    "patients": 131_771,
    "ecg_studies": 376_648,
    "record_usability": {"IDEAL": 4_364, "USABLE": 116_106, "CAUTION": 295_034, "EXCLUDE": 22_167},
    "sqi_category": {"GOOD": 372_024, "FAIR": 64_675, "POOR": 972},
    "intervention_status": {"CLEAN": 385_544, "FLAG_MODERATE_RISK": 26_112,
                            "FLAG_HIGH_RISK": 4_725, "EXCLUDE": 21_290},
    "split_records": {"TRAIN": 349_976, "VALIDATION": 44_718, "TEST": 42_977},
    "split_records_no_exclude": {"TRAIN": 332_241, "VALIDATION": 42_505, "TEST": 40_758},
    "split_patients_no_exclude": {"TRAIN": 105_261, "VALIDATION": 13_192, "TEST": 13_078},
}

DEFAULT_SEEDS = [42, 1, 2, 3, 4]


def default_csv() -> str:
    """CSV path from the GLUCOECG_CSV environment variable (empty string if unset)."""
    return os.environ.get(ENV_VAR, "")


def resolve_csv(path: str | None) -> Path:
    """Return an existing CSV path from ``--csv`` or the environment, or exit with a hint."""
    p = path or default_csv()
    if not p:
        sys.exit(f"No dataset path given. Pass --csv or set {ENV_VAR}=/path/to/mimiciv_ecg_glucose_aligned.csv")
    p = Path(p).expanduser()
    if not p.exists() and Path(str(p) + ".csv").exists():
        p = Path(str(p) + ".csv")
    if not p.exists():
        sys.exit(f"Dataset not found: {p}")
    return p


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def check_csv(path: Path, skip: bool = False) -> str:
    """Warn (not fail) if the CSV differs from the PhysioNet v1.0.0 file. Returns the hash."""
    if skip:
        return "not-checked"
    digest = sha256(path)
    if digest != EXPECTED_SHA256:
        warnings.warn(
            f"{path.name} SHA-256 {digest[:12]}... differs from PhysioNet v1.0.0 "
            f"({EXPECTED_SHA256[:12]}...). Results will not match the published numbers.",
            stacklevel=2,
        )
    return digest


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def write_run_info(out_dir: Path, csv_path: Path, csv_hash: str, extra: dict | None = None) -> Path:
    """Write run_info.json so every result file can be traced to code, data and settings."""
    import numpy, pandas, sklearn, xgboost  # noqa: E401  (versions only)

    info = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": Path(sys.argv[0]).name,
        "argv": sys.argv[1:],
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "csv": str(csv_path),
        "csv_sha256": csv_hash,
        "csv_matches_physionet_v1_0_0": csv_hash == EXPECTED_SHA256,
        "python": platform.python_version(),
        "packages": {"numpy": numpy.__version__, "pandas": pandas.__version__,
                     "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
    }
    try:
        import torch
        info["packages"]["torch"] = torch.__version__
    except ImportError:
        pass
    if extra:
        info.update(extra)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "run_info.json"
    path.write_text(json.dumps(info, indent=2, default=str))
    return path
