"""
load_ecg_waveform.py
════════════════════════════════════════════════════════════════════════
Supplementary Code S3 — ECG Waveform Loader
MIMIC-IV-Ext-ECG-Glucose Dataset

Usage:
    python3 load_ecg_waveform.py

Requirements:
    pip install wfdb pandas numpy matplotlib --break-system-packages
════════════════════════════════════════════════════════════════════════
"""

import os

import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ─── PATHS (environment variables MIMIC_ECG_ROOT and GLUCOECG_CSV, or edit) ──────────────────────────────────────────────────
# MIMIC_ECG_ROOT: parent directory of the 'files/' folder in MIMIC-IV-ECG v1.0
#   PhysioNet download: https://physionet.org/content/mimic-iv-ecg/1.0/
#   NOTE: waveform_path in the CSV already contains 'files/pXXXX/...' prefix,
#         so MIMIC_ECG_ROOT must point to the PARENT of 'files/', not to 'files/' itself.
#   Example (local):      '/data/mimic-iv-ecg/1.0'
#   Example (PhysioNet):  set pn_dir in wfdb.rdrecord instead (see comments below)
MIMIC_ECG_ROOT = os.environ.get('MIMIC_ECG_ROOT', '/path/to/mimic-iv-ecg/1.0')

# CSV_PATH: path to mimiciv_ecg_glucose_aligned.csv (this dataset)
#   PhysioNet download: https://physionet.org/content/mimic-iv-ext-ecg-glucose/
CSV_PATH       = os.environ.get('GLUCOECG_CSV', '/path/to/mimiciv_ecg_glucose_aligned.csv')
# ─────────────────────────────────────────────────────────────────────────────

LEAD_NAMES = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']


def load_ecg_waveform(waveform_path, root=MIMIC_ECG_ROOT):
    """Load a 12-lead ECG waveform from MIMIC-IV-ECG.

    Parameters
    ----------
    waveform_path : str
        Relative record path from the dataset column 'waveform_path'.
        Example: 'files/p1000/p10000032/s40689238/40689238'
    root : str
        Root directory of the MIMIC-IV-ECG v1.0 file system.

    Returns
    -------
    signal_norm : np.ndarray, shape (5000, 12), dtype float32
        Per-lead z-score normalised ECG signal (500 Hz, 10 seconds).
    record : wfdb.Record
        Raw wfdb record object (access record.p_signal for raw mV values).
    """
    record_path = f'{root}/{waveform_path}'
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal          # shape: (5000, 12), physical units (mV)

    # Per-lead z-score normalisation
    mean = np.nanmean(signal, axis=0, keepdims=True)
    std  = np.nanstd(signal,  axis=0, keepdims=True) + 1e-8
    signal_norm = (signal - mean) / std

    return signal_norm.astype(np.float32), record   # (5000, 12) float32


def plot_ecg(signal_norm, record, row, out_path=None):
    """Plot all 12 leads as a standard ECG strip."""
    fs   = record.fs        # sampling frequency (500 Hz)
    time = np.arange(signal_norm.shape[0]) / fs   # seconds

    fig, axes = plt.subplots(12, 1, figsize=(14, 14), sharex=True)
    fig.suptitle(
        f"12-Lead ECG — subject {row['subject_id']} | stay {row['stay_id']}\n"
        f"ECG time: {row['ecg_time']} | Glucose: {row['glucose_mg_dl']:.1f} mg/dL "
        f"({row['temporal_relationship']}) | SQI: {row.get('sqi_score', 'N/A')} "
        f"({row.get('sqi_category', '')})",
        fontsize=10, y=1.01
    )

    for i, (ax, lead) in enumerate(zip(axes, LEAD_NAMES)):
        ax.plot(time, signal_norm[:, i], lw=0.6, color='#1f4e79')
        ax.set_ylabel(lead, fontsize=9, rotation=0, labelpad=18, va='center')
        ax.set_ylim(-4, 4)
        ax.axhline(0, color='gray', lw=0.4, ls='--')
        ax.tick_params(labelleft=False, left=False)
        ax.spines[['top','right','left']].set_visible(False)

    axes[-1].set_xlabel('Time (seconds)')
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved → {out_path}")
    plt.show()
    plt.close()


def main():
    print("=" * 60)
    print("  Supplementary Code S3 — ECG Waveform Loader")
    print("  MIMIC-IV-Ext-ECG-Glucose Dataset")
    print("=" * 60)
    print()

    # ── 1. Load dataset CSV ───────────────────────────────────────────
    print(f"Loading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    print(f"  {len(df):,} records | {df['subject_id'].nunique():,} patients")
    print()

    # ── 2. Filter to IDEAL records in TEST split ──────────────────────
    # Use IDEAL TEST records — highest quality, ECG before glucose
    ideal_test = df[
        (df['record_usability'] == 'IDEAL') &
        (df['split'] == 'TEST') &
        (df['waveform_path'].notna())
    ].reset_index(drop=True)
    print(f"IDEAL TEST records with waveform_path: {len(ideal_test):,}")
    print()

    # ── 3. Load first 3 waveforms as examples ────────────────────────
    n_examples = min(3, len(ideal_test))
    results = []

    for i in range(n_examples):
        row = ideal_test.iloc[i]
        print(f"[{i+1}/{n_examples}] Loading waveform for subject {row['subject_id']}")
        print(f"  waveform_path : {row['waveform_path']}")
        print(f"  glucose_mg_dl : {row['glucose_mg_dl']:.1f} mg/dL")
        print(f"  ecg_time      : {row['ecg_time']}")
        print(f"  sqi_score     : {row.get('sqi_score', 'N/A')} ({row.get('sqi_category', '')})")
        print(f"  offset        : {row.get('abs_offset_minutes', 'N/A')} min")

        try:
            signal_norm, record = load_ecg_waveform(row['waveform_path'])

            # Verify shape and dtype
            assert signal_norm.shape == (5000, 12), \
                f"Unexpected shape: {signal_norm.shape}"
            assert signal_norm.dtype == np.float32, \
                f"Unexpected dtype: {signal_norm.dtype}"

            print(f"  ✅ Loaded successfully")
            print(f"     shape : {signal_norm.shape}")
            print(f"     dtype : {signal_norm.dtype}")
            print(f"     range : [{signal_norm.min():.2f}, {signal_norm.max():.2f}] (z-scored)")
            print(f"     fs    : {record.fs} Hz | duration: {signal_norm.shape[0]/record.fs:.0f}s")
            print(f"     leads : {record.sig_name}")

            # Save plot
            out_path = f"ecg_example_{i+1}_subject{row['subject_id']}.png"
            plot_ecg(signal_norm, record, row, out_path=out_path)

            results.append({
                'subject_id':   row['subject_id'],
                'glucose_mg_dl':row['glucose_mg_dl'],
                'sqi_score':    row.get('sqi_score'),
                'shape':        signal_norm.shape,
                'dtype':        str(signal_norm.dtype),
                'status':       'OK'
            })

        except FileNotFoundError:
            print(f"  ❌ File not found — check MIMIC_ECG_ROOT path")
            print(f"     Tried: {MIMIC_ECG_ROOT}/{row['waveform_path']}")
            results.append({'subject_id': row['subject_id'], 'status': 'FILE_NOT_FOUND'})
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({'subject_id': row['subject_id'], 'status': f'ERROR: {e}'})

        print()

    # ── 4. Summary ────────────────────────────────────────────────────
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    ok = sum(1 for r in results if r['status'] == 'OK')
    print(f"  Loaded successfully : {ok}/{n_examples}")
    for r in results:
        status_icon = '✅' if r['status'] == 'OK' else '❌'
        print(f"  {status_icon} Subject {r['subject_id']}: {r['status']}")

    if ok == 0:
        print()
        print("  TROUBLESHOOTING:")
        print("  1. Check MIMIC_ECG_ROOT — should point to the directory")
        print("     containing the 'files/' folder, e.g.:")
        print("     /path/to/mimic-iv-ecg/1.0/files/p100/p10000032/...")
        print("     Set MIMIC_ECG_ROOT to the PARENT of 'files/' (not to 'files/' itself)")
        print("  2. Verify wfdb is installed: pip install wfdb")
        print("  3. Check that .hea and .dat files are both present")


if __name__ == '__main__':
    main()