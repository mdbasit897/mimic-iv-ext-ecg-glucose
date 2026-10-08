"""
q3_sensitivity_analysis.py
════════════════════════════════════════════════════════════════════════════════
Q3 Pharmacokinetic Window Sensitivity Analysis

Tests whether shifting the fast-acting insulin effect window from ±120 min
(original) to ±90 min (tighter) or ±150 min (broader) materially changes:
  1. The intervention_status tier distribution (CLEAN / FLAG_MODERATE /
     FLAG_HIGH_RISK / EXCLUDE) — i.e. how many records get re-classified
  2. Leak-free XGBoost MAE (ECG + causal history, glucoecg.features) on the
     TEST IDEAL+USABLE subset of each variant, and on the TEST rows that are
     IDEAL+USABLE in all three variants (identical rows, comparable MAE)

Produces:
  → Supplementary Table S[X]: PK Window Sensitivity Analysis (3-column comparison)
  → supplementary_table_Q3_sensitivity.csv
  → supplementary_figure_Q3_sensitivity.png

INPUTS REQUIRED (export from BigQuery after running each SQL):
  --orig   : CSV exported from the published dataset (PhysioNet v1.0.0)
  --s90    : CSV exported from sql/pipeline_sensitivity_a_90min.sql (±90 min variant)
  --s150   : CSV exported from sql/pipeline_sensitivity_b_150min.sql (±150 min variant)
  --out    : Output directory for tables and figures

Usage:
  python validation/q3_sensitivity_analysis.py \\
    --orig  /path/to/original_dataset.csv \\
    --s90   /path/to/sensitivity_A_90min.csv \\
    --s150  /path/to/sensitivity_B_150min.csv \\
    --out   ./supp_results

════════════════════════════════════════════════════════════════════════════════
"""

import argparse
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glucoecg.features import TARGET, load_dataset as load_clean  # noqa: E402
from glucoecg.metrics import regression_metrics  # noqa: E402
from glucoecg.modeling import train_feature_set  # noqa: E402

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT PATHS — override with CLI arguments
# ─────────────────────────────────────────────────────────────────────────────
ORIG_CSV  = os.environ.get("GLUCOECG_CSV", "")
S90_CSV   = "./sensitivity_A_90min.csv"
S150_CSV  = "./sensitivity_B_150min.csv"
OUT_DIR   = "./supp_results"


plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# FEATURES: leak-free inputs from glucoecg.features (FEATURE_SETS["ecg_history"])
MAIN_FEATURE_SET = "ecg_history"

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(csv_path: str, label: str) -> pd.DataFrame:
    """Read a pipeline export with the shared cleaning and causal history features."""
    print(f"\nLoading [{label}]: {csv_path}")
    return load_clean(csv_path)


def tier_distribution(df: pd.DataFrame) -> dict:
    """Count intervention_status tiers and record_usability tiers."""
    counts = df["intervention_status"].value_counts().to_dict()
    total  = len(df)
    return {
        k: {"n": v, "pct": v / total * 100}
        for k, v in counts.items()
    }


def usability_distribution(df: pd.DataFrame) -> dict:
    counts = df["record_usability"].value_counts().to_dict()
    total  = len(df)
    return {
        k: {"n": v, "pct": v / total * 100}
        for k, v in counts.items()
    }


def compute_metrics(y_true, y_pred) -> dict:
    return regression_metrics(y_true, y_pred)


def train_and_evaluate(df: pd.DataFrame, label: str, common_ids=None) -> dict:
    """Leak-free XGBoost on IDEAL+USABLE TRAIN; metrics on this variant's IDEAL+USABLE
    TEST rows, plus (if given) on the labevent_ids that are IDEAL+USABLE TEST in every variant."""
    _, pred = train_feature_set(df, MAIN_FEATURE_SET, "iu", seed=42)
    y = df[TARGET].to_numpy(float)
    test = (df["split"] == "TEST").to_numpy() & df["record_usability"].isin(["IDEAL", "USABLE"]).to_numpy()
    m = compute_metrics(y[test], pred[test])
    if common_ids is not None:
        c = df["labevent_id"].isin(common_ids).to_numpy() & test
        mc = compute_metrics(y[c], pred[c])
        m.update({"mae_common": mc["mae"], "n_common": mc["n"]})
    print(f"  [{label}] MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}  R²={m['r2']:.3f}  n={m['n']:,}"
          + (f" | identical rows: MAE={m['mae_common']:.2f} n={m['n_common']:,}" if common_ids is not None else ""))
    return m


# ─────────────────────────────────────────────────────────────────────────────
# TIER SHIFT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def compute_tier_shifts(df_orig: pd.DataFrame, df_var: pd.DataFrame,
                        var_label: str) -> dict:
    """
    For each record present in both datasets (matched on labevent_id),
    count how many changed
    intervention_status tier and record_usability tier.
    """
    print(f"\n  Computing tier shifts: Original → {var_label}")

    # labevent_id identifies one glucose result and is unique per row in every
    # variant. The v1.0 key (subject_id, stay_id, ecg_time, glucose_time) is not
    # unique: two assays of one draw share it, so the merge duplicated rows and
    # reported spurious tier changes (308 when a file was compared with itself).
    available_keys = ["labevent_id"]
    for d in (df_orig, df_var):
        assert d["labevent_id"].is_unique, "labevent_id must be unique per row"
    print(f"    Join key columns: {available_keys}")

    cols = available_keys + ["intervention_status", "record_usability", "ecg_study_id"]
    merged = df_orig[cols].merge(df_var[cols], on=available_keys, suffixes=("_orig", "_var"))
    n_matched = len(merged)
    print(f"    Matched records: {n_matched:,}")

    # The pipeline picks the nearest ECG with ROW_NUMBER() and no tie-breaker, so when
    # two ECGs are equally close BigQuery may pick a different one on each run. Such
    # records can change SQI (and therefore record_usability) for reasons unrelated to
    # the insulin window, so usability shifts are counted only on records whose ECG is
    # the same in both files. intervention_status does not depend on the ECG.
    same_ecg = merged["ecg_study_id_orig"] == merged["ecg_study_id_var"]
    n_ecg_diff = int((~same_ecg).sum())
    print(f"    Records paired with a different (equally near) ECG: {n_ecg_diff:,}")

    # intervention_status shifts (all matched records)
    int_changed = (merged["intervention_status_orig"] != merged["intervention_status_var"]).sum()
    int_pct     = int_changed / n_matched * 100 if n_matched > 0 else 0

    # record_usability shifts (records with the same ECG in both files)
    same = merged[same_ecg]
    n_same = len(same)
    use_changed = (same["record_usability_orig"] != same["record_usability_var"]).sum()
    use_pct     = use_changed / n_same * 100 if n_same > 0 else 0
    use_changed_all = int((merged["record_usability_orig"] != merged["record_usability_var"]).sum())

    print(f"    intervention_status changed: {int_changed:,} ({int_pct:.2f}%)")
    print(f"    record_usability changed   : {use_changed:,} ({use_pct:.2f}% of {n_same:,} same-ECG records; "
          f"{use_changed_all:,} including ECG-tie records)")
    merged = same

    # Cross-tabulation of usability changes
    cross = pd.crosstab(
        merged["record_usability_orig"],
        merged["record_usability_var"],
        rownames=["Original"],
        colnames=[var_label]
    )
    print(f"\n    Usability cross-tabulation:")
    print(cross.to_string())

    return {
        "n_matched"            : n_matched,
        "intervention_changed" : int_changed,
        "intervention_pct"     : int_pct,
        "usability_changed"    : use_changed,
        "usability_pct"        : use_pct,
        "usability_changed_all": use_changed_all,
        "n_same_ecg"           : n_same,
        "n_ecg_diff"           : n_ecg_diff,
        "cross_tab"            : cross,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SUPPLEMENTARY TABLE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_supplementary_table(
    df_orig, df_90, df_150,
    tiers_orig, tiers_90, tiers_150,
    use_orig, use_90, use_150,
    metrics_orig, metrics_90, metrics_150,
    shifts_90, shifts_150,
    out_dir: Path
) -> pd.DataFrame:

    tier_order = ["CLEAN", "FLAG_MODERATE_RISK", "FLAG_HIGH_RISK", "EXCLUDE"]
    use_order  = ["IDEAL", "USABLE", "CAUTION", "EXCLUDE"]

    rows = []

    # Section 1: Dataset totals
    rows.append({"Section": "Dataset Size", "Metric": "Total records",
                 "Original (±120 min)": f"{len(df_orig):,}",
                 "Variant A (±90 min)": f"{len(df_90):,}",
                 "Variant B (±150 min)": f"{len(df_150):,}"})

    # Section 2: Intervention status distribution
    for tier in tier_order:
        o = tiers_orig.get(tier, {"n": 0, "pct": 0})
        a = tiers_90.get(tier, {"n": 0, "pct": 0})
        b = tiers_150.get(tier, {"n": 0, "pct": 0})
        rows.append({
            "Section" : "Intervention Status",
            "Metric"  : tier,
            "Original (±120 min)": f"{o['n']:,} ({o['pct']:.1f}%)",
            "Variant A (±90 min)": f"{a['n']:,} ({a['pct']:.1f}%)",
            "Variant B (±150 min)": f"{b['n']:,} ({b['pct']:.1f}%)",
        })

    # Section 3: Usability distribution
    for tier in use_order:
        o = use_orig.get(tier, {"n": 0, "pct": 0})
        a = use_90.get(tier, {"n": 0, "pct": 0})
        b = use_150.get(tier, {"n": 0, "pct": 0})
        rows.append({
            "Section" : "Record Usability",
            "Metric"  : tier,
            "Original (±120 min)": f"{o['n']:,} ({o['pct']:.1f}%)",
            "Variant A (±90 min)": f"{a['n']:,} ({a['pct']:.1f}%)",
            "Variant B (±150 min)": f"{b['n']:,} ({b['pct']:.1f}%)",
        })

    # Section 4: Tier shifts vs original
    rows.append({
        "Section" : "Tier Stability",
        "Metric"  : "intervention_status records changed vs. original",
        "Original (±120 min)": "—",
        "Variant A (±90 min)":
            f"{shifts_90['intervention_changed']:,} ({shifts_90['intervention_pct']:.2f}%)",
        "Variant B (±150 min)":
            f"{shifts_150['intervention_changed']:,} ({shifts_150['intervention_pct']:.2f}%)",
    })
    rows.append({
        "Section" : "Tier Stability",
        "Metric"  : "records paired with a different, equally near ECG (excluded below)",
        "Original (±120 min)": "—",
        "Variant A (±90 min)": f"{shifts_90['n_ecg_diff']:,}",
        "Variant B (±150 min)": f"{shifts_150['n_ecg_diff']:,}",
    })
    rows.append({
        "Section" : "Tier Stability",
        "Metric"  : "record_usability records changed vs. original (same-ECG records)",
        "Original (±120 min)": "—",
        "Variant A (±90 min)":
            f"{shifts_90['usability_changed']:,} ({shifts_90['usability_pct']:.2f}%)",
        "Variant B (±150 min)":
            f"{shifts_150['usability_changed']:,} ({shifts_150['usability_pct']:.2f}%)",
    })

    # Section 5: XGBoost model performance on IDEAL+USABLE TEST
    for metric_key, label in [("mae", "MAE (mg/dL)"), ("rmse", "RMSE (mg/dL)"),
                               ("r2", "R²"), ("n", "Test N (IDEAL+USABLE)"),
                               ("mae_common", "MAE on identical rows (mg/dL)"),
                               ("n_common", "Identical-row N")]:
        rows.append({
            "Section" : "Model Performance (leak-free XGBoost, ECG + causal history)",
            "Metric"  : label,
            "Original (±120 min)": (f"{metrics_orig[metric_key]:,}"
                                    if metric_key in ("n", "n_common")
                                    else f"{metrics_orig[metric_key]:.2f}" if metric_key != "r2"
                                    else f"{metrics_orig[metric_key]:.3f}"),
            "Variant A (±90 min)": (f"{metrics_90[metric_key]:,}"
                                    if metric_key in ("n", "n_common")
                                    else f"{metrics_90[metric_key]:.2f}" if metric_key != "r2"
                                    else f"{metrics_90[metric_key]:.3f}"),
            "Variant B (±150 min)": (f"{metrics_150[metric_key]:,}"
                                     if metric_key in ("n", "n_common")
                                     else f"{metrics_150[metric_key]:.2f}" if metric_key != "r2"
                                     else f"{metrics_150[metric_key]:.3f}"),
        })

    df_table = pd.DataFrame(rows)
    csv_path = out_dir / "supplementary_table_Q3_sensitivity.csv"
    df_table.to_csv(csv_path, index=False)
    print(f"\n  Supplementary table saved → {csv_path}")
    return df_table


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def plot_sensitivity_figure(
    tiers_orig, tiers_90, tiers_150,
    metrics_orig, metrics_90, metrics_150,
    out_dir: Path
) -> None:

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("white")

    tier_labels = ["CLEAN", "FLAG_MODERATE_RISK", "FLAG_HIGH_RISK", "EXCLUDE"]
    x = np.arange(len(tier_labels))
    width = 0.25
    colors = ["#0072B2", "#E69F00", "#009E73"]

    # Panel A — Intervention status tier counts
    ax = axes[0]
    vals_orig = [tiers_orig.get(t, {"pct": 0})["pct"] for t in tier_labels]
    vals_90   = [tiers_90.get(t,   {"pct": 0})["pct"] for t in tier_labels]
    vals_150  = [tiers_150.get(t,  {"pct": 0})["pct"] for t in tier_labels]

    ax.bar(x - width, vals_orig, width, label="±120 min (original)",
           color=colors[0], alpha=0.85, edgecolor="white")
    ax.bar(x,          vals_90,  width, label="±90 min (Variant A)",
           color=colors[1], alpha=0.85, edgecolor="white")
    ax.bar(x + width,  vals_150, width, label="±150 min (Variant B)",
           color=colors[2], alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(tier_labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Records (%)")
    ax.set_title("A  Intervention Status Distribution\nby PK Window Variant",
                 fontweight="bold")
    ax.legend(fontsize=8)

    # Panel B — MAE comparison bar chart
    ax2 = axes[1]
    windows = ["±120 min\n(Original)", "±90 min\n(Variant A)", "±150 min\n(Variant B)"]
    maes    = [metrics_orig["mae_common"], metrics_90["mae_common"], metrics_150["mae_common"]]  # identical rows
    bars    = ax2.bar(windows, maes, color=colors, alpha=0.85, edgecolor="white", width=0.4)
    for bar, val in zip(bars, maes):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.05,
                 f"{val:.2f}", ha="center", va="bottom", fontweight="bold")
    ax2.set_ylabel("MAE (mg/dL)")
    ax2.set_ylim(0, max(maes) * 1.2)
    ax2.set_title("B  Leak-free XGBoost MAE, identical test rows\nby PK Window Variant",
                  fontweight="bold")

    fig.suptitle(
        "Supplementary Figure: Sensitivity of Tier Assignment and Model Performance\n"
        "to Fast-Acting Insulin PK Window",
        fontweight="bold", y=1.02
    )
    plt.tight_layout()
    out_path = out_dir / "supplementary_figure_Q3_sensitivity.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Sensitivity figure saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TEXT
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(
    tiers_orig, tiers_90, tiers_150,
    use_orig, use_90, use_150,
    metrics_orig, metrics_90, metrics_150,
    shifts_90, shifts_150,
    out_dir: Path
) -> None:

    sep  = "═" * 72
    sep2 = "─" * 72

    # MAE differences
    delta_90  = metrics_90["mae_common"]  - metrics_orig["mae_common"]  # identical rows
    delta_150 = metrics_150["mae_common"] - metrics_orig["mae_common"]  # identical rows
    sign_90   = "+" if delta_90  >= 0 else ""
    sign_150  = "+" if delta_150 >= 0 else ""

    lines = [
        sep,
        "Q3 SENSITIVITY ANALYSIS — COMPLETE RESULTS",
        "PK Window: Fast-Acting Insulin ±120 min vs ±90 min vs ±150 min",
        sep, "",

        "TIER DISTRIBUTION (intervention_status)",
        sep2,
        f"  {'Tier':<25s} {'Original ±120':>18s} {'Variant A ±90':>18s} {'Variant B ±150':>18s}",
        sep2,
    ]
    for tier in ["CLEAN", "FLAG_MODERATE_RISK", "FLAG_HIGH_RISK", "EXCLUDE"]:
        o = tiers_orig.get(tier, {"n": 0, "pct": 0})
        a = tiers_90.get(tier,   {"n": 0, "pct": 0})
        b = tiers_150.get(tier,  {"n": 0, "pct": 0})
        lines.append(
            f"  {tier:<25s} {o['n']:>9,} ({o['pct']:5.1f}%)"
            f"  {a['n']:>9,} ({a['pct']:5.1f}%)"
            f"  {b['n']:>9,} ({b['pct']:5.1f}%)"
        )
    lines += ["",
        "TIER STABILITY (records that changed tier vs. original)",
        sep2,
        f"  intervention_status changed:",
        f"    Variant A (±90 min) : {shifts_90['intervention_changed']:,} "
        f"({shifts_90['intervention_pct']:.2f}% of matched records)",
        f"    Variant B (±150 min): {shifts_150['intervention_changed']:,} "
        f"({shifts_150['intervention_pct']:.2f}% of matched records)",
        f"",
        f"  record_usability changed (records paired with the same ECG in both files):",
        f"    Variant A (±90 min) : {shifts_90['usability_changed']:,} "
        f"({shifts_90['usability_pct']:.2f}% of {shifts_90['n_same_ecg']:,})",
        f"    Variant B (±150 min): {shifts_150['usability_changed']:,} "
        f"({shifts_150['usability_pct']:.2f}% of {shifts_150['n_same_ecg']:,})",
        f"",
        f"  Records paired with a different, equally near ECG (BigQuery tie-breaking,",
        f"  unrelated to the insulin window; excluded from the usability and MAE comparisons):",
        f"    Variant A: {shifts_90['n_ecg_diff']:,}   Variant B: {shifts_150['n_ecg_diff']:,}",
        "",
        "MODEL PERFORMANCE (XGBoost IDEAL+USABLE — TEST SET)",
        sep2,
        f"  {'Metric':<12s} {'Original ±120':>16s} {'Variant A ±90':>16s} {'Variant B ±150':>16s}",
        sep2,
        f"  {'MAE (mg/dL)':<12s} {metrics_orig['mae']:>16.2f} "
        f"{metrics_90['mae']:>16.2f} {metrics_150['mae']:>16.2f}",
        f"  {'RMSE (mg/dL)':<12s} {metrics_orig['rmse']:>16.2f} "
        f"{metrics_90['rmse']:>16.2f} {metrics_150['rmse']:>16.2f}",
        f"  {'R²':<12s} {metrics_orig['r2']:>16.3f} "
        f"{metrics_90['r2']:>16.3f} {metrics_150['r2']:>16.3f}",
        f"  {'Test N':<12s} {metrics_orig['n']:>16,} "
        f"{metrics_90['n']:>16,} {metrics_150['n']:>16,}",
        f"  {'MAE, same rows':<12s} {metrics_orig['mae_common']:>16.2f} "
        f"{metrics_90['mae_common']:>16.2f} {metrics_150['mae_common']:>16.2f}"
        f"   (n = {metrics_orig['n_common']:,} TEST rows IDEAL+USABLE in all variants)",
        "",
        f"  MAE change vs. original:",
        f"    Variant A (±90 min) : {sign_90}{delta_90:.2f} mg/dL",
        f"    Variant B (±150 min): {sign_150}{delta_150:.2f} mg/dL",
        "",
        "MANUSCRIPT PROSE TO PASTE (seventh limitation placeholder [X])",
        sep2,
        "",
        f"  A sensitivity analysis evaluating alternative pharmacokinetic",
        f"  window configurations (±90 min and ±150 min) for fast-acting",
        f"  insulin demonstrated that tier re-classification affected",
        f"  {shifts_90['usability_pct']:.1f}% of records under the tighter window",
        f"  and {shifts_150['usability_pct']:.1f}% under the broader window,",
        f"  with corresponding MAE changes of {sign_90}{delta_90:.2f} mg/dL",
        f"  and {sign_150}{delta_150:.2f} mg/dL respectively (see",
        f"  Supplementary Table S[X]).",
        "",
        sep,
        "All output files saved to: " + str(out_dir.resolve()),
        sep,
    ]

    out_path = out_dir / "q3_sensitivity_summary.txt"
    out_path.write_text("\n".join(lines))
    print(f"\n  Summary saved → {out_path}")
    print()
    print("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Q3 PK window sensitivity analysis — three BigQuery CSVs in, "
                    "one supplementary table + figure out"
    )
    parser.add_argument("--orig",  default=ORIG_CSV,
                        help="Original dataset CSV (±120 min)")
    parser.add_argument("--s90",   default=S90_CSV,
                        help="Sensitivity Variant A CSV (±90 min)")
    parser.add_argument("--s150",  default=S150_CSV,
                        help="Sensitivity Variant B CSV (±150 min)")
    parser.add_argument("--out",   default=OUT_DIR,
                        help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Q3 PK WINDOW SENSITIVITY ANALYSIS")
    print("MIMIC-IV-Ext-ECG-Glucose Dataset")
    print("=" * 60)

    # ── Load all three datasets ───────────────────────────────────────────────
    df_orig = load_dataset(args.orig, "Original ±120 min")
    df_90   = load_dataset(args.s90,  "Variant A ±90 min")
    df_150  = load_dataset(args.s150, "Variant B ±150 min")

    # ── Tier distributions ────────────────────────────────────────────────────
    print("\n── Intervention Status Distributions ──")
    tiers_orig = tier_distribution(df_orig)
    tiers_90   = tier_distribution(df_90)
    tiers_150  = tier_distribution(df_150)

    use_orig = usability_distribution(df_orig)
    use_90   = usability_distribution(df_90)
    use_150  = usability_distribution(df_150)

    # ── Tier shifts (matched record comparison) ───────────────────────────────
    print("\n── Tier Shift Analysis ──")
    shifts_90  = compute_tier_shifts(df_orig, df_90,  "Variant A ±90 min")
    shifts_150 = compute_tier_shifts(df_orig, df_150, "Variant B ±150 min")

    # ── XGBoost performance for each variant ──────────────────────────────────
    print("\n── Model Training and Evaluation (leak-free) ──")
    iu_test = lambda d: set(d.loc[(d["split"] == "TEST") &  # noqa: E731
                                  d["record_usability"].isin(["IDEAL", "USABLE"]), "labevent_id"])
    common = iu_test(df_orig) & iu_test(df_90) & iu_test(df_150)
    # keep only rows paired with the same ECG in all three files (see compute_tier_shifts)
    ecg = lambda d: d.set_index("labevent_id")["ecg_study_id"]  # noqa: E731
    e0, e90, e150 = ecg(df_orig), ecg(df_90), ecg(df_150)
    same_all = set(e0.index[(e0 == e90.reindex(e0.index)) & (e0 == e150.reindex(e0.index))])
    n_before = len(common)
    common &= same_all
    print(f"  TEST rows IDEAL+USABLE in all three variants: {n_before:,}; "
          f"with the same ECG in all three: {len(common):,}")
    print("  [1/3] Original (±120 min)...")
    metrics_orig = train_and_evaluate(df_orig, "Original ±120 min", common)
    print("  [2/3] Variant A (±90 min)...")
    metrics_90   = train_and_evaluate(df_90,   "Variant A ±90 min", common)
    print("  [3/3] Variant B (±150 min)...")
    metrics_150  = train_and_evaluate(df_150,  "Variant B ±150 min", common)

    # ── Build supplementary table ─────────────────────────────────────────────
    print("\n── Building Supplementary Table ──")
    build_supplementary_table(
        df_orig, df_90, df_150,
        tiers_orig, tiers_90, tiers_150,
        use_orig, use_90, use_150,
        metrics_orig, metrics_90, metrics_150,
        shifts_90, shifts_150,
        out_dir
    )

    # ── Figure ────────────────────────────────────────────────────────────────
    plot_sensitivity_figure(
        tiers_orig, tiers_90, tiers_150,
        metrics_orig, metrics_90, metrics_150,
        out_dir
    )

    # ── Summary + manuscript prose ────────────────────────────────────────────
    write_summary(
        tiers_orig, tiers_90, tiers_150,
        use_orig, use_90, use_150,
        metrics_orig, metrics_90, metrics_150,
        shifts_90, shifts_150,
        out_dir
    )

    print(f"\n{'═'*60}")
    print("Output files:")
    print("  supplementary_table_Q3_sensitivity.csv   ← Supplementary Table S[X]")
    print("  supplementary_figure_Q3_sensitivity.png  ← Supplementary Figure")
    print("  q3_sensitivity_summary.txt               ← Manuscript prose ready to paste")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()