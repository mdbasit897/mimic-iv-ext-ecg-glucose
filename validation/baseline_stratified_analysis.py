"""
baseline_stratified_analysis.py
════════════════════════════════════════════════════════════════════════════════
Supplementary analyses for MIMIC-IV-Ext-ECG-Glucose v1.0.0, leak-free.

  Q5  Last-value baseline: lag_glucose_mg_dl, only when the previous draw is
      >= 15 min earlier (shorter gaps are mostly the other assay of the same draw)
  Q6  Feature-set ablation: ECG only / causal history + context / both
  Q7  Main model by alignment tier and ECG-glucose direction
  Q10 Main model by sex and age group

Q7 and Q10 describe case mix, not data quality: the tiers and demographic groups
differ in patient mix, so these errors should not be read as validating (or
invalidating) the alignment tiers.

All models are trained on IDEAL+USABLE TRAIN rows with glucoecg.features inputs
and scored on the IDEAL+USABLE TEST rows with patient-level bootstrap CIs.

Usage
  export GLUCOECG_CSV=/path/to/mimiciv_ecg_glucose_aligned.csv
  python validation/baseline_stratified_analysis.py --out results/supplement
════════════════════════════════════════════════════════════════════════════════
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glucoecg import config, plots  # noqa: E402
from glucoecg.features import FEATURE_SET_LABELS, TARGET, load_dataset, test_sets  # noqa: E402
from glucoecg.metrics import bootstrap_ci, regression_metrics  # noqa: E402
from glucoecg.modeling import AGE_BINS, AGE_LABELS, train_feature_set  # noqa: E402

TIERS = ["TIGHT", "MODERATE", "LOOSE", "EXTENDED"]
DIRECTIONS = ["ECG_BEFORE_GLUCOSE", "ECG_AFTER_GLUCOSE", "SIMULTANEOUS"]


def row(label, group, yt, yp, ids, n_boot):
    r = regression_metrics(yt, yp)
    lo, hi = bootstrap_ci(yt, yp, ids, n_boot=n_boot) if n_boot and r["n"] > 1 else (np.nan, np.nan)
    return {"stratification": label, "subgroup": group, "n": r["n"], "mae": round(r["mae"], 2),
            "mae_ci_low": round(lo, 2), "mae_ci_high": round(hi, 2), "rmse": round(r["rmse"], 2),
            "mard": round(r["mard"], 1), "r2": round(r["r2"], 3), "bias": round(r["bias"], 2)}


def main():
    ap = argparse.ArgumentParser(description="Leak-free supplementary stratified analyses")
    ap.add_argument("--csv", default=None, help=f"dataset CSV (default: ${config.ENV_VAR})")
    ap.add_argument("--out", default="results/supplement")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--skip-checksum", action="store_true")
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv = config.resolve_csv(args.csv)
    digest = config.check_csv(csv, args.skip_checksum)
    df = load_dataset(csv)
    y = df[TARGET].to_numpy(float)
    ids = df["subject_id"].to_numpy()
    m = test_sets(df)["iu"]

    # ── Q5: last value ────────────────────────────────────────────────────────
    lag = df["lag_ok"].to_numpy(float)
    q5 = m & ~np.isnan(lag)
    q5_row = row("Q5 last value", "lag_glucose_mg_dl, gap >= 15 min", y[q5], lag[q5], ids[q5], args.bootstrap)
    gaps = df.loc[q5, "inter_measurement_gap_hr"]
    q5_row["median_gap_hr"] = round(float(gaps.median()), 2)
    print(f"Q5 last value: n={q5_row['n']:,} of {m.sum():,} test rows, MAE {q5_row['mae']}")

    # ── Q6: feature-set ablation ──────────────────────────────────────────────
    preds, q6 = {}, []
    for fs in ["ecg_only", "history_only", "ecg_history"]:
        _, p = train_feature_set(df, fs, "iu", seed=args.seed)
        preds[fs] = p
        q6.append(row("Q6 ablation", FEATURE_SET_LABELS[fs], y[m], p[m], ids[m], args.bootstrap))
        print(f"Q6 {FEATURE_SET_LABELS[fs]}: MAE {q6[-1]['mae']}")
    pm = preds["ecg_history"]

    # ── Q7: alignment and direction ───────────────────────────────────────────
    d = df.loc[m, ["alignment_quality", "temporal_relationship", "gender", "age"]].reset_index(drop=True)
    yt, yp, gid = y[m], pm[m], ids[m]
    q7 = []
    for t in TIERS:
        s = (d["alignment_quality"] == t).to_numpy()
        q7.append(row("Alignment quality", t, yt[s], yp[s], gid[s], args.bootstrap))
    for dr in DIRECTIONS:
        s = (d["temporal_relationship"] == dr).to_numpy()
        if s.sum():
            q7.append(row("Temporal direction", dr, yt[s], yp[s], gid[s], args.bootstrap))
    grid = np.full((len(TIERS), 2), np.nan)
    for i, t in enumerate(TIERS):
        for j, dr in enumerate(DIRECTIONS[:2]):
            s = ((d["alignment_quality"] == t) & (d["temporal_relationship"] == dr)).to_numpy()
            if s.sum() >= 5:
                q7.append(row("Alignment x direction", f"{t} x {dr}", yt[s], yp[s], gid[s], args.bootstrap))
                grid[i, j] = q7[-1]["mae"]
    q7 = pd.DataFrame(q7)

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    im = ax.imshow(grid, cmap="Blues")
    for i in range(len(TIERS)):
        for j in range(2):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", fontsize=10)
    ax.set_xticks([0, 1], ["ECG before\nglucose", "ECG after\nglucose"])
    ax.set_yticks(range(len(TIERS)), TIERS)
    ax.set_title("MAE (mg/dL) by alignment and direction\n(case mix, not a data-quality test)", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    plots.save(fig, out / "supp_figure_Q7_alignment_heatmap", args.formats)

    # ── Q10: sex and age ──────────────────────────────────────────────────────
    d["age_group"] = pd.cut(d["age"], AGE_BINS, labels=AGE_LABELS, right=False).astype(str)
    q10 = []
    for sex in ["F", "M"]:
        s = (d["gender"] == sex).to_numpy()
        q10.append(row("Sex", sex, yt[s], yp[s], gid[s], args.bootstrap))
    for a in AGE_LABELS:
        s = (d["age_group"] == a).to_numpy()
        q10.append(row("Age group", a, yt[s], yp[s], gid[s], args.bootstrap))
    for sex in ["F", "M"]:
        for a in AGE_LABELS:
            s = ((d["gender"] == sex) & (d["age_group"] == a)).to_numpy()
            if s.sum() >= 5:
                q10.append(row("Sex x age", f"{sex} {a}", yt[s], yp[s], gid[s], args.bootstrap))
    q10 = pd.DataFrame(q10)

    fig, ax = plt.subplots(figsize=(7, 4))
    sub = q10[q10["stratification"] == "Sex x age"]
    for k, sex in enumerate(["F", "M"]):
        r = sub[sub["subgroup"].str.startswith(sex)]
        x = np.arange(len(r)) + (k - 0.5) * 0.38
        ax.bar(x, r["mae"], width=0.38, color=["#9E9E9E", "#0072B2"][k], label=sex)
        ax.errorbar(x, r["mae"], yerr=[r["mae"] - r["mae_ci_low"], r["mae_ci_high"] - r["mae"]],
                    fmt="none", color="k", lw=0.8)
    ax.set_xticks(range(len(AGE_LABELS)), AGE_LABELS)
    ax.set_xlabel("Age group (years)")
    ax.set_ylabel("MAE (mg/dL), 95% CI")
    ax.legend(title="Sex")
    ax.set_title("Main model by sex and age (case mix, descriptive)", fontsize=10)
    plots.save(fig, out / "supp_figure_Q10_demographic_bars", args.formats)

    # ── Save ──────────────────────────────────────────────────────────────────
    pd.DataFrame([q5_row]).to_csv(out / "supp_table_Q5_last_value.csv", index=False)
    pd.DataFrame(q6).to_csv(out / "supp_table_Q6_ablation.csv", index=False)
    q7.to_csv(out / "supp_table_Q7_alignment_stratification.csv", index=False)
    q10.to_csv(out / "supp_table_Q10_demographic_stratification.csv", index=False)

    fmt = lambda r: (f"  {str(r['subgroup'])[:44]:44s} n={r['n']:6,d}  MAE {r['mae']:6.2f} "  # noqa: E731
                     f"[{r['mae_ci_low']:.2f}, {r['mae_ci_high']:.2f}]  bias {r['bias']:+.2f}")
    lines = ["SUPPLEMENTARY STRATIFIED ANALYSES (leak-free)", "=" * 80,
             f"IDEAL+USABLE test rows: {m.sum():,}; main model = {FEATURE_SET_LABELS['ecg_history']}", "",
             "Q5 last value", fmt(q5_row), f"  median gap {q5_row['median_gap_hr']} h", "",
             "Q6 ablation", *[fmt(r) for r in q6], "",
             "Q7 alignment / direction (case mix)", *[fmt(r) for r in q7.to_dict("records")], "",
             "Q10 sex / age (case mix)", *[fmt(r) for r in q10.to_dict("records")]]
    (out / "supplementary_analysis_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    config.write_run_info(out, csv, digest, {"seed": args.seed, "bootstrap": args.bootstrap})


if __name__ == "__main__":
    main()
