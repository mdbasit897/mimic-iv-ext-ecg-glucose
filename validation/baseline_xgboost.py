"""
baseline_xgboost.py
════════════════════════════════════════════════════════════════════════════════
Leak-free reference baselines for MIMIC-IV-Ext-ECG-Glucose v1.0.0 (Table 7).

Every model uses only inputs allowed by glucoecg.features (no target-derived,
whole-record, discharge-time or intervention columns) and is scored on the same
fixed TEST rows, with patient-level bootstrap 95% CIs.

Rows
  Training-set mean                  constant prediction (no-skill floor)
  Last value                         lag_glucose_mg_dl, gap >= 15 min (scored where available)
  Mean of earlier records            prior_mean (scored where available)
  Ridge / Random Forest              ECG + causal history + context      (--skip-extra to omit)
  XGBoost, ECG only                  ECG machine measurements only
  XGBoost, causal history + context  no ECG features
  XGBoost, ECG + causal history      main model
  Tier comparison                    main model trained on IDEAL+USABLE vs all tiers,
                                     scored on identical rows (paired bootstrap)

Test sets (glucoecg.features.test_sets): iu (primary), conservative, all,
iu_history, iu_no_history.

Usage
  export GLUCOECG_CSV=/path/to/mimiciv_ecg_glucose_aligned.csv
  python validation/baseline_xgboost.py --out results/baseline

Outputs (in --out)
  table7_main.csv / .pdf / .png     manuscript Table 7 (primary test set)
  results_long.csv                  every model x test set x metric
  tier_comparison.csv               paired MAE difference, IDEAL+USABLE vs all-tier training
  ecg_contribution.csv              paired MAE differences: ECG + history vs history only,
                                    ECG only vs training-set mean
  seed_stability.csv                MAE per seed for the three XGBoost feature sets
  strata_main_model.csv             main model by alignment, direction, glycaemic band, sex, age
  clarke_*.pdf, parkes_*.pdf        corrected error grids
  prediction_scatter, glycaemic_subgroup, residual_distribution, feature_importance_*, shap_ecg_only
  predictions_test.csv.gz           row-level TEST predictions (credentialed data: do not share)
  technical_validation_summary.txt  human-readable summary
  run_info.json                     git commit, CSV checksum, package versions, settings
════════════════════════════════════════════════════════════════════════════════
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glucoecg import config, plots  # noqa: E402
from glucoecg.features import (FEATURE_SET_LABELS, FEATURE_SETS, TARGET, TEST_SET_LABELS,  # noqa: E402
                               load_dataset, test_sets, train_mask)
from glucoecg.metrics import (bootstrap_ci, clarke_zones, paired_bootstrap_ci, parkes_zones,  # noqa: E402
                              regression_metrics, zone_summary)
from glucoecg.modeling import (AGE_BINS, AGE_LABELS, fit_random_forest, fit_ridge, matrix,  # noqa: E402
                               predict, train_feature_set)

MAIN = "ecg_history"


def score(name, pred, y, groups, tests, n_boot, train_label, n_train):
    """Metrics of one prediction vector on every test set (rows where pred is available)."""
    out = []
    for key, mask in tests.items():
        m = mask & ~np.isnan(pred)
        if m.sum() == 0:
            continue
        r = regression_metrics(y[m], pred[m])
        lo, hi = bootstrap_ci(y[m], pred[m], groups[m], n_boot=n_boot) if n_boot else (np.nan, np.nan)
        cz, pz = zone_summary(clarke_zones(y[m], pred[m])), zone_summary(parkes_zones(y[m], pred[m]))
        out.append({"model": name, "train_set": train_label, "n_train": n_train, "test_set": key,
                    "n_scored": r["n"], "n_test_set": int(mask.sum()), "mae": r["mae"],
                    "mae_ci_low": lo, "mae_ci_high": hi, "rmse": r["rmse"], "mard": r["mard"],
                    "r2": r["r2"], "bias": r["bias"], "clarke_A": cz["A"], "clarke_B": cz["B"],
                    "clarke_C": cz["C"], "clarke_D": cz["D"], "clarke_E": cz["E"],
                    "clarke_AB": cz["A+B"], "parkes_AB": pz["A+B"], "parkes_D": pz["D"]})
        print(f"  {key:14s} n={r['n']:6,d}  MAE {r['mae']:6.2f} [{lo:.2f}, {hi:.2f}]  R2 {r['r2']:.3f}")
    return out


def strata_table(df, y, pred, mask):
    """Main-model metrics by alignment, direction, glycaemic band, sex and age (descriptive)."""
    d = df.loc[mask, ["alignment_quality", "temporal_relationship", "gender", "age"]].copy()
    yt, yp = y[mask], pred[mask]
    d["glycaemic_band"] = np.select([yt < 70, yt <= 180], ["hypo (<70)", "eu (70-180)"], "hyper (>180)")
    d["age_group"] = pd.cut(d["age"], AGE_BINS, labels=AGE_LABELS, right=False).astype(str)
    d["alignment_x_direction"] = d["alignment_quality"] + " x " + d["temporal_relationship"]
    rows = []
    for col in ["alignment_quality", "temporal_relationship", "alignment_x_direction",
                "glycaemic_band", "gender", "age_group"]:
        for val in sorted(d[col].dropna().unique()):
            sel = (d[col] == val).to_numpy()
            if sel.sum() >= 5:
                rows.append({"stratifier": col, "subgroup": val, **regression_metrics(yt[sel], yp[sel])})
    return pd.DataFrame(rows)


def fmt(v, nd=2):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description="Leak-free reference baselines (Table 7)")
    ap.add_argument("--csv", default=None, help=f"dataset CSV (default: ${config.ENV_VAR})")
    ap.add_argument("--out", default="results/baseline")
    ap.add_argument("--seeds", type=int, nargs="+", default=config.DEFAULT_SEEDS,
                    help="the first seed gives the reported predictions; all seeds give stability")
    ap.add_argument("--bootstrap", type=int, default=1000, help="patient-level resamples (0 = off)")
    ap.add_argument("--skip-extra", action="store_true", help="skip Ridge and Random Forest")
    ap.add_argument("--skip-shap", action="store_true")
    ap.add_argument("--skip-checksum", action="store_true")
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv = config.resolve_csv(args.csv)
    digest = config.check_csv(csv, args.skip_checksum)
    print(f"Dataset: {csv}\nLoading and building causal history features...")
    df = load_dataset(csv)

    y = df[TARGET].to_numpy(float)
    groups = df["subject_id"].to_numpy()
    tests = test_sets(df)
    tr_iu, tr_all = train_mask(df, "iu"), train_mask(df, "all")
    n_iu, n_all = int(tr_iu.sum()), int(tr_all.sum())
    print(f"  Train rows: IDEAL+USABLE {n_iu:,} | all tiers {n_all:,}")
    print("  Test rows :", {k: int(v.sum()) for k, v in tests.items()})
    seed0 = args.seeds[0]

    rows, preds = [], {}

    # ── Simple baselines ──────────────────────────────────────────────────────
    preds["Training-set mean"] = np.full(len(df), y[tr_iu].mean())
    preds["Last value (gap >= 15 min)"] = df["lag_ok"].to_numpy(float)
    preds["Mean of earlier records"] = df["prior_mean"].to_numpy(float)
    for name in list(preds):
        print(f"\n[{name}]")
        rows += score(name, preds[name], y, groups, tests, args.bootstrap, "-", 0)

    # ── XGBoost feature-set ablation (IDEAL+USABLE training) ─────────────────
    models, seed_rows = {}, []
    for fs in ["ecg_only", "history_only", MAIN]:
        label = FEATURE_SET_LABELS[fs]
        print(f"\n[{label}] seeds {args.seeds}")
        for s in args.seeds:
            model, p = train_feature_set(df, fs, "iu", seed=s, n_jobs=args.n_jobs)
            mae = regression_metrics(y[tests["iu"]], p[tests["iu"]])["mae"]
            seed_rows.append({"model": label, "seed": s, "best_iteration": model.best_iteration,
                              "mae_iu_test": mae})
            print(f"  seed {s}: best_iteration={model.best_iteration}  MAE(iu)={mae:.2f}")
            if s == seed0:
                models[fs], preds[label] = model, p
        rows += score(label, preds[label], y, groups, tests, args.bootstrap, "IDEAL+USABLE", n_iu)

    # ── Ridge and Random Forest on the main feature set ──────────────────────
    if not args.skip_extra:
        X = matrix(df, FEATURE_SETS[MAIN])
        extra = [("Ridge, ECG + causal history", lambda: fit_ridge(X, y, tr_iu)),
                 ("Random Forest, ECG + causal history",
                  lambda: fit_random_forest(X, y, tr_iu, seed=seed0, n_jobs=args.n_jobs))]
        for name, fit in extra:
            print(f"\n[{name}]")
            preds[name] = predict(fit(), X)
            rows += score(name, preds[name], y, groups, tests, args.bootstrap, "IDEAL+USABLE", n_iu)

    # ── Tier comparison: same model, different training tiers, identical test rows ──
    main_label = FEATURE_SET_LABELS[MAIN]
    all_label = main_label + " [all-tier training]"
    print(f"\n[{all_label}]")
    _, preds[all_label] = train_feature_set(df, MAIN, "all", seed=seed0, n_jobs=args.n_jobs)
    rows += score(all_label, preds[all_label], y, groups, tests, args.bootstrap,
                  "CAUTION+USABLE+IDEAL", n_all)
    tier_rows = []
    print("\n[Tier comparison] MAE(IDEAL+USABLE-trained) - MAE(all-tier-trained), same rows")
    for key, mask in tests.items():
        pa, pb = preds[main_label][mask], preds[all_label][mask]
        d, lo, hi = paired_bootstrap_ci(y[mask], pa, pb, groups[mask], n_boot=max(args.bootstrap, 200))
        tier_rows.append({"test_set": key, "n": int(mask.sum()),
                          "mae_iu_trained": regression_metrics(y[mask], pa)["mae"],
                          "mae_all_trained": regression_metrics(y[mask], pb)["mae"],
                          "difference_iu_minus_all": d, "ci_low": lo, "ci_high": hi})
        print(f"  {key:14s} {d:+.2f} mg/dL  [{lo:+.2f}, {hi:+.2f}]")
    tier = pd.DataFrame(tier_rows)

    # ── ECG contribution: paired comparisons on identical rows ───────────────
    # (a) ECG + history vs history only: what ECG adds when history is available
    # (b) ECG only vs training-set mean: whether ECG alone beats the no-skill floor
    contrib_pairs = [
        ("ECG + history vs history only", main_label, FEATURE_SET_LABELS["history_only"]),
        ("ECG only vs training-set mean", FEATURE_SET_LABELS["ecg_only"], "Training-set mean"),
    ]
    contrib_rows = []
    print("\n[ECG contribution] MAE(model A) - MAE(model B), same rows (negative = A better)")
    for name, a_lab, b_lab in contrib_pairs:
        for key, mask in tests.items():
            pa, pb = preds[a_lab][mask], preds[b_lab][mask]
            d, lo, hi = paired_bootstrap_ci(y[mask], pa, pb, groups[mask], n_boot=max(args.bootstrap, 200))
            contrib_rows.append({"comparison": name, "model_a": a_lab, "model_b": b_lab, "test_set": key,
                                 "n": int(mask.sum()), "mae_a": regression_metrics(y[mask], pa)["mae"],
                                 "mae_b": regression_metrics(y[mask], pb)["mae"],
                                 "difference_a_minus_b": d, "ci_low": lo, "ci_high": hi})
            print(f"  {name:32s} {key:14s} {d:+.2f} mg/dL  [{lo:+.2f}, {hi:+.2f}]")
    contrib = pd.DataFrame(contrib_rows)

    # ── Tables ────────────────────────────────────────────────────────────────
    res = pd.DataFrame(rows)
    res.to_csv(out / "results_long.csv", index=False)
    tier.to_csv(out / "tier_comparison.csv", index=False)
    contrib.to_csv(out / "ecg_contribution.csv", index=False)
    seeds = pd.DataFrame(seed_rows)
    seeds.to_csv(out / "seed_stability.csv", index=False)

    main_tbl = res[res["test_set"] == "iu"]
    main_tbl.to_csv(out / "table7_main.csv", index=False)
    cols = ["Model", "Train N", "n scored", "MAE (95% CI)", "RMSE", "MARD %", "R²",
            "Clarke A+B %", "Clarke D %"]
    table = [[r.model, f"{r.n_train:,}" if r.n_train else "-", f"{r.n_scored:,}",
              f"{r.mae:.2f} ({fmt(r.mae_ci_low)}-{fmt(r.mae_ci_high)})", fmt(r.rmse), fmt(r.mard, 1),
              fmt(r.r2, 3), fmt(r.clarke_AB, 1), fmt(r.clarke_D, 1)] for r in main_tbl.itertuples()]
    plots.plot_results_table(table, cols, f"Leak-free baselines, {TEST_SET_LABELS['iu']} "
                             f"(n = {int(tests['iu'].sum()):,})", out / "table7_main", args.formats)

    # ── Figures ───────────────────────────────────────────────────────────────
    m = tests["iu"]
    pm = preds[main_label]
    for fs in ["ecg_only", MAIN]:
        p, lab = preds[FEATURE_SET_LABELS[fs]], FEATURE_SET_LABELS[fs]
        plots.plot_clarke(y[m], p[m], clarke_zones(y[m], p[m]), f"Clarke error grid: {lab}",
                          out / f"clarke_{fs}", args.formats)
        plots.plot_parkes(y[m], p[m], parkes_zones(y[m], p[m]), f"Parkes error grid (type 1): {lab}",
                          out / f"parkes_{fs}", args.formats)
        plots.plot_feature_importance(models[fs], FEATURE_SETS[fs], f"Gain importance: {lab}",
                                      out / f"feature_importance_{fs}", formats=args.formats)
    plots.plot_prediction_scatter(y[m], {FEATURE_SET_LABELS[k]: preds[FEATURE_SET_LABELS[k]][m]
                                         for k in ["ecg_only", "history_only", MAIN]},
                                  out / "prediction_scatter", args.formats)
    glyc = plots.plot_glycaemic_subgroup(y[m], pm[m], f"{main_label} by glycaemic band",
                                         out / "glycaemic_subgroup", args.formats)
    plots.plot_residuals_by_tier(y[m], pm[m], df.loc[m, "record_usability"],
                                 out / "residual_distribution", args.formats)
    strata_table(df, y, pm, m).to_csv(out / "strata_main_model.csv", index=False)

    if not args.skip_shap:
        try:
            import matplotlib.pyplot as plt
            import shap
            Xe = matrix(df, FEATURE_SETS["ecg_only"])[m]
            idx = np.random.default_rng(seed0).choice(len(Xe), min(2000, len(Xe)), replace=False)
            sv = shap.TreeExplainer(models["ecg_only"]).shap_values(Xe[idx])
            shap.summary_plot(sv, pd.DataFrame(Xe[idx], columns=FEATURE_SETS["ecg_only"]),
                              show=False, max_display=20)
            plt.title("SHAP, XGBoost ECG only (descriptive)", fontweight="bold")
            plots.save(plt.gcf(), out / "shap_ecg_only", args.formats)
        except ImportError:
            print("  shap not installed; skipping the SHAP figure")

    keep = tests["all"] | tests["iu"]
    pred_df = pd.DataFrame({"labevent_id": df["labevent_id"], "subject_id": df["subject_id"],
                            "record_usability": df["record_usability"], "y_true": y, **preds})
    pred_df[keep].to_csv(out / "predictions_test.csv.gz", index=False, compression="gzip")

    # ── Summary ───────────────────────────────────────────────────────────────
    match = "matches" if digest == config.EXPECTED_SHA256 else "DOES NOT match"
    lines = ["LEAK-FREE TECHNICAL VALIDATION BASELINES", "=" * 84,
             f"Dataset   : {csv.name}  SHA-256 {digest[:16]}... ({match} PhysioNet v1.0.0)",
             f"Seeds     : {args.seeds} (reported: {seed0})   Bootstrap: {args.bootstrap} patient resamples",
             f"Train rows: IDEAL+USABLE {n_iu:,} | all tiers {n_all:,}", ""]
    for key in ["iu", "conservative", "all", "iu_history", "iu_no_history"]:
        sub = res[res["test_set"] == key]
        lines += [f"{TEST_SET_LABELS[key]} (n = {int(tests[key].sum()):,})", "-" * 84,
                  f"{'Model':52s} {'n':>7s} {'MAE':>6s} {'95% CI':>13s} {'R2':>6s} {'A+B':>5s} {'D':>4s}"]
        for r in sub.itertuples():
            lines.append(f"{r.model[:52]:52s} {r.n_scored:7,d} {r.mae:6.2f} "
                         f"{fmt(r.mae_ci_low):>6s}-{fmt(r.mae_ci_high):<6s} {r.r2:6.3f} "
                         f"{r.clarke_AB:5.1f} {r.clarke_D:4.1f}")
        lines.append("")
    lines += ["Tier comparison: MAE(IDEAL+USABLE-trained) - MAE(all-tier-trained), same rows", "-" * 84]
    lines += [f"  {r.test_set:14s} {r.difference_iu_minus_all:+.2f} mg/dL  [{r.ci_low:+.2f}, {r.ci_high:+.2f}]"
              for r in tier.itertuples()]
    ss = seeds.groupby("model")["mae_iu_test"].agg(["mean", "min", "max"])
    lines += ["", "ECG contribution: MAE(A) - MAE(B), same rows; negative = A better", "-" * 84]
    lines += [f"  {r.comparison:32s} {r.test_set:14s} {r.difference_a_minus_b:+.2f} mg/dL  "
              f"[{r.ci_low:+.2f}, {r.ci_high:+.2f}]" for r in contrib.itertuples()]
    lines += ["", "Seed stability, MAE on IDEAL+USABLE test", "-" * 84]
    lines += [f"  {k:52s} mean {v['mean']:.2f}  range {v['min']:.2f}-{v['max']:.2f}"
              for k, v in ss.iterrows()]
    lines += ["", "Main model by glycaemic band (IDEAL+USABLE test)", "-" * 84]
    lines += [f"  {k:14s} n={v['n']:6,d}  MAE {v['mae']:.2f}  bias {v['bias']:+.2f}" for k, v in glyc.items()]
    (out / "technical_validation_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))

    config.write_run_info(out, csv, digest, {
        "seeds": args.seeds, "bootstrap": args.bootstrap, "feature_sets": FEATURE_SETS,
        "n_train": {"iu": n_iu, "all": n_all}, "n_test": {k: int(v.sum()) for k, v in tests.items()},
    })
    print(f"\nAll outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
