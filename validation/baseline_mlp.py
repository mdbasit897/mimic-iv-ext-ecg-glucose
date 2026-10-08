"""
baseline_mlp.py
════════════════════════════════════════════════════════════════════════════════
Leak-free MaskedDynamicsMLP baseline for MIMIC-IV-Ext-ECG-Glucose v1.0.0.

Same architecture and optimiser as v1.0 (4 dense blocks 512-256-128-64, tapered
dropout, AdamW 3e-4, CosineAnnealingWarmRestarts T0=30 Tmult=2, patience 25),
but every input comes from glucoecg.features:

  ecg_static      15 ECG interval / axis / QTc values (placeholders -> missing)
  ecg_binary      13 ECG plausibility and prolongation flags
  history_stats    8 statistics of strictly earlier glucose records + has-history flag
  clinical_scalar  6 timing and context values known at measurement time
  clinical_cat     5 ordinal / binary context values scaled to [0, 1]
  dynamics         3 masked positions: previous glucose, gap, previous change
                   (learned MASK embedding when missing)

Removed from v1.0: patient_* whole-record statistics, glucose_delta / rate
(computed from the target), decoupling_risk_score, n_glucose_ecg_pairs,
ref_range_lower / upper. Scalars are clipped to the training 0.5-99.5% range
and z-scored with TRAIN statistics only.

Usage
  export GLUCOECG_CSV=/path/to/mimiciv_ecg_glucose_aligned.csv
  python validation/baseline_mlp.py --out results/mlp --seeds 42 1 2
════════════════════════════════════════════════════════════════════════════════
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glucoecg import config, plots  # noqa: E402
from glucoecg.features import (ECG_BINARY_FEATURES, ECG_NUMERIC, TARGET, TEST_SET_LABELS,  # noqa: E402
                               assert_no_leak, load_dataset, test_sets, train_mask,
                               usability_weights, val_mask)
from glucoecg.metrics import (bootstrap_ci, clarke_zones, parkes_zones, regression_metrics,  # noqa: E402
                              zone_summary)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HISTORY_COLS = ["prior_n", "prior_mean", "prior_std", "prior_tir_pct", "prior_min", "prior_max",
                "prior_last", "prior_last_hr"]
CLINICAL_SCALAR_COLS = ["age", "abs_offset_minutes", "glucose_ecg_offset_minutes",
                        "glucose_seq_in_stay", "hours_since_admission", "hours_since_icu_admission"]
CLINICAL_CAT_COLS = {"alignment_quality_enc": 3.0, "hr_regime_enc": 4.0, "gender_m": 1.0,
                     "ecg_before": 1.0, "during_icu": 1.0}
DYNAMICS_COLS = ["lag_ok", "gap_ok_hr", "prior_change"]
LOG_COLS = {"prior_n", "prior_last_hr", "glucose_seq_in_stay", "gap_ok_hr"}  # heavy right tails

for _cols in (ECG_NUMERIC, ECG_BINARY_FEATURES, HISTORY_COLS, CLINICAL_SCALAR_COLS,
              list(CLINICAL_CAT_COLS), DYNAMICS_COLS):
    assert_no_leak(_cols)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE TENSORS
# ─────────────────────────────────────────────────────────────────────────────

class Scaler:
    """Clip to training quantiles, then z-score with training mean / SD. NaN stays NaN."""

    def fit(self, a):
        self.lo, self.hi = np.nanquantile(a, 0.005, axis=0), np.nanquantile(a, 0.995, axis=0)
        c = np.clip(a, self.lo, self.hi)
        self.mu, self.sd = np.nanmean(c, axis=0), np.nanstd(c, axis=0)
        self.sd[~(self.sd > 1e-6)] = 1.0
        self.mu = np.nan_to_num(self.mu)
        return self

    def transform(self, a):
        return (np.clip(a, self.lo, self.hi) - self.mu) / self.sd


def raw(df, cols):
    a = df[cols].astype(float).to_numpy()
    for j, c in enumerate(cols):
        if c in LOG_COLS:
            a[:, j] = np.sign(a[:, j]) * np.log1p(np.abs(a[:, j]))
    return a


def build_tensors(df, train_rows):
    """Return {group: float32 array for every row of df} using TRAIN-row statistics only."""
    groups = {}
    for name, cols in [("ecg_static", ECG_NUMERIC), ("history_stats", HISTORY_COLS),
                       ("clinical_scalar", CLINICAL_SCALAR_COLS)]:
        a = raw(df, cols)
        s = Scaler().fit(a[train_rows])
        groups[name] = np.nan_to_num(s.transform(a), nan=0.0)
    groups["history_stats"] = np.column_stack(
        [groups["history_stats"], (df["prior_n"].fillna(0) > 0).to_numpy(float)])
    groups["ecg_binary"] = np.nan_to_num(df[ECG_BINARY_FEATURES].astype(float).to_numpy())
    groups["clinical_cat"] = np.column_stack(
        [np.nan_to_num(df[c].astype(float).to_numpy() / m) for c, m in CLINICAL_CAT_COLS.items()])
    d = raw(df, DYNAMICS_COLS)
    mask = ~np.isnan(d)
    s = Scaler().fit(d[train_rows])
    groups["dynamics_feats"] = np.nan_to_num(np.clip(s.transform(d), -5, 5), nan=0.0)
    groups["dynamics_mask"] = mask.astype(float)
    return {k: v.astype(np.float32) for k, v in groups.items()}


STATIC_KEYS = ["ecg_static", "ecg_binary", "history_stats", "clinical_scalar", "clinical_cat"]
ALL_KEYS = STATIC_KEYS + ["dynamics_feats", "dynamics_mask"]


def loader(groups, rows, target_z, weights, batch_size, shuffle):
    tensors = [torch.from_numpy(groups[k][rows]) for k in ALL_KEYS]
    tensors += [torch.from_numpy(target_z[rows].astype(np.float32)),
                torch.from_numpy(weights[rows].astype(np.float32))]
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle,
                      drop_last=shuffle)


def unpack(batch):
    *feats, target, weight = [b.to(DEVICE) for b in batch]
    return dict(zip(ALL_KEYS, feats)), target, weight


# ─────────────────────────────────────────────────────────────────────────────
# MODEL (unchanged from v1.0)
# ─────────────────────────────────────────────────────────────────────────────

class MaskedDynamicsMLP(nn.Module):
    """Dense MLP; NULL dynamics positions use a learned MASK embedding instead of zero."""

    def __init__(self, d_static: int, n_dyn: int = 3, d_dyn: int = 16,
                 hidden: int = 512, n_layers: int = 4, dropout: float = 0.25):
        super().__init__()
        self.n_dyn = n_dyn
        self.dyn_proj = nn.ModuleList([nn.Linear(1, d_dyn) for _ in range(n_dyn)])
        self.mask_emb = nn.ParameterList([nn.Parameter(torch.randn(d_dyn) * 0.01) for _ in range(n_dyn)])
        layers, in_dim = [], d_static + n_dyn * d_dyn
        for i in range(n_layers):
            out_dim = max(hidden // (2 ** i), 64)
            layers += [nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU(),
                       nn.Dropout(dropout * (1 - i * 0.05))]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, batch: dict) -> torch.Tensor:
        static = torch.cat([batch[k] for k in STATIC_KEYS], dim=-1)
        feats, mask = batch["dynamics_feats"], batch["dynamics_mask"]
        dyn = []
        for i in range(self.n_dyn):
            x_i, m_i = feats[:, i:i + 1], mask[:, i:i + 1]
            masked = self.mask_emb[i].unsqueeze(0).expand(x_i.size(0), -1)
            dyn.append(m_i * self.dyn_proj[i](x_i) + (1 - m_i) * masked)
        return self.net(torch.cat([static, *dyn], dim=-1)).squeeze(-1)


@torch.no_grad()
def predict(model, groups, rows, g_mean, g_std, batch_size=4096):
    model.eval()
    out = []
    idx = np.flatnonzero(rows)
    for i in range(0, len(idx), batch_size):
        sl = idx[i:i + batch_size]
        batch = {k: torch.from_numpy(groups[k][sl]).to(DEVICE) for k in ALL_KEYS}
        out.append(model(batch).cpu().numpy())
    pred = np.full(len(rows), np.nan)
    pred[idx] = np.concatenate(out) * g_std + g_mean
    return pred


def train_one(groups, y, tr, va, weights, seed, args):
    torch.manual_seed(seed)
    np.random.seed(seed)
    g_mean, g_std = float(y[tr].mean()), float(y[tr].std())
    target_z = (y - g_mean) / g_std
    dl = loader(groups, tr, target_z, weights, args.batch_size, shuffle=True)
    d_static = sum(groups[k].shape[1] for k in STATIC_KEYS)
    model = MaskedDynamicsMLP(d_static).to(DEVICE)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=5e-3)
    sched = CosineAnnealingWarmRestarts(opt, T_0=30, T_mult=2, eta_min=1e-5)
    best, best_ep, best_state, stale, tr_curve, va_curve = np.inf, 0, None, 0, [], []
    for ep in range(args.epochs):
        model.train()
        maes = []
        for batch in dl:
            b, t, w = unpack(batch)
            opt.zero_grad()
            pred = model(b)
            loss = (w * (pred - t) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            maes.append((pred.detach() - t).abs().mean().item() * g_std)
        sched.step()
        va_mae = regression_metrics(y[va], predict(model, groups, va, g_mean, g_std)[va])["mae"]
        tr_curve.append(float(np.mean(maes)))
        va_curve.append(va_mae)
        if va_mae < best:
            best, best_ep, stale = va_mae, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if ep == 0 or (ep + 1) % 5 == 0 or stale == 0:
            print(f"    epoch {ep + 1:3d}  train {tr_curve[-1]:6.2f}  val {va_mae:6.2f}{'  best' if stale == 0 else ''}")
        if stale >= args.patience:
            print(f"    early stop at epoch {ep + 1}")
            break
    model.load_state_dict(best_state)
    return model, g_mean, g_std, best_ep, best, tr_curve, va_curve, d_static


def plot_training_curve(train_maes, val_maes, best_epoch, stem, formats, t0=30, t_mult=2):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ep = range(1, len(train_maes) + 1)
    ax.plot(ep, train_maes, color="#0072B2", lw=2, label="Train MAE")
    ax.plot(ep, val_maes, color="#E69F00", lw=2, label="Validation MAE")
    restart, t = t0, t0
    while restart < len(train_maes):
        ax.axvline(restart, color="#CC79A7", lw=1, ls=":")
        t *= t_mult
        restart += t
    ax.axvline(best_epoch + 1, color="#009E73", lw=1.5, ls="--",
               label=f"Best epoch {best_epoch + 1} ({val_maes[best_epoch]:.2f} mg/dL)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (mg/dL)")
    ax.set_title("MaskedDynamicsMLP training curve (dotted: LR warm restarts)", fontweight="bold")
    ax.legend(fontsize=9)
    plots.save(fig, stem, formats)


def main():
    ap = argparse.ArgumentParser(description="Leak-free MaskedDynamicsMLP baseline")
    ap.add_argument("--csv", default=None, help=f"dataset CSV (default: ${config.ENV_VAR})")
    ap.add_argument("--out", default="results/mlp")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--skip-checksum", action="store_true")
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv = config.resolve_csv(args.csv)
    digest = config.check_csv(csv, args.skip_checksum)
    print(f"Device: {DEVICE}\nDataset: {csv}")
    df = load_dataset(csv)
    y = df[TARGET].to_numpy(float)
    tr, va = train_mask(df, "iu"), val_mask(df, "iu")
    tests = test_sets(df)
    weights = usability_weights(df)
    groups = build_tensors(df, tr)
    print(f"  TRAIN {tr.sum():,} | VALIDATION {va.sum():,} | TEST(iu) {tests['iu'].sum():,}")
    print("  Input widths:", {k: groups[k].shape[1] for k in ALL_KEYS})

    seed_rows, keep = [], None
    for s in args.seeds:
        print(f"\n[seed {s}]")
        model, g_mean, g_std, best_ep, best_val, trc, vac, d_static = train_one(groups, y, tr, va, weights, s, args)
        pred = predict(model, groups, tests["all"] | tests["iu"], g_mean, g_std)
        mae = regression_metrics(y[tests["iu"]], pred[tests["iu"]])["mae"]
        seed_rows.append({"seed": s, "best_epoch": best_ep + 1, "best_val_mae": best_val, "mae_iu_test": mae})
        print(f"  seed {s}: best epoch {best_ep + 1}, val {best_val:.2f}, test(iu) MAE {mae:.2f}")
        if keep is None:
            keep = (s, model, g_mean, g_std, best_ep, best_val, trc, vac, d_static, pred)

    s, model, g_mean, g_std, best_ep, best_val, trc, vac, d_static, pred = keep
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    torch.save({"model_state": model.state_dict(), "seed": s, "epoch": best_ep, "val_mae": best_val,
                "gluc_mean": g_mean, "gluc_std": g_std, "d_static": d_static},
               out / "mlp_best_checkpoint.pt")
    plot_training_curve(trc, vac, best_ep, out / "mlp_training_curve", args.formats)

    rows = []
    groups_id = df["subject_id"].to_numpy()
    for key, m in tests.items():
        r = regression_metrics(y[m], pred[m])
        lo, hi = bootstrap_ci(y[m], pred[m], groups_id[m], n_boot=args.bootstrap) if args.bootstrap else (np.nan, np.nan)
        cz, pz = zone_summary(clarke_zones(y[m], pred[m])), zone_summary(parkes_zones(y[m], pred[m]))
        rows.append({"model": "MaskedDynamicsMLP", "test_set": key, "n_scored": r["n"], "mae": r["mae"],
                     "mae_ci_low": lo, "mae_ci_high": hi, "rmse": r["rmse"], "mard": r["mard"], "r2": r["r2"],
                     "bias": r["bias"], "clarke_AB": cz["A+B"], "clarke_D": cz["D"], "parkes_AB": pz["A+B"]})
    res = pd.DataFrame(rows)
    res.to_csv(out / "mlp_metrics.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(out / "mlp_seed_stability.csv", index=False)

    m = tests["iu"]
    plots.plot_clarke(y[m], pred[m], clarke_zones(y[m], pred[m]), "Clarke error grid: MaskedDynamicsMLP",
                      out / "clarke_mlp", args.formats)
    plots.plot_parkes(y[m], pred[m], parkes_zones(y[m], pred[m]), "Parkes error grid (type 1): MaskedDynamicsMLP",
                      out / "parkes_mlp", args.formats)
    plots.plot_prediction_scatter(y[m], {"MaskedDynamicsMLP": pred[m]}, out / "mlp_prediction_scatter", args.formats)
    plots.plot_glycaemic_subgroup(y[m], pred[m], "MaskedDynamicsMLP by glycaemic band",
                                  out / "mlp_glycaemic_subgroup", args.formats)
    plots.plot_residuals_by_tier(y[m], pred[m], df.loc[m, "record_usability"], out / "mlp_residual_distribution",
                                 args.formats)
    pd.DataFrame({"labevent_id": df["labevent_id"], "y_true": y, "mlp": pred})[tests["all"] | m].to_csv(
        out / "mlp_predictions_test.csv.gz", index=False, compression="gzip")

    lines = ["MASKED-DYNAMICS MLP, LEAK-FREE INPUTS", "=" * 72,
             f"Parameters: {n_params:,} | static width {d_static} | device {DEVICE}",
             f"Reported seed {s}: best epoch {best_ep + 1}, best validation MAE {best_val:.2f} mg/dL",
             f"Target normalisation (TRAIN IDEAL+USABLE): mean {g_mean:.2f}, SD {g_std:.2f} mg/dL", ""]
    for r in res.itertuples():
        lines.append(f"{TEST_SET_LABELS[r.test_set]:52s} n={r.n_scored:6,d}  MAE {r.mae:6.2f} "
                     f"[{r.mae_ci_low:.2f}, {r.mae_ci_high:.2f}]  R2 {r.r2:.3f}  "
                     f"Clarke A+B {r.clarke_AB:.1f}  D {r.clarke_D:.1f}")
    sr = pd.DataFrame(seed_rows)
    lines += ["", f"Seeds {list(sr.seed)}: test(iu) MAE {sr.mae_iu_test.mean():.2f} "
                  f"(range {sr.mae_iu_test.min():.2f}-{sr.mae_iu_test.max():.2f})"]
    (out / "mlp_results.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    config.write_run_info(out, csv, digest, {"seeds": args.seeds, "epochs": args.epochs,
                                             "patience": args.patience, "batch_size": args.batch_size,
                                             "lr": args.lr, "n_params": n_params})


if __name__ == "__main__":
    main()
