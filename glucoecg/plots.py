"""Shared figures for the baseline scripts. Every function saves one figure in each format."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .metrics import GLYCAEMIC_BANDS, ZONES, _parkes_type1_polygons, regression_metrics, zone_summary  # noqa: E402

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

ZONE_COLORS = {"A": "#1E8449", "B": "#2980B9", "C": "#B7950B", "D": "#E67E22", "E": "#C0392B"}
GREY, ACCENT = "#9E9E9E", "#0072B2"


def save(fig, stem: Path, formats=("pdf", "png")):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(stem.with_suffix(f".{fmt}"))
    plt.close(fig)


def _zone_scatter(ax, ref, pred, zones, lim):
    for z in ZONES:
        m = zones == z
        if m.any():
            ax.scatter(ref[m], pred[m], s=3, alpha=0.35, color=ZONE_COLORS[z], rasterized=True,
                       label=f"{z}: {m.sum():,} ({m.mean() * 100:.1f}%)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Reference glucose (mg/dL)")
    ax.set_ylabel("Predicted glucose (mg/dL)")
    ax.legend(loc="upper left", fontsize=8, markerscale=4, framealpha=0.9)


def plot_clarke(ref, pred, zones, title: str, stem: Path, formats=("pdf", "png")) -> None:
    ref, pred, zones = np.asarray(ref, float), np.asarray(pred, float), np.asarray(zones)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    _zone_scatter(ax, ref, pred, zones, 400)
    k = "k"
    for xs, ys in [([0, 400], [0, 400]), ([0, 175 / 3], [70, 70]), ([175 / 3, 400 / 1.2], [70, 400]),
                   ([70, 70], [84, 400]), ([0, 70], [180, 180]), ([70, 290], [180, 400]),
                   ([70, 70], [0, 56]), ([70, 400], [56, 320]), ([180, 180], [0, 70]),
                   ([180, 400], [70, 70]), ([240, 240], [70, 180]), ([240, 400], [180, 180]),
                   ([130, 180], [0, 70])]:
        ax.plot(xs, ys, color=k, lw=0.9, ls=":" if xs == [0, 400] else "-")
    s = zone_summary(zones)
    hidden = int(((ref > 400) | (pred > 400)).sum())
    ax.text(0.98, 0.02, f"A+B {s['A+B']:.1f}%   D {s['D']:.1f}%   E {s['E']:.2f}%\n"
            f"n = {len(ref):,} ({hidden:,} beyond 400 mg/dL not drawn)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7"))
    ax.set_title(title, fontweight="bold")
    save(fig, stem, formats)


def plot_parkes(ref, pred, zones, title: str, stem: Path, formats=("pdf", "png")) -> None:
    ref, pred, zones = np.asarray(ref, float), np.asarray(pred, float), np.asarray(zones)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    _zone_scatter(ax, ref, pred, zones, 550)
    for _, poly in _parkes_type1_polygons(550, 550):
        v = poly.vertices
        ax.plot(v[:, 0], v[:, 1], color="k", lw=0.8)
    s = zone_summary(zones)
    ax.text(0.98, 0.02, f"A+B {s['A+B']:.1f}%   C {s['C']:.1f}%   D {s['D']:.1f}%   E {s['E']:.2f}%",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7"))
    ax.set_title(title, fontweight="bold")
    save(fig, stem, formats)


def plot_prediction_scatter(ref, preds: dict, stem: Path, formats=("pdf", "png")) -> None:
    """One panel per model: predicted vs reference with identity line and metrics."""
    ref = np.asarray(ref, float)
    n = len(preds)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.6), squeeze=False)
    for ax, (label, p) in zip(axes[0], preds.items()):
        m = regression_metrics(ref, p)
        ax.scatter(ref, p, s=2, alpha=0.25, color=ACCENT, rasterized=True)
        ax.plot([0, 500], [0, 500], color="k", lw=0.8, ls="--")
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 500)
        ax.set_aspect("equal")
        ax.set_title(f"{label}\nMAE {m['mae']:.1f}  R² {m['r2']:.2f}", fontsize=9)
        ax.set_xlabel("Reference (mg/dL)")
    axes[0][0].set_ylabel("Predicted (mg/dL)")
    save(fig, stem, formats)


def plot_glycaemic_subgroup(ref, pred, title: str, stem: Path, formats=("pdf", "png")) -> dict:
    """MAE and bias by glycaemic band; returns the metrics."""
    ref, pred = np.asarray(ref, float), np.asarray(pred, float)
    res = {k: regression_metrics(ref[f(ref)], pred[f(ref)]) for k, f in GLYCAEMIC_BANDS.items()}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    names = list(res)
    for ax, key, lab in [(axes[0], "mae", "MAE (mg/dL)"), (axes[1], "bias", "Mean bias, predicted − reference (mg/dL)")]:
        vals = [res[k][key] for k in names]
        ax.bar(names, vals, color=[GREY, ACCENT, GREY], width=0.55)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:+.1f}" if key == "bias" else f"{v:.1f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel(lab)
        ax.set_xticks(range(len(names)), [f"{k}\nn={res[k]['n']:,}" for k in names])
    fig.suptitle(title, fontweight="bold")
    save(fig, stem, formats)
    return res


def plot_residuals_by_tier(ref, pred, tiers, stem: Path, formats=("pdf", "png")) -> None:
    ref, pred, tiers = np.asarray(ref, float), np.asarray(pred, float), np.asarray(tiers)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(-200, 200, 81)
    for t, c in [("USABLE", GREY), ("IDEAL", ACCENT)]:
        r = (pred - ref)[tiers == t]
        if len(r):
            ax.hist(r, bins=bins, density=True, alpha=0.6, color=c,
                    label=f"{t} (n={len(r):,}, bias {r.mean():+.1f}, SD {r.std():.1f})")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Residual, predicted − reference (mg/dL)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    save(fig, stem, formats)


def plot_feature_importance(model, names, title: str, stem: Path, top: int = 20,
                            formats=("pdf", "png")) -> None:
    gain = model.get_booster().get_score(importance_type="gain")
    pairs = sorted(((names[int(k[1:])] if k.startswith("f") and k[1:].isdigit() else k, v)
                    for k, v in gain.items()), key=lambda kv: kv[1])[-top:]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(pairs) + 1.2))
    ax.barh([p[0] for p in pairs], [p[1] for p in pairs], color=ACCENT)
    ax.set_xlabel("XGBoost gain")
    ax.set_title(title, fontweight="bold")
    save(fig, stem, formats)


def plot_results_table(rows: list, columns: list, title: str, stem: Path, formats=("pdf", "png")) -> None:
    """Render a list of row lists as a table figure."""
    lengths = [max(len(str(r[i])) for r in rows + [columns]) for i in range(len(columns))]
    widths = [n * 0.0062 + 0.025 for n in lengths]
    fig, ax = plt.subplots(figsize=(sum(widths) * 15, 0.32 * len(rows) + 0.9))
    ax.axis("off")
    tab = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center", colWidths=widths)
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.5)
    tab.scale(1, 1.4)
    for (r, c), cell in tab.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#EEEEEE")
        if c == 0:
            cell.set_text_props(ha="left")
            cell.PAD = 0.02
    ax.set_title(title, fontweight="bold")
    save(fig, stem, formats)
