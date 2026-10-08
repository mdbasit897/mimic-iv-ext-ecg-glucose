"""
figure6_quality_heatmap.py
════════════════════════════════════════════════════════════════════════════════
Figure 6 — Three-Dimensional Quality Stratification Heatmap

    Gate 1 (X-axis) : Pharmacological Confounding  → decoupling_risk_score
    Gate 2 (Y-axis) : Temporal Alignment Quality   → alignment_quality
    Gate 3 (Panel)  : ECG Signal Quality Index      → sqi_category

════════════════════════════════════════════════════════════════════════════════

USAGE:
    python3 validation/figure6_quality_heatmap.py --csv /path/to/dataset.csv --out ./figures

    # Or edit CSV_PATH below and run directly:
    python3 validation/figure6_quality_heatmap.py
"""

import argparse
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CSV_PATH = os.environ.get("GLUCOECG_CSV", "")   # or pass --csv
FORMATS = ["pdf", "png"]
OUT_DIR = "./results/figures"
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# ── Axis definitions ─────────────────────────────────────────────────────────
SQI_PANELS = ["GOOD", "FAIR", "POOR"]
ALIGN_ORDER = ["TIGHT", "MODERATE", "LOOSE", "EXTENDED"]
RISK_VALUES = [0, 1, 2, 4]  # score=3 does not exist; score=4 = dual intervention

ALIGN_LABELS = [
    "TIGHT\n(≤30 min)",
    "MODERATE\n(31–60 min)",
    "LOOSE\n(61–120 min)",
    "EXTENDED\n(121–240 min)",
]
RISK_LABELS = [
    "0\nNo fast insulin\nor dextrose",
    "1\nDextrose\nonly",
    "2\nFast insulin, or\nbasal + dextrose",
    "4\nFast insulin\n+ dextrose",
]

# ── Usability score map ───────────────────────────────────────────────────────
SCORE_MAP = {"IDEAL": 3, "USABLE": 2, "CAUTION": 1, "EXCLUDE": 0}

# ── Colour map: red (EXCLUDE=0) → orange (CAUTION=1) → blue (USABLE=2) → green (IDEAL=3)
TIER_CMAP = LinearSegmentedColormap.from_list(
    "usability",
    ["#C0392B", "#E67E22", "#2980B9", "#1E8449"],
    N=256,
)

SQI_HEADER_COLORS = {"GOOD": "#1E8449", "FAIR": "#E67E22", "POOR": "#C0392B"}
TIER_LEGEND_COLORS = {
    "IDEAL": "#1E8449",
    "USABLE": "#2980B9",
    "CAUTION": "#E67E22",
    "EXCLUDE": "#C0392B",
}


def load_data(csv_path: str) -> pd.DataFrame:
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    for col in ["gap_exceeds_threshold", "delta_is_intervention_confounded"]:
        if col in df.columns:
            df[col] = df[col].map(lambda x: bool(x) if not isinstance(x, float) else False)
    df["usability_score"] = df["record_usability"].map(SCORE_MAP)
    print(f"  {len(df):,} records | {df['subject_id'].nunique():,} patients")
    return df


def build_cell_matrices(df: pd.DataFrame, sqi: str):
    """
    For a given SQI panel, build three matrices over alignment × risk_score:
      score_matrix : mean usability score (float, NaN if no records)
      count_matrix : record count (int)
      label_matrix : dominant usability tier string
    """
    sqi_df = df[df["sqi_category"] == sqi]

    n_align = len(ALIGN_ORDER)
    n_risk = len(RISK_VALUES)

    score_matrix = np.full((n_align, n_risk), np.nan)
    count_matrix = np.zeros((n_align, n_risk), dtype=int)
    label_matrix = np.full((n_align, n_risk), "", dtype=object)

    for ri, risk in enumerate(RISK_VALUES):
        for ai, align in enumerate(ALIGN_ORDER):
            cell = sqi_df[
                (sqi_df["alignment_quality"] == align) &
                (sqi_df["decoupling_risk_score"] == risk)
                ]
            n = len(cell)
            count_matrix[ai, ri] = n
            if n > 0:
                score_matrix[ai, ri] = cell["usability_score"].mean()
                label_matrix[ai, ri] = cell["record_usability"].value_counts().index[0]

    return score_matrix, count_matrix, label_matrix


def draw_panel(ax, score_matrix, count_matrix, label_matrix,
               sqi: str, show_yticklabels: bool) -> mpl.image.AxesImage:
    """Draw a single SQI panel heatmap and return the image for colorbar."""

    im = ax.imshow(
        score_matrix,
        cmap=TIER_CMAP,
        vmin=0, vmax=3,
        aspect="auto",
        interpolation="nearest",
    )

    n_align, n_risk = score_matrix.shape

    # ── Cell annotations ──────────────────────────────────────────────────────
    for ai in range(n_align):
        for ri in range(n_risk):
            n = count_matrix[ai, ri]
            lbl = label_matrix[ai, ri]

            if n == 0 or lbl == "":
                ax.text(ri, ai, "—", ha="center", va="center",
                        fontsize=10, color="#AAAAAA")
                continue

            score = score_matrix[ai, ri]
            txt_col = "white" if (score <= 1.0 or score >= 2.8) else "#1A1A1A"

            # Tier label (top line, bold)
            ax.text(ri, ai - 0.18, lbl,
                    ha="center", va="center",
                    fontsize=9, fontweight="bold", color=txt_col)

            # Record count (bottom line, smaller)
            ax.text(ri, ai + 0.25, f"n = {n:,}",
                    ha="center", va="center",
                    fontsize=8, color=txt_col, alpha=0.92)

            # Score value (top-right corner, tiny)
            ax.text(ri + 0.45, ai - 0.42, f"{score:.2f}",
                    ha="right", va="top",
                    fontsize=6.5, color=txt_col, alpha=0.75)

    # ── Grid lines between cells ──────────────────────────────────────────────
    for x in np.arange(-0.5, n_risk, 1):
        ax.axvline(x, color="white", linewidth=2.0, zorder=3)
    for y in np.arange(-0.5, n_align, 1):
        ax.axhline(y, color="white", linewidth=2.0, zorder=3)

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xticks(range(n_risk))
    ax.set_xticklabels(RISK_LABELS, fontsize=8.5)
    ax.set_yticks(range(n_align))
    ax.set_yticklabels(ALIGN_LABELS if show_yticklabels else [""] * n_align,
                       fontsize=9)
    ax.tick_params(length=0)

    ax.set_xlabel(
        "Decoupling Risk Score  (Pharmacological Confounding Gate)",
        fontsize=9, labelpad=10,
    )
    if show_yticklabels:
        ax.set_ylabel("Temporal Alignment Quality", fontsize=9, labelpad=8)

    # ── SQI panel title with coloured background ──────────────────────────────
    n_sqi = df_global[df_global["sqi_category"] == sqi].shape[0] if df_global is not None else 0
    ax.set_title(
        f"SQI = {sqi}   (n = {n_sqi:,})\nECG Signal Quality Gate",
        fontsize=10, fontweight="bold", color="white", pad=8,
        bbox=dict(boxstyle="round,pad=0.45",
                  facecolor=SQI_HEADER_COLORS[sqi],
                  edgecolor="none"),
    )

    return im


# Module-level reference for panel title record counts (set in main)
df_global = None


def generate_figure6(df: pd.DataFrame, out_dir: Path) -> None:
    """Full figure 6 — three SQI panels side by side with shared colorbar."""

    global df_global
    # Ensure usability_score column exists regardless of how df was loaded
    if "usability_score" not in df.columns:
        df = df.copy()
        df["usability_score"] = df["record_usability"].map(SCORE_MAP)
    df_global = df

    print("  Generating Figure 6 — 3D Quality Stratification Heatmap...")

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 7),
        gridspec_kw={"width_ratios": [1.18, 1, 1], "wspace": 0.35},
    )
    fig.patch.set_facecolor("white")

    # Shared colourbar on the right
    cbar_ax = fig.add_axes([0.93, 0.18, 0.014, 0.64])

    last_im = None
    for panel_idx, sqi in enumerate(SQI_PANELS):
        score_m, count_m, label_m = build_cell_matrices(df, sqi)
        im = draw_panel(
            axes[panel_idx], score_m, count_m, label_m,
            sqi=sqi,
            show_yticklabels=(panel_idx == 0),
        )
        last_im = im

    # ── Shared colourbar ──────────────────────────────────────────────────────
    norm = mpl.colors.Normalize(vmin=0, vmax=3)
    sm = mpl.cm.ScalarMappable(cmap=TIER_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(
        ["EXCLUDE\n(0)", "CAUTION\n(1)", "USABLE\n(2)", "IDEAL\n(3)"],
        fontsize=9,
    )
    cbar.set_label("Mean Usability Score", fontsize=9, labelpad=10)

    # ── Overall title ─────────────────────────────────────────────────────────
    fig.suptitle(
        "Three-Dimensional Quality Stratification:\n"
        "ECG Signal Quality  ×  Temporal Alignment  ×  Pharmacological Confounding",
        fontsize=12, fontweight="bold", y=1.03,
    )

    # ── Tier legend ───────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(facecolor=TIER_LEGEND_COLORS[t],
                       edgecolor="white", label=t)
        for t in ["IDEAL", "USABLE", "CAUTION", "EXCLUDE"]
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=4,
        fontsize=9.5,
        framealpha=0.92,
        edgecolor="#CCCCCC",
        bbox_to_anchor=(0.46, -0.06),
        title="Dominant Usability Tier per Cell",
        title_fontsize=9,
    )

    # ── Footnote ──────────────────────────────────────────────────────────────
    fig.text(
        0.46, -0.13,
        "decoupling_risk_score = 2 x fast-acting insulin + 1 x dextrose + 1 x dual (insulin + dextrose):  "
        "0 = neither fast-acting insulin nor dextrose (basal insulin alone also scores 0)  |  1 = dextrose only  |  "
        "2 = fast-acting insulin only, or basal insulin + dextrose  |  4 = fast-acting insulin + dextrose\n"
        "Cell colour encodes mean usability score across all records in that (SQI, alignment, risk) cell. "
        "Score shown in top-right corner of each cell. "
        "Dominant tier label and record count (n) shown per cell.",
        ha="center", fontsize=7.5, color="gray", style="italic",
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = out_dir / "figure6_quality_stratification_heatmap.png"
    for fmt in FORMATS:
        plt.savefig(out_path.with_suffix(f".{fmt}"), bbox_inches="tight")
    plt.close()
    print(f"    Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 6 — 3D Quality Stratification Heatmap"
    )
    parser.add_argument("--csv", type=str, default=CSV_PATH)
    parser.add_argument("--out", type=str, default=OUT_DIR)
    parser.add_argument("--formats", nargs="+", default=FORMATS)
    args = parser.parse_args()
    FORMATS[:] = args.formats

    csv_path = Path(args.csv)
    out_dir = Path(args.out)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            f"Set GLUCOECG_CSV or pass --csv /your/path.csv"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {out_dir.resolve()}\n")

    df = load_data(str(csv_path))
    generate_figure6(df, out_dir)

    print(f"\nDone → {out_dir / 'figure6_quality_stratification_heatmap.png'}\n")


if __name__ == "__main__":
    main()