"""
generate_figures.py
════════════════════════════════════════════════════════════════════════════════
Generates 5 figures in high-resolution PDF (300 DPI):
  Figure 1 — PRISMA-style Pipeline Attrition Flowchart
  Figure 2 — Temporal Integrity (ECG–glucose offset + inter-measurement gap)
  Figure 3 — Glucose by Glycaemic Class (boxplots; v1.0 z-score panel removed)
  Figure 4 — QTc Formula Divergence by HR Regime (scatter + violin)
  Figure 5 — ECG Features across Glucose Trajectory Classes
════════════════════════════════════════════════════════════════════════════════

USAGE:
    python3 validation/generate_figures.py --csv /path/to/mimiciv_ecg_glucose_aligned.csv --out results/figures

    # or, with GLUCOECG_CSV set:
    python3 validation/generate_figures.py
"""

import argparse
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for headless servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glucoecg.features import clean_ecg  # noqa: E402

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# ▶▶  SET YOUR PATHS HERE  ◀◀
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH = os.environ.get("GLUCOECG_CSV", "")   # or pass --csv
OUT_DIR  = "./results/figures"                  # output folder (created if missing)
FORMATS  = ["pdf", "png"]                       # overridden by --formats
# ─────────────────────────────────────────────────────────────────────────────

# ── Wong (2011) colorblind-safe palette ──────────────────────────────────────
WONG = {
    "black":    "#000000",
    "orange":   "#E69F00",
    "sky":      "#56B4E9",
    "green":    "#009E73",
    "yellow":   "#F0E442",
    "blue":     "#0072B2",
    "vermilion":"#D55E00",
    "pink":     "#CC79A7",
}

# Glycaemic class colour map (clinically intuitive: green=normal, red=severe)
GLYCEMIC_COLORS = {
    "SEVERE_HYPO":  WONG["vermilion"],
    "HYPO":         WONG["orange"],
    "EUGLYCEMIC":   WONG["green"],
    "HYPER":        WONG["sky"],
    "SEVERE_HYPER": WONG["blue"],
}

GLYCEMIC_ORDER = ["SEVERE_HYPO", "HYPO", "EUGLYCEMIC", "HYPER", "SEVERE_HYPER"]
GLYCEMIC_LABELS = {
    "SEVERE_HYPO":  "Severe\nHypo\n(<54)",
    "HYPO":         "Hypo\n(54–69)",
    "EUGLYCEMIC":   "Euglycaemic\n(70–180)",
    "HYPER":        "Hyper\n(181–250)",
    "SEVERE_HYPER": "Severe\nHyper\n(>250)",
}

HR_REGIME_ORDER  = ["BRADYCARDIA", "LOW_NORMAL", "NORMAL", "MILD_TACHY", "TACHYCARDIA"]
HR_REGIME_LABELS = ["Bradycardia\n(<50)", "Low Normal\n(50–60)",
                    "Normal\n(60–90)", "Mild Tachy\n(90–120)", "Tachycardia\n(>120)"]
HR_COLORS = [WONG["blue"], WONG["sky"], WONG["green"], WONG["orange"], WONG["vermilion"]]

TRAJ_ORDER  = ["RAPID_FALL", "MODERATE_FALL", "STABLE", "MODERATE_RISE", "RAPID_RISE"]
TRAJ_LABELS = ["Rapid Fall\n(<−60)", "Moderate\nFall\n(−60→−20)",
               "Stable\n(±20)", "Moderate\nRise\n(+20→+60)", "Rapid Rise\n(>+60)"]
TRAJ_COLORS = [WONG["vermilion"], WONG["orange"], WONG["green"],
               WONG["sky"], WONG["blue"]]

# Global style
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.labelsize":    10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches":0.05,
})


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _save(out_path: Path, **kw) -> None:
    """Save the current figure once per format in FORMATS (suffix of out_path is replaced)."""
    for fmt in FORMATS:
        plt.savefig(Path(out_path).with_suffix(f".{fmt}"), **kw)


def _retained_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Records kept in the dataset (not EXCLUDE), with ECG placeholder codes and the
    intervals derived from them set to missing (glucoecg.features.clean_ecg)."""
    return clean_ecg(df[df["record_usability"] != "EXCLUDE"].copy())


def load_data(csv_path: str) -> pd.DataFrame:
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    # Fix mixed-type booleans from BigQuery export
    for col in ["gap_exceeds_threshold", "delta_is_intervention_confounded"]:
        if col in df.columns:
            df[col] = df[col].map(lambda x: bool(x) if not isinstance(x, float) else False)
    print(f"  Loaded {len(df):,} records | {df['subject_id'].nunique():,} patients")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — PRISMA-STYLE PIPELINE ATTRITION FLOWCHART
# ─────────────────────────────────────────────────────────────────────────────

def figure1_prisma(df: pd.DataFrame, out_dir: Path) -> None:
    """
    PRISMA-style flowchart showing the full data attrition pipeline from
    raw MIMIC-IV records down to the final quality-stratified dataset.
    """
    print("  Generating Figure 1 — PRISMA Flowchart...")

    # ── Compute counts ────────────────────────────────────────────────────────
    total          = len(df)
    n_patients     = df["subject_id"].nunique()
    n_ecg_studies  = df["ecg_study_id"].nunique()
    # EXCLUDE gates are applied in priority order (as in the SQL CASE statement):
    # dual intervention -> POOR SQI -> bradycardia QTc divergence. Counts below are
    # the records removed at each step, so they add up to the EXCLUDE total.
    is_interv      = df["intervention_status"] == "EXCLUDE"
    is_poor        = df["sqi_category"] == "POOR"
    is_brady       = (df["hr_regime"] == "BRADYCARDIA") & (df["qtc_formula_range_ms"] > 200)
    n_interv_excl  = int(is_interv.sum())
    n_poor_sqi     = int((is_poor & ~is_interv).sum())
    n_brady_qtc    = int((is_brady & ~is_interv & ~is_poor).sum())
    n_exclude      = (df["record_usability"] == "EXCLUDE").sum()
    n_ideal        = (df["record_usability"] == "IDEAL").sum()
    n_usable       = (df["record_usability"] == "USABLE").sum()
    n_caution      = (df["record_usability"] == "CAUTION").sum()
    n_retained     = n_ideal + n_usable + n_caution

    fig, ax = plt.subplots(figsize=(10, 13))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis("off")

    fig.patch.set_facecolor("white")

    def draw_box(ax, x, y, w, h, text, color="#EBF5FB", edge="#1A5276",
                 fontsize=9, bold=False, radius=0.2):
        box = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle=f"round,pad=0.05,rounding_size={radius}",
            facecolor=color, edgecolor=edge, linewidth=1.5, zorder=3
        )
        ax.add_patch(box)
        weight = "bold" if bold else "normal"
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                fontweight=weight, color="#1A1A1A", zorder=4,
                wrap=True, multialignment="center",
                linespacing=1.4)

    def draw_arrow(ax, x, y_start, y_end, color="#1A5276"):
        ax.annotate("", xy=(x, y_end + 0.05),
                    xytext=(x, y_start - 0.05),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=1.5, mutation_scale=14),
                    zorder=2)

    def draw_exclusion(ax, x_main, y, text, color="#FADBD8", edge="#922B21"):
        """Side exclusion box with horizontal arrow from main flow."""
        ax.annotate("", xy=(7.8, y), xytext=(x_main + 1.25, y),
                    arrowprops=dict(arrowstyle="-|>", color="#922B21",
                                   lw=1.2, mutation_scale=12), zorder=2)
        box = FancyBboxPatch(
            (7.8, y - 0.42), 2.0, 0.84,
            boxstyle="round,pad=0.05,rounding_size=0.15",
            facecolor=color, edgecolor=edge, linewidth=1.2, zorder=3
        )
        ax.add_patch(box)
        ax.text(8.8, y, text, ha="center", va="center", fontsize=8,
                color="#922B21", zorder=4, multialignment="center")

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.text(5, 12.6, "MIMIC-IV-Ext-ECG-Glucose: Pipeline Attrition",
            ha="center", va="center", fontsize=12, fontweight="bold",
            color="#1A5276")

    # ── Box 0: Data sources ───────────────────────────────────────────────────
    draw_box(ax, 5, 11.8, 7.5, 0.9,
             f"MIMIC-IV Hospital + ICU + MIMIC-IV-ECG\n"
             f"Temporal join: blood glucose labs ↔ ECG recordings (±240 min window)\n"
             f"{total:,} glucose–ECG pairs | {n_patients:,} patients | {n_ecg_studies:,} ECG studies",
             color="#D6EAF8", edge="#1A5276", fontsize=8.5, bold=False)

    # ── Arrow 0→1 ─────────────────────────────────────────────────────────────
    draw_arrow(ax, 5, 11.35, 10.75)

    # ── Box 1: Intervention gate (applied first) ──────────────────────────────
    draw_box(ax, 5, 10.35, 7.5, 0.75,
             f"Step 1. Pharmacological Intervention Gate (ICU inputevents)\n"
             f"Fast insulin (±120 min) · Basal insulin (±240 min) · IV Dextrose (±60 min)\n"
             f"CLEAN: {(df['intervention_status']=='CLEAN').sum():,}  |  "
             f"FLAG_MOD: {(df['intervention_status']=='FLAG_MODERATE_RISK').sum():,}  |  "
             f"FLAG_HIGH: {(df['intervention_status']=='FLAG_HIGH_RISK').sum():,}  |  "
             f"dual: {n_interv_excl:,}",
             color="#EBF5FB", fontsize=8.5)
    draw_exclusion(ax, 5, 10.35,
                   f"Dual intervention\n{n_interv_excl:,} excluded")

    draw_arrow(ax, 5, 9.97, 9.45)

    # ── Box 2: SQI gate (among records still retained) ────────────────────────
    draw_box(ax, 5, 9.05, 7.5, 0.75,
             f"Step 2. Signal Quality Index (SQI) Gate\n"
             f"Interval plausibility · Internal consistency · Report artifact scan\n"
             f"GOOD: {(df['sqi_category']=='GOOD').sum():,}  |  "
             f"FAIR: {(df['sqi_category']=='FAIR').sum():,}  |  "
             f"POOR: {is_poor.sum():,} ({n_poor_sqi:,} not already excluded)",
             color="#EBF5FB", fontsize=8.5)
    draw_exclusion(ax, 5, 9.05,
                   f"SQI POOR\n{n_poor_sqi:,} excluded")

    draw_arrow(ax, 5, 8.67, 8.15)

    # ── Box 3: Bradycardia QTc gate ───────────────────────────────────────────
    draw_box(ax, 5, 7.75, 7.5, 0.75,
             f"Step 3. Bradycardia QTc Divergence Gate\n"
             f"BRADYCARDIA + qtc_formula_range_ms > 200 ms → erroneous RR interval\n"
             f"{is_brady.sum():,} records meet the criterion; {n_brady_qtc:,} not already excluded",
             color="#EBF5FB", fontsize=8.5)
    draw_exclusion(ax, 5, 7.75,
                   f"Brady QTc artefact\n{n_brady_qtc:,} excluded")

    draw_arrow(ax, 5, 7.37, 6.85)

    # ── Box 4: Total excluded ─────────────────────────────────────────────────
    draw_box(ax, 5, 6.45, 7.5, 0.75,
             f"Total EXCLUDED: {n_exclude:,} records "
             f"(= {n_interv_excl:,} + {n_poor_sqi:,} + {n_brady_qtc:,})\n"
             f"Retained for analysis: {n_retained:,} records | "
             f"{df[df['record_usability']!='EXCLUDE']['subject_id'].nunique():,} patients",
             color="#D5F5E3", edge="#1E8449", fontsize=8.5, bold=False)

    draw_arrow(ax, 5, 6.07, 5.55)

    # ── Box 5: Usability tiers ────────────────────────────────────────────────
    draw_box(ax, 5, 5.15, 7.5, 0.75,
             f"Record Usability Stratification (composite 4-tier flag)\n"
             f"ECG direction · Temporal alignment · Intervention status · SQI · dG/dt validity",
             color="#EBF5FB", fontsize=8.5)

    draw_arrow(ax, 5, 4.77, 4.2)

    # ── Four tier boxes (IDEAL / USABLE / CAUTION / EXCLUDE) ────────────────
    tier_data = [
        (1.4,  "IDEAL",   n_ideal,   "#D5F5E3", "#1E8449",
         "ECG before glucose\nTIGHT/MODERATE alignment\nCLEAN intervention\nGOOD SQI\ndG/dt available & valid"),
        (3.7,  "USABLE",  n_usable,  "#EBF5FB", "#1A5276",
         "All other records\n(e.g. first draw in stay,\nLOOSE/EXTENDED offset,\nECG after glucose)"),
        (6.0,  "CAUTION", n_caution, "#FEF9E7", "#B7950B",
         "FLAG_HIGH_RISK, FAIR SQI,\nconfounded delta, or\ngap <15 min / >6 hr"),
        (8.6,  "EXCLUDE", n_exclude, "#FADBD8", "#922B21",
         "SQI POOR\nor dual intervention\nor Brady QTc artefact"),
    ]

    for xc, label, count, fc, ec, desc in tier_data:
        pct = count / total * 100
        draw_box(ax, xc, 3.25, 1.95, 1.9,
                 f"{label}\n{count:,}\n({pct:.1f}%)\n\n{desc}",
                 color=fc, edge=ec, fontsize=7.5)
        # Arrow from stratification box down to tier boxes
        ax.annotate("", xy=(xc, 4.2), xytext=(5, 4.2),
                    arrowprops=dict(arrowstyle="-|>", color=ec,
                                   lw=1.2, mutation_scale=12), zorder=2)

    # ── Train/Val/Test split note ─────────────────────────────────────────────
    n_train = (df["split"] == "TRAIN").sum()
    n_val   = (df["split"] == "VALIDATION").sum()
    n_test  = (df["split"] == "TEST").sum()

    draw_box(ax, 5, 1.2, 7.5, 1.3,
             f"Subject-Level 80/10/10 Train / Validation / Test Split\n"
             f"FARM_FINGERPRINT hashing on subject_id — zero patient leakage\n\n"
             f"TRAIN: {n_train:,} records  |  "
             f"VALIDATION: {n_val:,} records  |  "
             f"TEST: {n_test:,} records\n"
             f"({df[df['split']=='TRAIN']['subject_id'].nunique():,} / "
             f"{df[df['split']=='VALIDATION']['subject_id'].nunique():,} / "
             f"{df[df['split']=='TEST']['subject_id'].nunique():,} patients)",
             color="#F4ECF7", edge="#6C3483", fontsize=8.5)


    plt.tight_layout()
    out_path = out_dir / "figure1_prisma_flowchart.png"
    _save(out_path)
    plt.close()
    print(f"    Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — TEMPORAL INTEGRITY
# ─────────────────────────────────────────────────────────────────────────────

def figure2_temporal(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Two-panel figure showing:
      Panel A: Distribution of ECG–glucose temporal offset (abs_offset_minutes)
               coloured by alignment_quality tier
      Panel B: Distribution of inter-measurement gap (inter_measurement_gap_hr)
               with the 6-hour upper gate and 15-min lower gate annotated
    """
    print("  Generating Figure 2 — Temporal Integrity...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("white")

    # ── Panel A: ECG–Glucose offset coloured by alignment quality ─────────────
    ax = axes[0]
    align_colors = {
        "TIGHT":    WONG["green"],
        "MODERATE": WONG["sky"],
        "LOOSE":    WONG["orange"],
        "EXTENDED": WONG["vermilion"],
    }
    align_order = ["TIGHT", "MODERATE", "LOOSE", "EXTENDED"]
    bins = np.linspace(0, 240, 49)

    bottom = np.zeros(len(bins) - 1)
    for tier in align_order:
        subset = df[df["alignment_quality"] == tier]["abs_offset_minutes"].dropna()
        counts, _ = np.histogram(subset, bins=bins)
        ax.bar(bins[:-1], counts, width=(bins[1]-bins[0]),
               bottom=bottom, color=align_colors[tier],
               label=f"{tier} (n={len(subset):,})",
               alpha=0.9, edgecolor="white", linewidth=0.3)
        bottom += counts

    # Threshold lines
    ax.axvline(30,  color=WONG["green"],    lw=1.5, ls="--", alpha=0.8)
    ax.axvline(60,  color=WONG["sky"],      lw=1.5, ls="--", alpha=0.8)
    ax.axvline(120, color=WONG["orange"],   lw=1.5, ls="--", alpha=0.8)
    for x, label, c in [(30, "TIGHT", WONG["green"]),
                         (60, "MODERATE", WONG["sky"]),
                         (120, "LOOSE", WONG["orange"])]:
        ax.text(x+2, ax.get_ylim()[1]*0.92 if ax.get_ylim()[1] > 0 else 100,
                label, fontsize=7.5, color=c, va="top")

    # Median annotation
    med = df["abs_offset_minutes"].median()
    ax.axvline(med, color="black", lw=1.5, ls=":")
    ax.text(med+3, 0.72, f"Median\n{med:.0f} min",
            transform=ax.get_xaxis_transform(), fontsize=8, color="black")

    ax.set_xlabel("ECG–Glucose Temporal Offset (minutes)")
    ax.set_ylabel("Number of Records")
    ax.set_title("A  ECG–Glucose Temporal Offset Distribution")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax.set_xlim(0, 242)

    # ── Panel B: Inter-measurement gap with gates annotated ───────────────────
    ax = axes[1]
    gap = df["inter_measurement_gap_hr"].dropna()

    # FIX: Do NOT clip >24hr values into the last bin (creates artificial spike).
    # Instead, build histogram only from records naturally within 0-24hr range,
    # so the distribution decays naturally without a wall of truncated data.
    # Records >24hr are reported in the annotation box but not plotted.
    gap_within_24 = gap[gap <= 24]
    gap_above_24  = gap[gap > 24]
    pct_within_24 = len(gap_within_24) / len(gap) * 100

    # Use explicit bin edges so no values fall outside the plotted range
    bin_edges = np.linspace(0, 24, 73)   # 72 bins × 20 min each
    ax.hist(gap_within_24, bins=bin_edges, color=WONG["blue"], alpha=0.8,
            edgecolor="white", linewidth=0.3)

    # Gate annotations
    ax.axvline(0.25, color=WONG["vermilion"], lw=2, ls="--")  # 15 min lower gate
    ax.axvline(6.0,  color=WONG["orange"],   lw=2, ls="--")  # 6 hr upper gate

    y_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 100
    ax.text(0.27, y_top * 0.92, "Lower\ngate\n(15 min)",
            fontsize=7.5, color=WONG["vermilion"], va="top")
    ax.text(6.15, y_top * 0.92, "Upper\ngate\n(6 hr)",
            fontsize=7.5, color=WONG["orange"], va="top")

    # Shade valid zone
    ax.axvspan(0.25, 6.0, alpha=0.08, color=WONG["green"], label="Valid rate window")

    # Stats annotation
    valid = gap[(gap >= 0.25) & (gap <= 6.0)]
    ax.text(0.97, 0.97,
            f"Total gaps: {len(gap):,}\n"
            f"Valid (15 min–6 hr): {len(valid):,} ({len(valid)/len(gap)*100:.1f}%)\n"
            f"Shown (≤24 hr): {len(gap_within_24):,} ({pct_within_24:.1f}%)\n"
            f"Not plotted (>24 hr): {len(gap_above_24):,} ({len(gap_above_24)/len(gap)*100:.1f}%)\n"
            f"Median (all gaps): {gap.median():.1f} hr",
            transform=ax.transAxes, fontsize=7.8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.85))

    ax.set_xlim(0, 24)   # FIX: natural decay — no spike from binning truncated values
    ax.set_xlabel("Inter-Measurement Gap (hours)")
    ax.set_ylabel("Number of Records")
    ax.set_title(f"B  Inter-Measurement Gap Distribution\n"
                 f"(≤24 hr shown; {len(gap_above_24):,} gaps >24 hr not plotted)")
    ax.legend(loc="upper center", framealpha=0.9, fontsize=8)

    plt.tight_layout(w_pad=3)
    out_path = out_dir / "figure2_temporal_integrity.png"
    _save(out_path)
    plt.close()
    print(f"    Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — GLUCOSE PHENOTYPING BY GLYCAEMIC CLASS
# ─────────────────────────────────────────────────────────────────────────────

def figure3_phenotyping(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Raw glucose (mg/dL) boxplots by glycaemic class with clinical threshold lines.

    The v1.0 panel B (glucose_z_score by class) was removed: glucose_z_score is
    built from whole-record patient statistics that include the plotted value,
    so it cannot validate anything.
    """
    print("  Generating Figure 3 — Glucose Phenotyping...")

    fig, axes = plt.subplots(1, 1, figsize=(7.5, 6), squeeze=False)
    axes = axes[0]
    fig.patch.set_facecolor("white")

    for panel_idx, (col, title, ylabel, unit) in enumerate([
        ("glucose_mg_dl",  "Blood Glucose by Glycaemic Class",
         "Blood Glucose (mg/dL)", "mg/dL"),
    ]):
        ax = axes[panel_idx]
        data_by_class = [
            df[df["glycemic_class"] == cls][col].dropna().values
            for cls in GLYCEMIC_ORDER
        ]
        counts = [
            len(df[df["glycemic_class"] == cls])
            for cls in GLYCEMIC_ORDER
        ]

        bp = ax.boxplot(
            data_by_class,
            patch_artist=True,
            notch=True,
            showfliers=True,
            flierprops=dict(marker="o", markersize=2, alpha=0.3, linestyle="none"),
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            widths=0.55,
        )

        for patch, cls in zip(bp["boxes"], GLYCEMIC_ORDER):
            patch.set_facecolor(GLYCEMIC_COLORS[cls])
            patch.set_alpha(0.82)

        # Clinical threshold lines (Panel A only)
        if col == "glucose_mg_dl":
            thresholds = [(54, "Severe Hypo", WONG["vermilion"]),
                          (70, "Hypo",        WONG["orange"]),
                          (180, "Hyper",       WONG["sky"]),
                          (250, "Severe Hyper",WONG["blue"])]
            for thresh, label, color in thresholds:
                ax.axhline(thresh, color=color, lw=1.2, ls="--", alpha=0.7)
                ax.text(5.55, thresh + 3, label, fontsize=7.5, color=color, va="bottom")

        # X-axis labels with counts
        xlabels = [
            f"{GLYCEMIC_LABELS[cls]}\nn={counts[i]:,}"
            for i, cls in enumerate(GLYCEMIC_ORDER)
        ]
        ax.set_xticks(range(1, len(GLYCEMIC_ORDER) + 1))
        ax.set_xticklabels(xlabels, fontsize=8.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # Colour-coded x-tick labels
        for tick, cls in zip(ax.get_xticklabels(), GLYCEMIC_ORDER):
            tick.set_color(GLYCEMIC_COLORS[cls])
            tick.set_fontweight("bold")

    plt.tight_layout(w_pad=3)
    out_path = out_dir / "figure3_glucose_phenotyping.png"
    _save(out_path)
    plt.close()
    print(f"    Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — QTc FORMULA DIVERGENCE BY HR REGIME
# ─────────────────────────────────────────────────────────────────────────────

def figure4_qtc_divergence(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Three-panel figure:
      Panel A: Scatter — HR vs each of the 4 QTc formulas (colour = formula)
               annotated with mean per HR regime
      Panel B: Violin — QTc formula range (divergence) by HR regime
               justifying Fridericia as the recommended formula
      Panel C: Bar — Mean QTc per formula per HR regime
    """
    print("  Generating Figure 4 — QTc Formula Divergence...")
    # Panels A and C: retained records with cleaned ECG values. Panel B keeps all
    # records, because it shows the divergence that motivates the bradycardia gate.
    ret = _retained_clean(df)

    fig = plt.figure(figsize=(16, 6))
    gs  = gridspec.GridSpec(1, 3, width_ratios=[2.2, 1.4, 2.2], wspace=0.32)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    fig.patch.set_facecolor("white")

    qtc_formulas = {
        "Bazett":      ("qtc_bazett_ms",     WONG["vermilion"]),
        "Fridericia":  ("qtc_fridericia_ms",  WONG["blue"]),
        "Framingham":  ("qtc_framingham_ms",  WONG["green"]),
        "Hodges":      ("qtc_hodges_ms",      WONG["orange"]),
    }

    # ── Panel A: HR vs QTc scatter ────────────────────────────────────────────
    ax = axes[0]
    sample = ret.sample(min(3000, len(ret)), random_state=42)  # subsample for clarity
    for label, (col, color) in qtc_formulas.items():
        valid = sample[[col, "heart_rate_bpm"]].dropna()
        ax.scatter(valid["heart_rate_bpm"], valid[col],
                   c=color, alpha=0.25, s=8, label=label, rasterized=True)

    # Normal QTc band
    ax.axhspan(350, 460, alpha=0.06, color=WONG["green"], label="Normal QTc range")
    ax.axhline(500, color="black", lw=1.2, ls="--", alpha=0.8)
    ax.text(155, 505, "500 ms (Torsades threshold)", fontsize=7.5,
            ha="right", color="black")

    # HR regime boundaries
    for hr, label in [(50, ""), (60, ""), (90, ""), (120, "")]:
        ax.axvline(hr, color="gray", lw=0.8, ls=":", alpha=0.5)

    ax.set_xlabel("Heart Rate (bpm)")
    ax.set_ylabel("QTc (ms)")
    ax.set_title("A  Heart Rate vs QTc (retained records)")
    ax.set_xlim(0, 160)
    ax.set_ylim(200, 700)
    ax.legend(fontsize=8, markerscale=2, framealpha=0.9, loc="upper right")

    # ── Panel B: QTc formula range by HR regime (violin) ─────────────────────
    # FIX: Raise cap from 150ms to 500ms so the Bradycardia violin body is
    # fully visible — at 150ms it was cut off and looked like missing data.
    # The Bradycardia mean of ~251ms is the key clinical finding here.
    ax = axes[1]
    cap_val = 500  # generous upper bound — shows full Brady distribution
    range_data = [
        df[df["hr_regime"] == r]["qtc_formula_range_ms"].dropna().clip(upper=cap_val).values
        for r in HR_REGIME_ORDER
    ]
    means = [df[df["hr_regime"] == r]["qtc_formula_range_ms"].dropna().mean()
             for r in HR_REGIME_ORDER]

    parts = ax.violinplot(range_data, positions=range(len(HR_REGIME_ORDER)),
                          showmedians=True, showextrema=False)
    for i, (pc, color) in enumerate(zip(parts["bodies"], HR_COLORS)):
        pc.set_facecolor(color)
        pc.set_alpha(0.75)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)

    # Mean markers
    ax.scatter(range(len(HR_REGIME_ORDER)), means, color="black",
               s=40, zorder=5, marker="D")

    # Annotate means — position above violin body, clamp to cap for display
    for i, (m, r) in enumerate(zip(means, HR_REGIME_ORDER)):
        display_y = min(m + 8, cap_val - 20)   # keep label inside axes
        ax.text(i, display_y, f"{m:.0f} ms",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                color=HR_COLORS[i])

    ax.axhline(30, color="black", lw=1.2, ls="--", alpha=0.7)
    ax.text(4.55, 33, "30 ms\n(Bazett unreliable)", fontsize=7, ha="right",
            va="bottom", color="black")

    ax.set_xticks(range(len(HR_REGIME_ORDER)))
    ax.set_xticklabels(["Brady", "Low\nNorm", "Normal", "Mild\nTachy", "Tachy"],
                       fontsize=8)
    ax.set_ylabel("QTc Formula Range — max−min (ms)")
    ax.set_title("B  Inter-Formula Divergence by HR Regime\n(all records, before exclusion)")
    ax.set_ylim(0, cap_val)
    ax.text(0.5, 0.97, f"(values clipped at {cap_val} ms for display; "
            f"bradycardia mean = {means[0]:.0f} ms)",
            transform=ax.transAxes, fontsize=7, ha="center", va="top",
            color="gray")

    # ── Panel C: Mean QTc per formula per HR regime (grouped bars) ────────────
    ax = axes[2]
    x       = np.arange(len(HR_REGIME_ORDER))
    n_forms = len(qtc_formulas)
    width   = 0.18

    for i, (label, (col, color)) in enumerate(qtc_formulas.items()):
        means_per_regime = [
            ret[ret["hr_regime"] == r][col].dropna().mean()
            for r in HR_REGIME_ORDER
        ]
        offset = (i - n_forms / 2 + 0.5) * width
        bars = ax.bar(x + offset, means_per_regime, width,
                      label=label, color=color, alpha=0.85,
                      edgecolor="white", linewidth=0.5)

    ax.axhline(460, color="black", lw=1, ls="--", alpha=0.7)
    ax.text(4.55, 463, "460 ms\n(prolongation)", fontsize=7,
            ha="right", va="bottom", color="black")
    ax.axhline(500, color=WONG["vermilion"], lw=1, ls="--", alpha=0.8)
    ax.text(4.55, 503, "500 ms\n(Torsades)", fontsize=7,
            ha="right", va="bottom", color=WONG["vermilion"])

    ax.set_xticks(x)
    ax.set_xticklabels(HR_REGIME_LABELS, fontsize=8)
    ax.set_ylabel("Mean QTc (ms)")
    ax.set_title("C  Mean QTc by Formula and HR Regime\n"
                 "(retained records; Fridericia = recommended)")
    ax.legend(fontsize=8, framealpha=0.9, loc="upper left")
    ax.set_ylim(300, 560)

    out_path = out_dir / "figure4_qtc_divergence.png"
    _save(out_path)
    plt.close()
    print(f"    Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — ECG FEATURES ACROSS GLUCOSE TRAJECTORY CLASSES
# ─────────────────────────────────────────────────────────────────────────────

def figure5_trajectory(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Three-panel figure showing how ECG features vary across glucose trajectory
    classes (STABLE / MODERATE_FALL / RAPID_FALL / MODERATE_RISE / RAPID_RISE).

    Important caption note: At lab-glucose temporal resolution (multi-hour gaps),
    RAPID_FALL reflects hyperglycaemic correction rather than counter-regulatory
    hypoglycaemia. Interpret accordingly in Methods.

    Panel A: Heart Rate (bpm) by trajectory
    Panel B: QTcF (Fridericia) by trajectory
    Panel C: Glucose level by trajectory (validates class separation)
    """
    print("  Generating Figure 5 — ECG Features × Glucose Trajectory...")

    # Filter to classified trajectories only (exclude UNKNOWN_*)
    ret = _retained_clean(df)
    traj_df = ret[ret["glucose_trajectory"].isin(TRAJ_ORDER)].copy()
    n_traj  = traj_df["glucose_trajectory"].value_counts()

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.patch.set_facecolor("white")

    # FIX: Pre-clip QTcF at 700ms — values above this are physiologically
    # impossible (patient would be in cardiac arrest) and are ECG cart artefacts
    # that survived the SQI filter. Without clipping, extreme outliers compress
    # the IQR boxes into an unreadable line at the bottom of the panel.
    traj_df = traj_df.copy()
    traj_df["qtc_fridericia_ms_plot"] = traj_df["qtc_fridericia_ms"].clip(upper=700)

    panels = [
        ("heart_rate_bpm",          "A  Heart Rate by Glucose Trajectory",
         "Heart Rate (bpm)",         None),
        ("qtc_fridericia_ms_plot",   "B  QTcF (Fridericia) by Glucose Trajectory\n"
                                     "(values >700 ms clipped — physiologically impossible)",
         "QTcF (ms)",                460),
        ("glucose_mg_dl",            "C  Glucose Level by Trajectory",
         "Blood Glucose (mg/dL)",    None),
    ]

    for ax, (col, title, ylabel, threshold) in zip(axes, panels):
        data_by_traj = [
            traj_df[traj_df["glucose_trajectory"] == t][col].dropna().values
            for t in TRAJ_ORDER
        ]
        counts = [n_traj.get(t, 0) for t in TRAJ_ORDER]

        bp = ax.boxplot(
            data_by_traj,
            patch_artist=True,
            notch=False,
            showfliers=True,
            flierprops=dict(marker="o", markersize=2.5, alpha=0.3, linestyle="none"),
            medianprops=dict(color="black", linewidth=2.5),
            whiskerprops=dict(linewidth=1.3),
            capprops=dict(linewidth=1.3),
            widths=0.55,
        )

        for patch, color in zip(bp["boxes"], TRAJ_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.80)

        # Mean markers
        means = [np.mean(d) if len(d) > 0 else np.nan for d in data_by_traj]
        ax.scatter(range(1, len(TRAJ_ORDER) + 1), means,
                   color="black", s=50, zorder=5, marker="D",
                   label="Mean", clip_on=False)

        if threshold is not None:
            ax.axhline(threshold, color=WONG["vermilion"], lw=1.5, ls="--", alpha=0.8)
            ax.text(5.55, threshold + 2, f"{threshold}", fontsize=8,
                    color=WONG["vermilion"], va="bottom")

        # Glucose: add clinical thresholds
        if col == "glucose_mg_dl":
            for thresh, c in [(70, WONG["orange"]), (180, WONG["sky"])]:
                ax.axhline(thresh, color=c, lw=1.2, ls=":", alpha=0.7)

        xlabels = [
            f"{TRAJ_LABELS[i]}\n(n={counts[i]:,})"
            for i in range(len(TRAJ_ORDER))
        ]
        ax.set_xticks(range(1, len(TRAJ_ORDER) + 1))
        ax.set_xticklabels(xlabels, fontsize=8)

        for tick, color in zip(ax.get_xticklabels(), TRAJ_COLORS):
            tick.set_color(color)
            tick.set_fontweight("bold")

        ax.set_ylabel(ylabel)
        ax.set_title(title)

    # Note about interpretation at lab-glucose resolution
    fig.text(0.5, -0.04,
             "Note: At retrospective lab-glucose temporal resolution (multi-hour inter-measurement gaps), "
             "RAPID_FALL reflects hyperglycaemic\ncorrection (insulin effect) rather than acute "
             "hypoglycaemic counter-regulation. Rate thresholds: ±60 mg/dL/hr = ±1 mg/dL/min (CGM rate-of-change convention). "
             "Retained (non-EXCLUDE) records with valid ECG values.",
             ha="center", fontsize=7.5, color="gray", style="italic",
             wrap=True)

    plt.tight_layout(w_pad=2.5)
    out_path = out_dir / "figure5_trajectory_ecg.png"
    _save(out_path, bbox_inches="tight")
    plt.close()
    print(f"    Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate figures for the MIMIC-IV Glucose–ECG Dataset paper"
    )
    parser.add_argument(
        "--csv", type=str, default=CSV_PATH,
        help="Path to mimiciv_ecg_glucose_aligned.csv (default: $GLUCOECG_CSV)"
    )
    parser.add_argument(
        "--out", type=str, default=OUT_DIR,
        help="Output directory for figures (created if missing)"
    )
    parser.add_argument("--formats", nargs="+", default=FORMATS)
    args = parser.parse_args()
    FORMATS[:] = args.formats

    # ── Resolve paths ─────────────────────────────────────────────────────────
    csv_path = Path(args.csv)
    out_dir  = Path(args.out)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            f"Set GLUCOECG_CSV or pass --csv /your/path.csv"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {out_dir.resolve()}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    df = load_data(str(csv_path))

    # ── Generate all figures ──────────────────────────────────────────────────
    print("\nGenerating figures...")
    figure1_prisma(df, out_dir)
    figure2_temporal(df, out_dir)
    figure3_phenotyping(df, out_dir)
    figure4_qtc_divergence(df, out_dir)
    figure5_trajectory(df, out_dir)

    print(f"\n{'─'*55}")
    print(f"All 5 figures saved to: {out_dir.resolve()}")
    print(f"{'─'*55}")
    for name in ["figure1_prisma_flowchart", "figure2_temporal_integrity", "figure3_glucose_phenotyping",
                 "figure4_qtc_divergence", "figure5_trajectory_ecg"]:
        print(f"  {name}.{{{','.join(FORMATS)}}}")
    print(f"{'─'*55}\n")


if __name__ == "__main__":
    main()