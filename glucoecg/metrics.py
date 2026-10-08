"""
Evaluation metrics: regression metrics, Clarke and Parkes error grids, and
patient-level bootstrap confidence intervals.

Clarke error grid
    Zones of Clarke et al. (Diabetes Care 1987;10:622-628), Figure 1 and Methods,
    evaluated with precedence A > E > C > D > B and the paper's strict thresholds
    (< 70, > 180, > 240). Unlike the v1.0 code, a true hypoglycaemia (< 70 mg/dL)
    predicted at 70-180 mg/dL is Zone D (failure to detect), not B or C.

Parkes (consensus) error grid, type 1 diabetes
    Zone polygons of Parkes et al. (Diabetes Care 2000;23:1143-1148) as encoded in
    the CRAN package 'ega' (getParkesZones, type = 1).
"""

import numpy as np
from matplotlib.path import Path as _MplPath

ZONES = ("A", "B", "C", "D", "E")


# ── Regression metrics ────────────────────────────────────────────────────────

def regression_metrics(y_true, y_pred) -> dict:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    ok = ~(np.isnan(yt) | np.isnan(yp))
    yt, yp = yt[ok], yp[ok]
    if len(yt) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "mard": np.nan, "r2": np.nan, "bias": np.nan}
    e = yp - yt
    ss_tot = ((yt - yt.mean()) ** 2).sum()
    return {
        "n": int(len(yt)),
        "mae": float(np.abs(e).mean()),
        "rmse": float(np.sqrt((e ** 2).mean())),
        "mard": float((np.abs(e) / np.maximum(yt, 1)).mean() * 100),
        "r2": float(1 - (e ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan,
        "bias": float(e.mean()),
    }


# ── Clarke error grid ─────────────────────────────────────────────────────────

def clarke_zones(ref, pred) -> np.ndarray:
    """
    Clarke error grid zone per pair (reference, prediction), mg/dL.

    Follows the definitions and Figure 1 of Clarke et al. (1987), whose assumptions are
    a target range of 70-180 mg/dL, hypoglycaemia < 70, hyperglycaemia > 180, and
    "failure to treat blood glucose values < 70 or > 240":

      A  prediction within 20% of the reference, or both reference and prediction < 70
      E  reference < 70 and prediction > 180, or reference > 180 and prediction < 70
      C  over-correction: prediction >= reference + 110 for reference 70-290 (upper C), or
         prediction <= 1.4 x reference - 182 for reference 130-180 (lower C)
      D  failure to detect: reference < 70 or > 240 with prediction in the 70-180 target range
      B  everything else

    Zones are assigned with precedence A > E > C > D > B, so the part of the reference
    58.3-70 band that lies within 20% stays A.
    """
    ref = np.asarray(ref, dtype=float)
    pred = np.asarray(pred, dtype=float)
    in_target = (pred >= 70) & (pred <= 180)
    a = (np.abs(pred - ref) <= 0.2 * ref) | ((ref < 70) & (pred < 70))
    e = ((ref < 70) & (pred > 180)) | ((ref > 180) & (pred < 70))
    c = (((ref >= 70) & (ref <= 290) & (pred >= ref + 110))
         | ((ref >= 130) & (ref <= 180) & (pred <= 7 / 5 * ref - 182)))
    d = ((ref < 70) | (ref > 240)) & in_target
    return np.select([a, e, c, d], ["A", "E", "C", "D"], default="B")


# ── Parkes (consensus) error grid, type 1 ─────────────────────────────────────

def _parkes_type1_polygons(max_x: float, max_y: float) -> list:
    slope = lambda x, y, xe, ye: (ye - y) / (xe - x)          # noqa: E731
    end_x = lambda sx, sy, my, k: (my - sy) / k + sx           # noqa: E731
    end_y = lambda sx, sy, mx, k: (mx - sx) * k + sy           # noqa: E731
    ce, cdu, cdl = slope(35, 155, 50, 550), slope(80, 215, 125, 550), slope(250, 40, 550, 150)
    ccu, ccl = slope(70, 110, 260, 550), slope(260, 130, 550, 250)
    cbu, cbl = slope(280, 380, 430, 550), slope(385, 300, 550, 450)
    poly = lambda xs, ys: _MplPath(np.column_stack([xs, ys]))  # noqa: E731
    # Order matters: later zones overwrite earlier ones (B, then C, then D, then E)
    return [
        ("B", poly([50, 50, 170, 385, max_x, max_x, 50], [0, 30, 145, 300, end_y(385, 300, max_x, cbl), 0, 0])),
        ("B", poly([0, 30, 140, 280, end_x(280, 380, max_y, cbu), 0, 0], [50, 50, 170, 380, max_y, max_y, 50])),
        ("C", poly([120, 120, 260, max_x, max_x, 120], [0, 30, 130, end_y(260, 130, max_x, ccl), 0, 0])),
        ("C", poly([0, 30, 50, 70, end_x(70, 110, max_y, ccu), 0, 0], [60, 60, 80, 110, max_y, max_y, 60])),
        ("D", poly([250, 250, max_x, max_x, 250], [0, 40, end_y(410, 110, max_x, cdl), 0, 0])),
        ("D", poly([0, 25, 50, 80, end_x(80, 215, max_y, cdu), 0, 0], [100, 100, 125, 215, max_y, max_y, 100])),
        ("E", poly([0, 35, end_x(35, 155, max_y, ce), 0, 0], [150, 155, max_y, max_y, 150])),
    ]


def parkes_zones(ref, pred) -> np.ndarray:
    ref = np.asarray(ref, dtype=float)
    pred = np.asarray(pred, dtype=float)
    max_x = max(np.nanmax(ref) + 20, 550)
    max_y = max(np.nanmax(pred) + 20, max_x, 550)
    pts = np.column_stack([ref, pred])
    zones = np.full(len(ref), "A", dtype="<U1")
    for z, p in _parkes_type1_polygons(max_x, max_y):
        zones[p.contains_points(pts)] = z
    return zones


def zone_summary(zones) -> dict:
    zones = np.asarray(zones)
    n = len(zones)
    out = {z: float((zones == z).mean() * 100) if n else np.nan for z in ZONES}
    out["A+B"] = out["A"] + out["B"]
    out["counts"] = {z: int((zones == z).sum()) for z in ZONES}
    return out


# ── Patient-level bootstrap ───────────────────────────────────────────────────

def _cluster_sums(groups, values):
    _, inv = np.unique(groups, return_inverse=True)
    return inv.max() + 1, np.bincount(inv, values), np.bincount(inv)


def bootstrap_ci(y_true, y_pred, groups, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """95% CI of MAE by resampling patients (clusters) with replacement."""
    err = np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float))
    k, s, c = _cluster_sums(np.asarray(groups), err)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, k, k), minlength=k)
        stats[b] = (w * s).sum() / (w * c).sum()
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def paired_bootstrap_ci(y_true, pred_a, pred_b, groups, n_boot: int = 1000, seed: int = 0,
                        alpha: float = 0.05):
    """Point estimate and 95% CI of MAE(a) - MAE(b) on the same rows, resampling patients."""
    yt = np.asarray(y_true, float)
    ea = np.abs(np.asarray(pred_a, float) - yt)
    eb = np.abs(np.asarray(pred_b, float) - yt)
    k, sa, c = _cluster_sums(np.asarray(groups), ea)
    _, sb, _ = _cluster_sums(np.asarray(groups), eb)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, k, k), minlength=k)
        stats[b] = ((w * sa).sum() - (w * sb).sum()) / (w * c).sum()
    point = float(ea.mean() - eb.mean())
    return point, float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


GLYCAEMIC_BANDS = {
    "hypo (<70)": lambda y: y < 70,
    "eu (70-180)": lambda y: (y >= 70) & (y <= 180),
    "hyper (>180)": lambda y: y > 180,
}


def subgroup_metrics(y_true, y_pred, labels) -> dict:
    """Regression metrics per value of ``labels`` (e.g. alignment tier, sex)."""
    y_true, y_pred, labels = map(np.asarray, (y_true, y_pred, labels))
    return {str(v): regression_metrics(y_true[labels == v], y_pred[labels == v])
            for v in pd_unique(labels)}


def pd_unique(a):
    seen, out = set(), []
    for v in a:
        if v not in seen and not (isinstance(v, float) and np.isnan(v)):
            seen.add(v)
            out.append(v)
    return out
