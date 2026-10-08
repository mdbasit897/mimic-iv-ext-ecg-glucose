import numpy as np
import pytest

from glucoecg.metrics import (bootstrap_ci, clarke_zones, paired_bootstrap_ci, parkes_zones,
                              regression_metrics, zone_summary)


@pytest.mark.parametrize("ref,pred,zone", [
    (100, 110, "A"),   # within 20 %
    (60, 65, "A"),     # both below 70
    (50, 100, "D"),    # hypoglycaemia missed (v1.0 code returned B)
    (65, 110, "D"),    # hypoglycaemia missed (v1.0 code returned B)
    (60, 150, "D"),    # hypoglycaemia missed (v1.0 code returned C)
    (300, 150, "D"),   # severe hyperglycaemia missed
    (250, 100, "D"),   # v1.0 code returned C
    (100, 250, "C"),   # over-correction, pred >= ref + 110
    (170, 50, "C"),    # lower Zone C: pred <= 1.4 * ref - 182
    (200, 60, "E"),    # hyper read as hypo
    (60, 200, "E"),    # hypo read as hyper
    (150, 100, "B"),   # benign
    (69, 50, "A"),     # both below 70 (Clarke 1987: "<70 mg/dl ... when the reference is also <70")
    (69, 150, "D"),    # hypoglycaemia read as in-target
    (250, 150, "D"),   # "failure to treat ... >240" read as in-target
    (230, 150, "B"),   # reference 180-240 read as in-target: benign (Figure 1, lower B)
    (200, 65, "E"),    # hyperglycaemia read as hypoglycaemia
])
def test_clarke_reference_points(ref, pred, zone):
    assert clarke_zones([ref], [pred])[0] == zone


def test_clarke_identity_is_zone_a():
    ref = np.linspace(30, 700, 500)
    assert (clarke_zones(ref, ref) == "A").all()


def test_parkes_reference_points():
    ref = np.array([100, 100, 100, 300, 50, 20])
    pred = np.array([100, 140, 200, 120, 160, 300])
    z = parkes_zones(ref, pred)
    assert z[0] == "A"
    assert z[1] == "B"
    assert z[2] in {"C", "D"}
    assert z[3] in {"C", "D"}
    assert z[5] == "E"


def test_parkes_points_named_in_parkes_2000():
    # Parkes et al. 2000, Methods: "(90, 240) was assigned risk category C by most
    # respondents, whereas (300, 310) was always assigned A"
    assert list(parkes_zones([90, 300], [240, 310])) == ["C", "A"]


def test_parkes_type1_de_boundary_starts_at_150():
    # Parkes et al. 2000: type 1 D/E boundary begins at a measured BG of 150 mg/dL
    assert parkes_zones([10], [145])[0] == "D"
    assert parkes_zones([10], [160])[0] == "E"


def test_parkes_type1_cd_boundary_at_low_true_bg():
    # Parkes et al. 2000: at true BG < 30 mg/dL the type 1 C/D boundary is horizontal and
    # 20 mg/dL higher than type 2 (100 vs 80 mg/dL measured)
    assert parkes_zones([10], [95])[0] == "C"
    assert parkes_zones([10], [105])[0] == "D"


def test_parkes_identity_is_zone_a():
    ref = np.linspace(30, 700, 500)
    assert (parkes_zones(ref, ref) == "A").all()


def test_zone_summary_sums_to_100():
    s = zone_summary(np.array(list("AABBCDE")))
    assert abs(sum(s[z] for z in "ABCDE") - 100) < 1e-9
    assert s["counts"]["A"] == 2


def test_regression_metrics():
    m = regression_metrics([100, 200], [110, 190])
    assert m["mae"] == 10 and m["bias"] == 0 and m["n"] == 2


def test_bootstrap_contains_point_and_resamples_patients():
    rng = np.random.default_rng(0)
    y = rng.normal(140, 40, 2000)
    p = y + rng.normal(0, 20, 2000)
    g = np.repeat(np.arange(400), 5)
    lo, hi = bootstrap_ci(y, p, g, n_boot=300)
    mae = np.abs(p - y).mean()
    assert lo < mae < hi
    point, lo, hi = paired_bootstrap_ci(y, p, p, g, n_boot=100)
    assert point == 0 and lo == 0 and hi == 0
