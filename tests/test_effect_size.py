import math

import numpy as np
import pytest

import statkit as sk
from statkit.effect_size import hedges_j


def test_hedges_correction_is_exact_gamma_ratio():
    assert hedges_j(2) == pytest.approx(math.gamma(1) / (math.sqrt(1) * math.gamma(0.5)))
    assert hedges_j(10) == pytest.approx(math.gamma(5) / (math.sqrt(5) * math.gamma(4.5)))
    assert 0.922 < hedges_j(10) < 0.923                               # Hedges & Olkin (1985) table
    assert hedges_j(10_000) == pytest.approx(1 - 3 / (4 * 10_000 - 1), abs=1e-9)
    assert math.isnan(hedges_j(1))


def test_hedges_g_validates_input():
    with pytest.raises(ValueError):
        sk.hedges_g([[1, 2], [3, 4]], [1, 2, 3])
    with pytest.raises(ValueError):
        sk.hedges_g([1, float("nan")], [1, 2, 3])


def test_glass_delta():
    assert sk.glass_delta([5, 7, 9], [1, 2, 3]) == pytest.approx((7 - 2) / 1.0)
    with pytest.raises(ValueError):
        sk.glass_delta([1, 2], [3, 3])


def test_dz_paired_and_one_sample():
    x, y = [5.0, 7.0, 9.0, 8.0], [4.0, 5.0, 8.0, 5.0]
    diff = np.array(x) - np.array(y)
    assert sk.cohens_dz(x, y) == pytest.approx(diff.mean() / diff.std(ddof=1))
    assert sk.cohens_dz(x, mu=6.0) == pytest.approx((np.mean(x) - 6) / np.std(x, ddof=1))
    assert abs(sk.cohens_dz(x, y, bias_correct=True)) < abs(sk.cohens_dz(x, y))
    with pytest.raises(ValueError):
        sk.cohens_dz([1, 2, 3], [1, 2])


def test_exact_ci_matches_scipy_pivot():
    st = pytest.importorskip("scipy.stats")
    optimize = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(7)
    a, b = rng.normal(0, 1, 18), rng.normal(0.8, 1.3, 25)
    d, lo, hi = sk.cohens_d_ci(a, b)
    scale, df = math.sqrt(1 / 18 + 1 / 25), 18 + 25 - 2
    t = d / scale
    ref_lo = optimize.brentq(lambda x: st.nct.cdf(t, df, x) - 0.975, t - 8, t + 8)
    ref_hi = optimize.brentq(lambda x: st.nct.cdf(t, df, x) - 0.025, t - 8, t + 8)
    assert (lo, hi) == pytest.approx((ref_lo * scale, ref_hi * scale), rel=1e-8)
    g, glo, ghi = sk.hedges_g_ci(a, b)
    j = hedges_j(df)
    assert (g, glo, ghi) == pytest.approx((d * j, lo * j, hi * j))


def test_ci_contains_estimate_and_widens_with_confidence():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 30), rng.normal(0.5, 1, 30)
    d, lo95, hi95 = sk.cohens_d_ci(a, b, 0.05)
    _, lo99, hi99 = sk.cohens_d_ci(a, b, 0.01)
    assert lo95 < d < hi95 and lo99 < lo95 and hi99 > hi95
    dz, lo, hi = sk.cohens_dz_ci(a, b)
    assert lo < dz < hi


def test_cliffs_delta_and_probability_of_superiority():
    a, b = [1, 2, 3, 4], [3, 4, 5, 6]
    # 16 pairs: a > b only (4, 3); ties (3, 3) and (4, 4); a < b in the other 13
    assert sk.prob_superiority(a, b) == pytest.approx((1 + 0.5 * 2) / 16)
    assert sk.cliffs_delta(a, b) == pytest.approx((1 - 13) / 16)
    assert sk.cliffs_delta(b, a) == pytest.approx((13 - 1) / 16)
    assert sk.cliffs_delta([1, 2], [1, 2]) == 0.0
    assert sk.cliffs_delta([5, 6, 7], [1, 2, 3]) == 1.0
