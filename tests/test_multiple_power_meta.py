import math

import numpy as np
import pytest

import statkit as sk

P = [0.001, 0.008, 0.02, 0.03, 0.04, 0.2, 0.5, 0.74, 0.01]
# reference values from statsmodels.stats.multitest.multipletests
REFERENCE = {
    "sidak": [0.008964, 0.069738, 0.166252, 0.239769, 0.307466, 0.865782, 0.998047, 0.999995,
              0.086483],
    "holm-sidak": [0.008964, 0.062236, 0.114158, 0.141266, 0.150653, 0.488, 0.75, 0.75, 0.067935],
    "hochberg": [0.009, 0.064, 0.12, 0.15, 0.16, 0.6, 0.74, 0.74, 0.07],
    "fdr_by": [0.025461, 0.084869, 0.127304, 0.152764, 0.169738, 0.727449, 1.0, 1.0, 0.084869],
    "holm": [0.009, 0.064, 0.12, 0.15, 0.16, 0.6, 1.0, 1.0, 0.07],
    "fdr_bh": [0.009, 0.03, 0.045, 0.054, 0.06, 0.257143, 0.5625, 0.74, 0.03],
}


@pytest.mark.parametrize("method", sorted(REFERENCE))
def test_adjustments_match_statsmodels(method):
    assert sk.adjust_pvalues(P, method) == pytest.approx(REFERENCE[method], abs=1e-6)


def test_adjustment_ordering_and_aliases():
    adj = {m: sk.adjust_pvalues(P, m) for m in ("bonferroni", "holm", "holm-sidak", "hochberg")}
    assert np.all(adj["holm"] <= adj["bonferroni"] + 1e-12)
    assert np.all(adj["hochberg"] <= adj["holm"] + 1e-12)
    assert np.all(adj["holm-sidak"] <= adj["holm"] + 1e-12)
    assert sk.adjust_pvalues(P, "benjamini-hochberg") == pytest.approx(sk.benjamini_hochberg(P))
    with pytest.raises(ValueError):
        sk.adjust_pvalues(P, "bogus")


@pytest.mark.parametrize("fn", [sk.sidak, sk.holm_sidak, sk.hochberg, sk.benjamini_yekutieli])
def test_new_corrections_validate_and_stay_in_bounds(fn):
    out = fn([0.5, 0.9, 0.99])
    assert np.all(out >= 0) and np.all(out <= 1)
    with pytest.raises(ValueError):
        fn([0.5, -0.1])
    assert fn([0.0, 0.0]).tolist() == [0.0, 0.0]


# ------------------------------------------------------------------------------ power
@pytest.mark.parametrize("d,kind,n", [(0.5, "two-sample", 64), (0.8, "two-sample", 26),
                                      (0.2, "two-sample", 394), (0.5, "one-sample", 34),
                                      (0.5, "paired", 34)])
def test_sample_size_matches_gpower(d, kind, n):
    assert sk.sample_size_ttest(d, kind=kind) == n
    assert sk.power_ttest(d, n, kind=kind) >= 0.8 > sk.power_ttest(d, n - 1, kind=kind)


def test_power_matches_scipy_noncentral_t():
    st = pytest.importorskip("scipy.stats")
    for d, n, alt in [(0.35, 40, "two-sided"), (0.6, 22, "greater"), (1.1, 8, "two-sided")]:
        df, ncp = 2 * n - 2, d * math.sqrt(n / 2)
        if alt == "two-sided":
            crit = st.t.isf(0.025, df)
            ref = st.nct.sf(crit, df, ncp) + st.nct.cdf(-crit, df, ncp)
        else:
            ref = st.nct.sf(st.t.isf(0.05, df), df, ncp)
        assert sk.power_ttest(d, n, alternative=alt) == pytest.approx(ref, rel=1e-8)


def test_power_properties_and_unequal_groups():
    assert sk.power_ttest(0.5, 100) > sk.power_ttest(0.5, 50) > sk.power_ttest(0.5, 20)
    assert sk.power_ttest(0.5, 50, alternative="greater") > sk.power_ttest(0.5, 50)
    assert sk.power_ttest(0.0001, 30) == pytest.approx(0.05, abs=1e-4)      # size of the test
    assert sk.sample_size_ttest(0.5, ratio=2.0) < sk.sample_size_ttest(0.5)  # n1 shrinks, n2 = 2 n1
    assert sk.power_ttest(-0.5, 64) == sk.power_ttest(0.5, 64)


def test_min_detectable_effect_round_trips():
    d = sk.min_detectable_effect(30)
    assert d == pytest.approx(0.73562, abs=1e-4)
    assert sk.power_ttest(d, 30) == pytest.approx(0.8, abs=1e-8)


def test_power_validation():
    with pytest.raises(ValueError):
        sk.power_ttest(0.5, 1)
    with pytest.raises(ValueError):
        sk.power_ttest(0.5, 30, kind="three-sample")
    with pytest.raises(ValueError):
        sk.sample_size_ttest(0.0)
    with pytest.raises(ValueError):
        sk.sample_size_ttest(0.5, power=1.2)


# ----------------------------------------------------------------------- meta-analysis
Y = [0.30, 0.15, 0.62, -0.05, 0.41, 0.22]
V = [0.04, 0.02, 0.09, 0.03, 0.05, 0.015]


def test_meta_analysis_matches_statsmodels():
    fe = sk.meta_analysis(Y, V, model="fixed")
    assert fe.estimate == pytest.approx(0.20905660377358487, rel=1e-12)
    assert fe.q == pytest.approx(5.310094339622641, rel=1e-12)
    assert fe.q_df == 5 and fe.i2 == pytest.approx(5.8397143, rel=1e-6)
    dl = sk.meta_analysis(Y, V)
    assert (dl.estimate, dl.se, dl.tau2) == pytest.approx((0.210884, 0.072540, 0.001921), abs=1e-6)
    pm = sk.meta_analysis(Y, V, tau2_method="PM")
    assert (pm.estimate, pm.se, pm.tau2) == pytest.approx((0.211550, 0.073569, 0.002637), abs=1e-6)


def test_meta_analysis_structure():
    r = sk.meta_analysis(Y, ses=np.sqrt(V))
    lo, hi = r.ci
    assert lo < r.estimate < hi and r.weights.sum() == pytest.approx(1.0)
    assert r.prediction_interval[0] < lo and r.prediction_interval[1] > hi
    assert "Random-effects" in str(r) and "I²" in str(r)
    assert isinstance(r.to_dict()["weights"], list)
    assert sk.meta_analysis(Y[:2], V[:2]).prediction_interval is None      # needs k >= 3


def test_meta_analysis_homogeneous_studies_have_zero_tau2():
    r = sk.meta_analysis([0.2, 0.2, 0.2], [0.01, 0.02, 0.03])
    assert r.tau2 == 0.0 and r.i2 == 0.0
    assert sk.meta_analysis([0.2, 0.2, 0.2], [0.01, 0.02, 0.03], tau2_method="PM").tau2 == 0.0


def test_meta_analysis_validation():
    with pytest.raises(ValueError):
        sk.meta_analysis(Y, V, ses=V)
    with pytest.raises(ValueError):
        sk.meta_analysis(Y)
    with pytest.raises(ValueError):
        sk.meta_analysis(Y, V[:-1])
    with pytest.raises(ValueError):
        sk.meta_analysis([0.1], [0.01])
    with pytest.raises(ValueError):
        sk.meta_analysis(Y, [0.0] + V[1:])
    with pytest.raises(ValueError):
        sk.meta_analysis(Y, V, model="bayes")


# ------------------------------------------------------------------------ reliability
def test_cohens_kappa_textbook_table():
    r1 = [0] * 25 + [1] * 25
    r2 = [0] * 20 + [1] * 5 + [0] * 10 + [1] * 15       # [[20, 5], [10, 15]] -> po .7, pe .5
    k, lo, hi = sk.cohens_kappa(r1, r2)
    assert k == pytest.approx(0.4) and lo < k < hi
    assert sk.cohens_kappa(r1, r2, weights="linear")[0] == pytest.approx(0.4)
    assert sk.cohens_kappa(r1, r1)[0] == pytest.approx(1.0)


def test_cohens_kappa_matches_statsmodels():
    inter = pytest.importorskip("statsmodels.stats.inter_rater")
    rng = np.random.default_rng(4)
    a = rng.integers(0, 3, 80)
    b = np.where(rng.random(80) < 0.6, a, rng.integers(0, 3, 80))
    table = np.zeros((3, 3))
    np.add.at(table, (a, b), 1)
    for weights in (None, "linear", "quadratic"):
        ref = inter.cohens_kappa(table, wt=weights)
        assert sk.cohens_kappa(a, b, weights=weights) == pytest.approx(
            (ref.kappa, ref.kappa_low, ref.kappa_upp), rel=1e-8)


def test_cohens_kappa_works_with_labels_and_validates():
    k, _, _ = sk.cohens_kappa(["yes", "no", "yes", "no"], ["yes", "no", "no", "no"])
    assert 0 < k < 1
    with pytest.raises(ValueError):
        sk.cohens_kappa([1, 1, 1], [1, 1, 1])
    with pytest.raises(ValueError):
        sk.cohens_kappa([1, 2, 3], [1, 2])
    with pytest.raises(ValueError):
        sk.cohens_kappa([1, 2], [1, 2], weights="cubic")


def test_cronbach_alpha():
    rng = np.random.default_rng(2)
    items = rng.normal(0, 1, (60, 1)) + rng.normal(0, 1, (60, 5))
    a, lo, hi = sk.cronbach_alpha(items)
    cov = np.cov(items, rowvar=False)
    assert a == pytest.approx(5 / 4 * (1 - np.trace(cov) / cov.sum()))
    assert lo < a < hi <= 1
    same = np.tile(rng.normal(0, 1, (30, 1)), (1, 4))
    assert sk.cronbach_alpha(same)[0] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        sk.cronbach_alpha([1, 2, 3])
    with pytest.raises(ValueError):
        sk.cronbach_alpha([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
