"""Every hypothesis test is compared with SciPy on randomised data (tolerance 1e-8)."""
import numpy as np
import pytest

import statkit as sk

st = pytest.importorskip("scipy.stats")

ALTS = ["two-sided", "less", "greater"]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(2024)
    return {
        "a": rng.normal(0, 1, 23), "b": rng.normal(0.4, 1.7, 31), "c": rng.normal(1, 3, 17),
        "paired": (rng.normal(0, 1, 20), rng.normal(0.3, 1, 20)),
    }


def close(x, y):
    assert x == pytest.approx(y, rel=1e-8, abs=1e-12)


@pytest.mark.parametrize("alt", ALTS)
@pytest.mark.parametrize("equal_var", [False, True])
def test_ttest_ind(data, alt, equal_var):
    r = sk.ttest_ind(data["a"], data["b"], equal_var=equal_var, alternative=alt)
    s = st.ttest_ind(data["a"], data["b"], equal_var=equal_var, alternative=alt)
    close(r.statistic, s.statistic)
    close(r.p_value, s.pvalue)
    if hasattr(s, "confidence_interval"):                      # newer SciPy only
        close(r.df, s.df)
        ci = s.confidence_interval(0.95)
        if np.isfinite(r.ci[0]):
            close(r.ci[0], ci.low)
        if np.isfinite(r.ci[1]):
            close(r.ci[1], ci.high)


@pytest.mark.parametrize("alt", ALTS)
def test_ttest_rel_and_1samp(data, alt):
    x, y = data["paired"]
    r, s = sk.ttest_rel(x, y, alternative=alt), st.ttest_rel(x, y, alternative=alt)
    close(r.statistic, s.statistic)
    close(r.p_value, s.pvalue)
    r, s = sk.ttest_1samp(x, 0.3, alternative=alt), st.ttest_1samp(x, 0.3, alternative=alt)
    close(r.statistic, s.statistic)
    close(r.p_value, s.pvalue)


def test_anova_and_levene(data):
    a, b, c = data["a"], data["b"], data["c"]
    r, s = sk.anova_oneway(a, b, c), st.f_oneway(a, b, c)
    close(r.statistic, s.statistic)
    close(r.p_value, s.pvalue)
    assert 0 < r.effect_size < 1 and r.details["omega_squared"] <= r.effect_size
    for center in ("median", "mean"):
        r, s = sk.levene(a, b, c, center=center), st.levene(a, b, c, center=center)
        close(r.statistic, s.statistic)
        close(r.p_value, s.pvalue)


def test_welch_anova_reduces_to_welch_t_for_two_groups(data):
    r = sk.anova_oneway(data["a"], data["b"], equal_var=False)
    t = sk.ttest_ind(data["a"], data["b"], equal_var=False)
    close(r.statistic, t.statistic ** 2)
    close(r.p_value, t.p_value)
    close(r.df[1], t.df)


def test_welch_anova_known_value():
    # reference: statsmodels.stats.oneway.anova_oneway(use_var="unequal") on these data
    g = ([24, 27, 21, 25, 30, 22], [31, 35, 29, 40, 33, 38, 36], [20, 26, 19, 23])
    r = sk.anova_oneway(*g, equal_var=False)
    assert r.statistic == pytest.approx(18.19525301197009, rel=1e-9)
    assert r.df == pytest.approx((2.0, 8.29609338661709), rel=1e-9)
    assert r.p_value == pytest.approx(0.0009258022051885412, rel=1e-8)
    classic = sk.anova_oneway(*g)
    assert classic.statistic == pytest.approx(20.28936747253805, rel=1e-9)


@pytest.mark.parametrize("alt", ALTS)
@pytest.mark.parametrize("continuity", [True, False])
def test_mannwhitney_asymptotic_with_and_without_ties(data, alt, continuity):
    for a, b in ((data["a"], data["b"]), (np.round(data["a"] * 2), np.round(data["b"] * 2))):
        r = sk.mannwhitneyu(a, b, alternative=alt, method="asymptotic", continuity=continuity)
        s = st.mannwhitneyu(a, b, alternative=alt, method="asymptotic", use_continuity=continuity)
        close(r.statistic, s.statistic)
        close(r.p_value, s.pvalue)


@pytest.mark.parametrize("alt", ALTS)
def test_mannwhitney_exact(data, alt):
    a, b = data["a"][:9], data["b"][:12]
    r = sk.mannwhitneyu(a, b, alternative=alt, method="exact")
    close(r.p_value, st.mannwhitneyu(a, b, alternative=alt, method="exact").pvalue)
    assert sk.mannwhitneyu(a, b, alternative=alt).details["method"] == "exact"  # auto


def test_mannwhitney_exact_rejects_ties():
    with pytest.raises(ValueError):
        sk.mannwhitneyu([1, 2, 2], [2, 3, 4], method="exact")


@pytest.mark.parametrize("alt", ALTS)
def test_wilcoxon(data, alt):
    x, y = data["paired"]
    r = sk.wilcoxon(x, y, alternative=alt, method="exact")
    s = st.wilcoxon(x, y, alternative=alt, method="exact")
    close(r.p_value, s.pvalue)
    close(r.statistic, s.statistic)
    for cont in (True, False):
        r = sk.wilcoxon(x, y, alternative=alt, method="asymptotic", continuity=cont)
        s = st.wilcoxon(x, y, alternative=alt, method="approx", correction=cont)
        close(r.p_value, s.pvalue)


def test_wilcoxon_with_ties_and_zeros(data):
    x, y = (np.round(v * 3) for v in data["paired"])
    r = sk.wilcoxon(x, y, continuity=False)
    s = st.wilcoxon(x, y, method="approx", correction=False, zero_method="wilcox")
    assert r.details["n_zero"] > 0 and r.details["method"] == "asymptotic"
    close(r.p_value, s.pvalue)
    with pytest.raises(ValueError):
        sk.wilcoxon([1, 2, 3], [1, 2, 3])


def test_kruskal_with_and_without_ties(data):
    a, b, c = data["a"], data["b"], data["c"]
    for groups in ((a, b, c), (np.round(a), np.round(b), np.round(c))):
        r, s = sk.kruskal(*groups), st.kruskal(*groups)
        close(r.statistic, s.statistic)
        close(r.p_value, s.pvalue)


@pytest.mark.parametrize("n", [3, 4, 5, 6, 11, 12, 30, 200])
def test_shapiro_wilk(n):
    rng = np.random.default_rng(n)
    for x in (rng.normal(0, 1, n), rng.exponential(1, n)):
        r, s = sk.shapiro(x), st.shapiro(x)
        # older SciPy (e.g. 1.10) evaluates the algorithm in single precision: looser bound
        assert r.statistic == pytest.approx(s.statistic, rel=1e-5)
        assert r.p_value == pytest.approx(s.pvalue, rel=1e-4, abs=1e-12)


def test_shapiro_detects_non_normal_data():
    x = np.random.default_rng(0).exponential(1, 80)
    assert sk.shapiro(x).p_value < 0.001
    with pytest.raises(ValueError):
        sk.shapiro([2.0, 2.0, 2.0, 2.0])


@pytest.mark.parametrize("alt", ALTS)
def test_correlations(alt):
    rng = np.random.default_rng(5)
    u = rng.normal(0, 1, 45)
    w = 0.5 * u + rng.normal(0, 1, 45)
    r, s = sk.pearsonr(u, w, alternative=alt), st.pearsonr(u, w, alternative=alt)
    close(r.statistic, s.statistic)
    close(r.p_value, s.pvalue)
    if alt == "two-sided":
        ci = s.confidence_interval(0.95)
        close(r.ci[0], ci.low)
        close(r.ci[1], ci.high)
    r, s = sk.spearmanr(u, w, alternative=alt), st.spearmanr(u, w, alternative=alt)
    close(r.statistic, s.statistic)
    close(r.p_value, s.pvalue)
    ui, wi = np.round(u * 2), np.round(w * 2)               # ties
    r = sk.kendalltau(ui, wi, alternative=alt)
    s = st.kendalltau(ui, wi, alternative=alt, method="asymptotic")
    close(r.statistic, s.statistic)
    close(r.p_value, s.pvalue)


def test_correlation_edge_cases():
    assert sk.pearsonr([1, 2, 3, 4], [2, 4, 6, 8]).p_value == 0.0
    assert sk.pearsonr([1, 2, 3], [3, 1, 2]).ci is None      # CI needs n >= 4
    with pytest.raises(ValueError):
        sk.pearsonr([1, 1, 1, 1], [1, 2, 3, 4])
    with pytest.raises(ValueError):
        sk.spearmanr([1, 2, 3], [1, 2])


@pytest.mark.parametrize("correction", [True, False])
def test_chi2_contingency(correction):
    rng = np.random.default_rng(3)
    for shape in ((2, 2), (2, 3), (3, 4)):
        t = rng.integers(2, 30, size=shape)
        r = sk.chi2_contingency(t, correction=correction)
        s = st.chi2_contingency(t, correction=correction)
        close(r.statistic, s.statistic)
        close(r.p_value, s.pvalue)
    assert r.effect_size_name == "Cramér's V" and 0 <= r.effect_size <= 1


@pytest.mark.parametrize("alt", ALTS)
def test_fisher_exact(alt):
    rng = np.random.default_rng(11)
    for _ in range(6):
        t = rng.integers(1, 15, size=(2, 2))
        r, s = sk.fisher_exact(t, alternative=alt), st.fisher_exact(t, alternative=alt)
        close(r.p_value, s.pvalue)
        close(r.statistic, s.statistic)


def test_fisher_exact_tea_tasting():
    # Fisher's lady tasting tea: p = 17/70 two-sided = 0.4857..., one-sided = 0.2429
    r = sk.fisher_exact([[3, 1], [1, 3]])
    assert r.p_value == pytest.approx(34 / 70)
    one_sided = sk.fisher_exact([[3, 1], [1, 3]], alternative="greater")
    assert one_sided.p_value == pytest.approx(17 / 70)
    assert r.statistic == 9.0


@pytest.mark.parametrize("k", [0, 3, 17, 40])
def test_proportion_ci(k):
    for method, scipy_method in (("wilson", "wilson"), ("clopper-pearson", "exact")):
        est, lo, hi = sk.proportion_ci(k, 40, method=method)
        ref = st.binomtest(k, 40).proportion_ci(0.95, method=scipy_method)
        assert est == k / 40
        close(lo, ref.low)
        close(hi, ref.high)
    est, lo, hi = sk.proportion_ci(k, 40, method="jeffreys")
    assert 0 <= lo <= est <= hi <= 1
    with pytest.raises(ValueError):
        sk.proportion_ci(41, 40)
    with pytest.raises(ValueError):
        sk.proportion_ci(3, 40, method="nope")


def test_describe(data):
    a = data["a"]
    d = sk.describe(a)
    close(d.skewness, st.skew(a, bias=False))
    close(d.kurtosis, st.kurtosis(a, bias=False))
    close(d.sem, st.sem(a))
    ci = st.t.interval(0.95, a.size - 1, loc=a.mean(), scale=st.sem(a))
    close(d.ci_mean[0], ci[0])
    close(d.ci_mean[1], ci[1])
    assert d.n == a.size and d.q1 <= d.median <= d.q3
    assert sk.describe([1, 2]).skewness != sk.describe([1, 2]).skewness      # NaN for tiny n
