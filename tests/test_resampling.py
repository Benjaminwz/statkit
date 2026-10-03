import numpy as np
import pytest

import statkit as sk
import statkit.bootstrap as bootstrap_module


@pytest.fixture(scope="module")
def sample():
    rng = np.random.default_rng(1)
    return rng.normal(10, 2, 200), rng.normal(11, 2.5, 150)


def test_defaults_reproduce_statkit_0_1_results(sample):
    """Same seed, same numbers as the first release (reproducibility is the point)."""
    x, y = sample
    assert sk.bootstrap_ci(x, seed=7, n_boot=2000) == pytest.approx(
        (9.85267506721072, 9.592569955669047, 10.114857469275119), rel=1e-12)
    assert sk.bootstrap_ci(x, statistic=np.median, seed=3, n_boot=1500, alpha=0.1) == \
        pytest.approx((10.005466189302467, 9.77869704210123, 10.204497115291002), rel=1e-12)
    assert sk.permutation_test(x, y, n_perm=3000, seed=5) == pytest.approx(
        (-0.7662991619723751, 0.0006664445184938354), rel=1e-12)


def test_blocked_resampling_gives_identical_numbers(sample, monkeypatch):
    x, _ = sample
    whole = sk.bootstrap_ci(x, seed=7, n_boot=2000)
    monkeypatch.setattr(bootstrap_module, "_BLOCK", 5000)           # 25 rows per block
    assert sk.bootstrap_ci(x, seed=7, n_boot=2000) == whole


@pytest.mark.parametrize("method", ["percentile", "basic", "bca"])
def test_bootstrap_methods_cover_the_mean(sample, method):
    x, _ = sample
    est, lo, hi = sk.bootstrap_ci(x, n_boot=3000, seed=1, method=method)
    assert lo < est < hi and lo < 10 < hi
    assert sk.bootstrap_ci(x, n_boot=3000, seed=1, method=method) == (est, lo, hi)


def test_bca_and_basic_agree_with_scipy():
    st = pytest.importorskip("scipy.stats")
    x = np.random.default_rng(5).exponential(2, 40)
    _, lo, hi = sk.bootstrap_ci(x, n_boot=40_000, seed=1, method="bca")
    ref = st.bootstrap((x,), np.mean, n_resamples=40_000, method="BCa", random_state=2)
    width = ref.confidence_interval.high - ref.confidence_interval.low
    assert lo == pytest.approx(ref.confidence_interval.low, abs=0.03 * width)
    assert hi == pytest.approx(ref.confidence_interval.high, abs=0.03 * width)
    _, lo, hi = sk.bootstrap_ci(x, np.median, n_boot=20_000, seed=1, method="basic")
    ref = st.bootstrap((x,), np.median, n_resamples=20_000, method="basic", random_state=2)
    assert (lo, hi) == pytest.approx((ref.confidence_interval.low, ref.confidence_interval.high),
                                     rel=0.05)


def test_bootstrap_degenerate_bca_falls_back_with_warning():
    x = np.arange(1.0, 9.0)
    only_the_original = lambda s: float(np.array_equal(s, x))   # noqa: E731  (bias z0 = infinity)
    with pytest.warns(RuntimeWarning):
        est, lo, hi = sk.bootstrap_ci(x, only_the_original, n_boot=200, seed=0, method="bca")
    assert est == 1.0 and lo <= hi


def test_bootstrap_validation():
    with pytest.raises(ValueError):
        sk.bootstrap_ci([1, 2, 3], method="magic")
    with pytest.raises(ValueError):
        sk.bootstrap_ci([1, 2, 3], n_boot=10)


def test_bootstrap_diff_ci(sample):
    x, y = sample
    est, lo, hi = sk.bootstrap_diff_ci(x, y, n_boot=2000, seed=2)
    assert est == pytest.approx(x.mean() - y.mean()) and lo < est < hi and hi < 0
    est2, lo2, hi2 = sk.bootstrap_diff_ci(x[:100], x[:100] + 1.0, n_boot=500, seed=2,
                                          paired=True)
    assert (est2, lo2, hi2) == pytest.approx((-1.0, -1.0, -1.0), abs=1e-9)   # constant paired shift
    with pytest.raises(ValueError):
        sk.bootstrap_diff_ci(x, y, paired=True)


def test_bootstrap_diff_bca_agrees_with_scipy():
    st = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(8)
    a, b = rng.exponential(2, 30), rng.exponential(3, 35)
    pa = rng.normal(0, 1, 25)
    pb = pa + rng.normal(0.3, 0.7, 25)

    def mean_diff(u, v):
        return np.mean(u) - np.mean(v)

    for (u, v), paired in (((a, b), False), ((pa, pb), True)):
        _, lo, hi = sk.bootstrap_diff_ci(u, v, n_boot=30_000, seed=3, paired=paired, method="bca")
        ref = st.bootstrap((u, v), mean_diff, n_resamples=30_000, method="BCa", paired=paired,
                           random_state=4).confidence_interval
        width = ref.high - ref.low
        assert lo == pytest.approx(ref.low, abs=0.04 * width)
        assert hi == pytest.approx(ref.high, abs=0.04 * width)


def test_permutation_alternatives_and_symmetry(sample):
    x, y = sample
    _, p_two = sk.permutation_test(x, y, n_perm=2000, seed=1)
    _, p_less = sk.permutation_test(x, y, n_perm=2000, seed=1, alternative="less")
    _, p_greater = sk.permutation_test(x, y, n_perm=2000, seed=1, alternative="greater")
    assert p_less < 0.01 and p_greater > 0.99 and p_two < 0.01
    with pytest.raises(ValueError):
        sk.permutation_test(x, y, alternative="sideways")


def test_permutation_exact_matches_scipy():
    st = pytest.importorskip("scipy.stats")
    a = np.array([12.1, 14.3, 9.8, 15.2, 11.0, 13.3])
    b = np.array([10.2, 9.1, 12.0, 8.7, 10.9])
    for alt in ("two-sided", "greater", "less"):
        _, p = sk.permutation_test(a, b, exact=True, alternative=alt)
        ref = st.permutation_test((a, b), lambda u, v: u.mean() - v.mean(), n_resamples=10 ** 6,
                                  permutation_type="independent", alternative=alt)
        assert p == pytest.approx(ref.pvalue, abs=1e-12)
    pa = np.array([5.1, 4.8, 6.0, 5.5, 4.9, 6.2, 5.7])
    pb = np.array([4.7, 4.9, 5.1, 5.0, 4.6, 5.8, 5.2])
    for alt in ("two-sided", "greater", "less"):
        _, p = sk.permutation_test(pa, pb, exact=True, paired=True, alternative=alt)
        ref = st.permutation_test((pa, pb), lambda u, v, axis: np.mean(u - v, axis=axis),
                                  n_resamples=10 ** 6, permutation_type="samples",
                                  alternative=alt)
        assert p == pytest.approx(ref.pvalue, abs=1e-12)


def test_permutation_paired_monte_carlo_and_custom_statistic():
    rng = np.random.default_rng(3)
    before = rng.normal(0, 1, 30)
    after = before + rng.normal(0.8, 0.5, 30)
    _, p = sk.permutation_test(after, before, n_perm=2000, seed=1, paired=True)
    assert p < 0.01
    obs, p = sk.permutation_test(after, before, n_perm=500, seed=1,
                                 statistic=lambda u, v: np.median(u) - np.median(v))
    assert obs == pytest.approx(np.median(after) - np.median(before)) and 0 < p <= 1


def test_permutation_exact_guards_and_validation():
    with pytest.raises(ValueError):
        sk.permutation_test(np.arange(40.0), np.arange(40.0) + 1, exact=True)
    with pytest.raises(ValueError):
        sk.permutation_test([1, 2, 3], [1, 2], paired=True)
    with pytest.raises(ValueError):
        sk.permutation_test([1, 2, 3], [4, 5, 6], n_perm=10)
