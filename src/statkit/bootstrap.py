"""Bootstrap confidence intervals: percentile, basic and bias-corrected accelerated (BCa)."""
from __future__ import annotations

import warnings

import numpy as np

from ._dist import norm_cdf, norm_ppf
from ._util import as_sample, check_alpha

_METHODS = ("percentile", "basic", "bca")
_BLOCK = 4_000_000          # cap on the size of one index matrix, keeps memory bounded


def _check(alpha, n_boot, method):
    alpha = check_alpha(alpha)
    if n_boot < 100:
        raise ValueError("n_boot must be at least 100")
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}")
    return alpha


def _bca_bounds(boot, theta, jack, alpha):
    """Quantile levels for the BCa interval, or ``None`` if it is degenerate."""
    prop = float(np.mean(boot < theta) + 0.5 * np.mean(boot == theta))
    if not 0.0 < prop < 1.0:
        return None
    z0 = norm_ppf(prop)
    dev = jack.mean() - jack
    denom = 6.0 * float(np.sum(dev ** 2)) ** 1.5
    accel = float(np.sum(dev ** 3)) / denom if denom > 0 else 0.0
    levels = []
    for z in (norm_ppf(alpha / 2), norm_ppf(1 - alpha / 2)):
        shift = z0 + z
        d = 1.0 - accel * shift
        if d <= 0:
            return None
        levels.append(norm_cdf(z0 + shift / d))
    return levels


def _interval(boot, theta, jack_fn, alpha, method):
    if method == "bca":
        levels = _bca_bounds(boot, theta, jack_fn(), alpha)
        if levels is None:
            warnings.warn("BCa interval is degenerate here; falling back to the percentile "
                          "interval", RuntimeWarning, stacklevel=3)
        else:
            lower, upper = np.quantile(boot, levels)
            return float(lower), float(upper)
    lower, upper = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    if method == "basic":
        return float(2 * theta - upper), float(2 * theta - lower)
    return float(lower), float(upper)


def bootstrap_ci(data, statistic=np.mean, n_boot=10_000, alpha=0.05, seed=None,
                 method="percentile"):
    """Bootstrap CI for ``statistic`` of one sample.

    ``method`` is ``'percentile'`` (default), ``'basic'`` or ``'bca'`` (bias-corrected and
    accelerated; second-order accurate, recommended for skewed statistics, costs ``n`` extra
    evaluations of ``statistic``). Returns ``(estimate, lower, upper)``. Pass ``seed`` for
    reproducible results; the default percentile method gives the same numbers as statkit 0.1.
    """
    alpha = _check(alpha, n_boot, method)
    x = as_sample(data, "data")
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    block = max(1, _BLOCK // x.size)
    for start in range(0, n_boot, block):
        idx = rng.integers(0, x.size, size=(min(block, n_boot - start), x.size))
        boot[start:start + idx.shape[0]] = [statistic(x[row]) for row in idx]
    theta = float(statistic(x))
    lo, hi = _interval(
        boot, theta,
        lambda: np.array([statistic(np.delete(x, i)) for i in range(x.size)]),
        alpha, method)
    return theta, lo, hi


def bootstrap_diff_ci(a, b, statistic=np.mean, n_boot=10_000, alpha=0.05, seed=None,
                      paired=False, method="percentile"):
    """Bootstrap CI for ``statistic(a) - statistic(b)``: ``(estimate, lower, upper)``.

    Independent samples are resampled separately; ``paired=True`` resamples pairs jointly
    (``a`` and ``b`` must then have equal length).
    """
    alpha = _check(alpha, n_boot, method)
    a, b = as_sample(a, "a"), as_sample(b, "b")
    if paired and a.size != b.size:
        raise ValueError("paired samples must have the same length")
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    block = max(1, _BLOCK // max(a.size, b.size))
    for start in range(0, n_boot, block):
        m = min(block, n_boot - start)
        ia = rng.integers(0, a.size, size=(m, a.size))
        ib = ia if paired else rng.integers(0, b.size, size=(m, b.size))
        boot[start:start + m] = [statistic(a[r]) - statistic(b[s]) for r, s in zip(ia, ib)]
    theta = float(statistic(a) - statistic(b))

    def jackknife():
        if paired:
            return np.array([statistic(np.delete(a, i)) - statistic(np.delete(b, i))
                             for i in range(a.size)])
        sb, sa = statistic(b), statistic(a)
        return np.array([statistic(np.delete(a, i)) - sb for i in range(a.size)]
                        + [sa - statistic(np.delete(b, i)) for i in range(b.size)])

    lo, hi = _interval(boot, theta, jackknife, alpha, method)
    return theta, lo, hi
