"""Power and sample-size calculations for t-tests (exact, via the non-central t)."""
from __future__ import annotations

import math

from ._dist import bisect, nct_cdf, norm_isf, t_isf
from ._util import check_alpha, check_alternative

_KINDS = ("two-sample", "one-sample", "paired")


def _setup(kind, n, ratio):
    """Return (non-centrality per unit effect size, degrees of freedom)."""
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}")
    if kind == "two-sample":
        n2 = n * ratio
        return math.sqrt(n * n2 / (n + n2)), n + n2 - 2
    return math.sqrt(n), n - 1


def power_ttest(effect_size, n, alpha=0.05, kind="two-sample", alternative="two-sided",
                ratio=1.0):
    """Statistical power of a t-test.

    ``effect_size`` is Cohen's d (``dz`` for paired designs) in the direction of the
    alternative hypothesis, so pass it positive also for ``alternative='less'``. ``n`` is the
    size of the first group (or the number of pairs / observations) and ``ratio = n2 / n1``.
    Matches G*Power's "t tests - Means" results.
    """
    alpha, alternative = check_alpha(alpha), check_alternative(alternative)
    if n < 2:
        raise ValueError("n must be at least 2")
    scale, df = _setup(kind, n, ratio)
    ncp = abs(effect_size) * scale
    if alternative == "two-sided":
        crit = t_isf(alpha / 2, df)
        return float(1.0 - nct_cdf(crit, df, ncp) + nct_cdf(-crit, df, ncp))
    return float(1.0 - nct_cdf(t_isf(alpha, df), df, ncp))


def sample_size_ttest(effect_size, power=0.8, alpha=0.05, kind="two-sample",
                      alternative="two-sided", ratio=1.0):
    """Smallest ``n`` (per first group, or pairs) reaching the requested power.

    Returns an integer ``n``; for two-sample designs the second group needs
    ``ceil(ratio * n)`` observations.
    """
    alpha, alternative = check_alpha(alpha), check_alternative(alternative)
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")
    if effect_size == 0:
        raise ValueError("effect_size must be non-zero")
    d = abs(effect_size)
    za = norm_isf(alpha / 2 if alternative == "two-sided" else alpha)
    guess = (za + norm_isf(1.0 - power)) ** 2 / d ** 2
    if kind == "two-sample":
        guess *= (1.0 + 1.0 / ratio)
    n = max(2, int(math.ceil(guess)))

    def reaches(m):
        return power_ttest(d, m, alpha, kind, alternative, ratio) >= power

    while not reaches(n):
        n += 1
    while n > 2 and reaches(n - 1):
        n -= 1
    return n


def min_detectable_effect(n, power=0.8, alpha=0.05, kind="two-sample", alternative="two-sided",
                          ratio=1.0):
    """Smallest standardised effect detectable with the given ``n`` and power."""
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")
    return float(bisect(
        lambda d: power_ttest(d, n, alpha, kind, alternative, ratio) - power, 1e-6, 50.0))
