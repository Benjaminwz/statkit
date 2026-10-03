"""Correlation coefficients with p-values and confidence intervals."""
from __future__ import annotations

import math

import numpy as np

from ._dist import norm_cdf, norm_isf, norm_sf, t_cdf, t_sf, t_sf2
from ._result import TestResult
from ._util import as_sample, check_alpha, check_alternative, rankdata


def _paired(x, y, min_size):
    x, y = as_sample(x, "x", min_size), as_sample(y, "y", min_size)
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    return x, y


def _pearson(x, y):
    xd, yd = x - x.mean(), y - y.mean()
    denom = math.sqrt(float(np.sum(xd ** 2) * np.sum(yd ** 2)))
    if denom == 0:
        raise ValueError("an input is constant; the correlation is undefined")
    return max(-1.0, min(1.0, float(np.sum(xd * yd)) / denom))


def _fisher_ci(r, n, alpha, alternative, var_factor=1.0):
    """Fisher z interval; ``var_factor`` is 1 for Pearson, ``1 + r^2/2`` for Spearman."""
    if n < 4:
        return None
    if abs(r) >= 1:
        return (r, r)
    se = math.sqrt(var_factor / (n - 3))
    z = math.atanh(r)
    if alternative == "two-sided":
        crit = norm_isf(alpha / 2)
        return math.tanh(z - crit * se), math.tanh(z + crit * se)
    crit = norm_isf(alpha)
    if alternative == "greater":
        return math.tanh(z - crit * se), 1.0
    return -1.0, math.tanh(z + crit * se)


def _t_p(r, df, alternative):
    if abs(r) >= 1:
        return 0.0 if alternative == "two-sided" or (r > 0) == (alternative == "greater") else 1.0
    t = r * math.sqrt(df / (1 - r * r))
    if alternative == "two-sided":
        return t_sf2(t, df)
    return t_sf(t, df) if alternative == "greater" else t_cdf(t, df)


def pearsonr(x, y, alternative="two-sided", alpha=0.05):
    """Pearson product-moment correlation; CI by Fisher's z transformation (needs n >= 4)."""
    alternative, alpha = check_alternative(alternative), check_alpha(alpha)
    x, y = _paired(x, y, 3)
    r, n = _pearson(x, y), x.size
    return TestResult(
        test="Pearson correlation", statistic=r,
        p_value=float(min(1.0, _t_p(r, n - 2, alternative))),
        stat_label="r", df=float(n - 2), ci=_fisher_ci(r, n, alpha, alternative),
        conf_level=1 - alpha, n=n, alternative=alternative,
    )


def spearmanr(x, y, alternative="two-sided", alpha=0.05):
    """Spearman rank correlation. The p-value uses the t approximation; the CI uses Fisher's
    z with the Bonett-Wright (2000) variance ``(1 + r^2 / 2) / (n - 3)``."""
    alternative, alpha = check_alternative(alternative), check_alpha(alpha)
    x, y = _paired(x, y, 3)
    r, n = _pearson(rankdata(x)[0], rankdata(y)[0]), x.size
    return TestResult(
        test="Spearman rank correlation", statistic=r,
        p_value=float(min(1.0, _t_p(r, n - 2, alternative))), stat_label="rs",
        df=float(n - 2), ci=_fisher_ci(r, n, alpha, alternative, 1 + r * r / 2),
        conf_level=1 - alpha, n=n, alternative=alternative,
    )


def _kendall_s(x, y):
    """Concordant minus discordant pairs, in memory-bounded blocks."""
    n = x.size
    block = max(1, 4_000_000 // n)
    total = 0.0
    for i in range(0, n, block):
        sx = np.sign(x[i:i + block, None] - x[None, :])
        sy = np.sign(y[i:i + block, None] - y[None, :])
        total += float(np.sum(sx * sy))
    return total / 2.0


def kendalltau(x, y, alternative="two-sided"):
    """Kendall's tau-b with the tie-corrected normal approximation for the p-value.

    Cost is O(n^2) time (but bounded memory). Exact small-sample p-values are not provided.
    """
    alternative = check_alternative(alternative)
    x, y = _paired(x, y, 3)
    n = x.size
    s = _kendall_s(x, y)
    tx = rankdata(x)[1].astype(float)
    ty = rankdata(y)[1].astype(float)
    n0 = n * (n - 1) / 2.0
    n1, n2 = np.sum(tx * (tx - 1)) / 2.0, np.sum(ty * (ty - 1)) / 2.0
    if n0 == n1 or n0 == n2:
        raise ValueError("an input is constant; the correlation is undefined")
    tau = s / math.sqrt((n0 - n1) * (n0 - n2))
    v0 = n * (n - 1) * (2 * n + 5)
    vt = float(np.sum(tx * (tx - 1) * (2 * tx + 5)))
    vu = float(np.sum(ty * (ty - 1) * (2 * ty + 5)))
    v1 = float(np.sum(tx * (tx - 1)) * np.sum(ty * (ty - 1))) / (2.0 * n * (n - 1))
    v2 = float(np.sum(tx * (tx - 1) * (tx - 2)) * np.sum(ty * (ty - 1) * (ty - 2))) \
        / (9.0 * n * (n - 1) * (n - 2))
    var = (v0 - vt - vu) / 18.0 + v1 + v2
    z = s / math.sqrt(var)
    p = {"two-sided": 2 * norm_sf(abs(z)), "greater": norm_sf(z), "less": norm_cdf(z)}[alternative]
    return TestResult(
        test="Kendall rank correlation (tau-b)", statistic=float(tau), p_value=float(min(1.0, p)),
        stat_label="tau", n=n, alternative=alternative,
        details={"z": z, "concordant_minus_discordant": s},
    )
