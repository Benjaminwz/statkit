"""Rank-based tests: Mann-Whitney U, Wilcoxon signed-rank and Kruskal-Wallis."""
from __future__ import annotations

import math

import numpy as np

from ._dist import chi2_sf, norm_cdf, norm_sf
from ._result import TestResult
from ._util import as_groups, as_sample, check_alternative, rankdata, tie_term


def _mw_exact_pmf(m, n):
    """Exact null distribution of the Mann-Whitney U statistic (no ties)."""
    total = m + n
    maxsum = m * total - m * (m - 1) // 2
    dp = np.zeros((m + 1, maxsum + 1))
    dp[0, 0] = 1.0
    for i in range(1, total + 1):                    # count m-subsets of ranks by their sum
        for k in range(min(i, m), 0, -1):
            dp[k, i:] += dp[k - 1, : maxsum + 1 - i]
    lo = m * (m + 1) // 2
    counts = dp[m, lo: lo + m * n + 1]
    return counts / counts.sum()


def _signed_rank_exact_pmf(n):
    """Exact null distribution of the Wilcoxon positive-rank sum T+ (no ties, no zeros)."""
    size = n * (n + 1) // 2
    c = np.zeros(size + 1)
    c[0] = 1.0
    for i in range(1, n + 1):
        c[i:] = c[i:] + c[: size + 1 - i]
    return c / c.sum()


def _tails(pmf, k):
    """``(P(S <= k), P(S >= k))`` for an integer-valued statistic."""
    k = int(round(k))
    return float(min(1.0, pmf[: k + 1].sum())), float(min(1.0, pmf[k:].sum()))


def mannwhitneyu(a, b, alternative="two-sided", method="auto", continuity=True):
    """Mann-Whitney U (Wilcoxon rank-sum) test.

    ``method='exact'`` uses the exact null distribution (requires no ties); ``'asymptotic'``
    uses the tie-corrected normal approximation (with continuity correction by default);
    ``'auto'`` picks exact when there are no ties and both samples have at most 50 values.
    ``statistic`` is U for ``a``; the effect size is the rank-biserial correlation
    (Cliff's delta). ``alternative='greater'`` means ``a`` tends to be larger than ``b``.
    """
    alternative = check_alternative(alternative)
    a, b = as_sample(a, "a", 1), as_sample(b, "b", 1)
    n1, n2 = a.size, b.size
    ranks, counts = rankdata(np.concatenate([a, b]))
    u1 = float(ranks[:n1].sum() - n1 * (n1 + 1) / 2.0)
    u2 = n1 * n2 - u1
    has_ties = bool(counts.max() > 1)
    if method not in ("auto", "exact", "asymptotic"):
        raise ValueError("method must be 'auto', 'exact' or 'asymptotic'")
    if method == "auto":
        method = "exact" if not has_ties and max(n1, n2) <= 50 else "asymptotic"
    if method == "exact" and has_ties:
        raise ValueError("the exact distribution requires data without ties")
    details = {"method": method, "u1": u1, "u2": u2,
               "prob_superiority": u1 / (n1 * n2), "n_a": n1, "n_b": n2}
    if method == "exact":
        le, ge = _tails(_mw_exact_pmf(n1, n2), u1)
        p = {"greater": ge, "less": le, "two-sided": min(1.0, 2 * min(le, ge))}[alternative]
    else:
        total = n1 + n2
        var = n1 * n2 / 12.0 * ((total + 1) - tie_term(counts) / (total * (total - 1)))
        if var <= 0:
            raise ValueError("all observations are identical; the test is undefined")
        s, mu, cc = math.sqrt(var), n1 * n2 / 2.0, 0.5 if continuity else 0.0
        if alternative == "greater":
            z = (u1 - mu - cc) / s
            p = norm_sf(z)
        elif alternative == "less":
            z = (u1 - mu + cc) / s
            p = norm_cdf(z)
        else:
            z = (max(u1, u2) - mu - cc) / s
            p = min(1.0, 2 * norm_sf(z))
        details["z"] = z
    return TestResult(
        test="Mann-Whitney U test", statistic=u1, p_value=float(p), stat_label="U",
        effect_size=2 * u1 / (n1 * n2) - 1, effect_size_name="rank-biserial correlation",
        n=n1 + n2, alternative=alternative, details=details,
    )


def wilcoxon(x, y=None, alternative="two-sided", method="auto", continuity=True):
    """Wilcoxon signed-rank test of ``x - y`` (or of ``x`` alone) against a zero median.

    Zero differences are dropped (the classical Wilcoxon treatment). ``method='exact'``
    needs no ties and no zero differences; ``'auto'`` uses it for at most 50 pairs.
    ``statistic`` follows SciPy: the smaller rank sum (two-sided) or the positive-rank
    sum T+ (one-sided). The effect size is the matched-pairs rank-biserial correlation.
    """
    alternative = check_alternative(alternative)
    x = as_sample(x, "x", 1)
    d = x
    if y is not None:
        y = as_sample(y, "y", 1)
        if y.size != x.size:
            raise ValueError("paired samples must have the same length")
        d = x - y
    n_zero = int(np.sum(d == 0))
    d_nz = d[d != 0]
    n = d_nz.size
    if n == 0:
        raise ValueError("all differences are zero; the test is undefined")
    ranks, counts = rankdata(np.abs(d_nz))
    t_plus = float(ranks[d_nz > 0].sum())
    t_minus = float(ranks[d_nz < 0].sum())
    has_ties = bool(counts.max() > 1)
    if method not in ("auto", "exact", "asymptotic"):
        raise ValueError("method must be 'auto', 'exact' or 'asymptotic'")
    if method == "auto":
        method = "exact" if n <= 50 and not has_ties and n_zero == 0 else "asymptotic"
    if method == "exact" and (has_ties or n_zero):
        raise ValueError("the exact distribution requires no ties and no zero differences")
    details = {"method": method, "t_plus": t_plus, "t_minus": t_minus, "n_zero": n_zero,
               "median_difference": float(np.median(d))}
    if method == "exact":
        le, ge = _tails(_signed_rank_exact_pmf(n), t_plus)
        p = {"greater": ge, "less": le, "two-sided": min(1.0, 2 * min(le, ge))}[alternative]
    else:
        mn = n * (n + 1) / 4.0
        var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term(counts) / 48.0
        if var <= 0:
            raise ValueError("degenerate differences; the test is undefined")
        se, cc = math.sqrt(var), 0.5 if continuity else 0.0
        if alternative == "greater":
            z = (t_plus - mn - cc) / se
            p = norm_sf(z)
        elif alternative == "less":
            z = (t_plus - mn + cc) / se
            p = norm_cdf(z)
        else:
            z = (abs(t_plus - mn) - cc) / se
            p = min(1.0, 2 * norm_sf(z))
        details["z"] = z
    stat = min(t_plus, t_minus) if alternative == "two-sided" else t_plus
    return TestResult(
        test="Wilcoxon signed-rank test", statistic=stat, p_value=float(p), stat_label="W",
        effect_size=(t_plus - t_minus) / (t_plus + t_minus),
        effect_size_name="rank-biserial correlation", n=n, alternative=alternative,
        details=details,
    )


def kruskal(*groups):
    """Kruskal-Wallis H test (tie-corrected, chi-square approximation).

    The effect size is epsilon squared, ``H / (N - 1)``.
    """
    gs = as_groups(groups, min_size=1)
    sizes = np.array([g.size for g in gs], dtype=float)
    total = int(sizes.sum())
    ranks, counts = rankdata(np.concatenate(gs))
    h, start = 0.0, 0
    for g in gs:
        h += ranks[start: start + g.size].sum() ** 2 / g.size
        start += g.size
    h = 12.0 / (total * (total + 1)) * h - 3.0 * (total + 1)
    correction = 1.0 - tie_term(counts) / (total ** 3 - total)
    if correction <= 0:
        raise ValueError("all observations are identical; the test is undefined")
    h /= correction
    df = len(gs) - 1
    return TestResult(
        test="Kruskal-Wallis H test", statistic=float(h), p_value=float(min(1.0, chi2_sf(h, df))),
        stat_label="H", df=float(df), effect_size=float(h / (total - 1)),
        effect_size_name="epsilon squared", n=total,
        details={"group_sizes": sizes.astype(int)},
    )
