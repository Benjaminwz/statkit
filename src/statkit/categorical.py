"""Tests and intervals for counts and proportions."""
from __future__ import annotations

import math

import numpy as np

from ._dist import beta_ppf, chi2_sf, norm_isf
from ._result import TestResult
from ._util import check_alpha, check_alternative


def _table(table, name="table"):
    t = np.asarray(table, dtype=float)
    if t.ndim != 2 or min(t.shape) < 2:
        raise ValueError(f"{name} must be a 2-D contingency table of at least 2 x 2")
    if np.any(t < 0) or not np.all(np.isfinite(t)):
        raise ValueError(f"{name} must contain non-negative finite counts")
    return t


def chi2_contingency(table, correction=True):
    """Pearson chi-square test of independence for an r x c table of counts.

    Yates' continuity correction is applied to 2 x 2 tables when ``correction=True``.
    The effect size is Cramer's V (computed from the uncorrected statistic). Check
    ``details['min_expected']``: the approximation is poor when expected counts are below 5
    (use :func:`fisher_exact` for small 2 x 2 tables).
    """
    obs = _table(table)
    total = obs.sum()
    rows, cols = obs.sum(axis=1), obs.sum(axis=0)
    expected = np.outer(rows, cols) / total
    if np.any(expected == 0):
        raise ValueError("a row or column of the table is all zeros")
    dof = (obs.shape[0] - 1) * (obs.shape[1] - 1)
    diff = obs - expected
    stat_uncorrected = float(np.sum(diff ** 2 / expected))
    applied = bool(correction and dof == 1)
    if applied:
        diff = np.sign(diff) * np.maximum(np.abs(diff) - 0.5, 0.0)
    stat = float(np.sum(diff ** 2 / expected))
    v = math.sqrt(stat_uncorrected / (total * (min(obs.shape) - 1)))
    return TestResult(
        test="Pearson chi-square test of independence", statistic=stat,
        p_value=float(min(1.0, chi2_sf(stat, dof))), stat_label="χ²", df=float(dof),
        effect_size=v, effect_size_name="Cramér's V", n=int(round(total)),
        details={"yates_correction": applied, "expected": expected,
                 "min_expected": float(expected.min()),
                 "share_expected_below_5": float(np.mean(expected < 5))},
    )


def fisher_exact(table, alternative="two-sided", alpha=0.05):
    """Fisher's exact test for a 2 x 2 table ``[[a, b], [c, d]]``.

    The two-sided p-value sums every table at most as probable as the observed one (as in R
    and SciPy). ``statistic``/``estimate`` is the sample odds ratio ``ad / bc`` with a Woolf
    (log-scale Wald) confidence interval when all four cells are non-zero.
    ``alternative='greater'`` means the odds ratio is greater than 1.
    """
    alternative, alpha = check_alternative(alternative), check_alpha(alpha)
    t = _table(table)
    if t.shape != (2, 2) or np.any(t != np.round(t)):
        raise ValueError("table must be a 2 x 2 array of integer counts")
    a, b, c, d = (int(v) for v in t.ravel())
    r1, c1, total = a + b, a + c, a + b + c + d
    ks = range(max(0, r1 + c1 - total), min(r1, c1) + 1)
    denom = math.comb(total, r1)
    pmf = {k: math.comb(c1, k) * math.comb(total - c1, r1 - k) / denom for k in ks}
    p_obs = pmf[a]
    if alternative == "greater":
        p = sum(v for k, v in pmf.items() if k >= a)
    elif alternative == "less":
        p = sum(v for k, v in pmf.items() if k <= a)
    else:
        p = sum(v for v in pmf.values() if v <= p_obs * (1 + 1e-7))
    if b * c > 0:
        odds = a * d / (b * c)
    else:
        odds = math.inf if a * d > 0 else math.nan
    ci = None
    if min(a, b, c, d) > 0:
        se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        z = norm_isf(alpha / 2)
        ci = (math.exp(math.log(odds) - z * se), math.exp(math.log(odds) + z * se))
    return TestResult(
        test="Fisher's exact test", statistic=odds, p_value=float(min(1.0, p)), stat_label="OR",
        estimate=odds, estimate_name="odds ratio", ci=ci, conf_level=1 - alpha, n=total,
        alternative=alternative, details={"ci_method": "Woolf (Wald on log odds ratio)"},
    )


def proportion_ci(successes, n, alpha=0.05, method="wilson"):
    """Confidence interval for a binomial proportion: ``(estimate, lower, upper)``.

    Methods: ``'wilson'`` (default, recommended), ``'clopper-pearson'`` (exact, conservative),
    ``'jeffreys'``, ``'agresti-coull'`` and ``'wald'`` (poor coverage, for comparison only).
    """
    alpha = check_alpha(alpha)
    k = int(successes)
    if int(n) != n or n < 1 or k != successes or not 0 <= k <= n:
        raise ValueError("need integers with 0 <= successes <= n and n >= 1")
    n = int(n)
    p = k / n
    z = norm_isf(alpha / 2)
    if method == "wilson":
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        lo, hi = centre - half, centre + half
    elif method == "agresti-coull":
        n_t = n + z * z
        p_t = (k + z * z / 2) / n_t
        half = z * math.sqrt(p_t * (1 - p_t) / n_t)
        lo, hi = p_t - half, p_t + half
    elif method == "wald":
        half = z * math.sqrt(p * (1 - p) / n)
        lo, hi = p - half, p + half
    elif method == "clopper-pearson":
        lo = beta_ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
        hi = beta_ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    elif method == "jeffreys":
        lo = beta_ppf(alpha / 2, k + 0.5, n - k + 0.5) if k > 0 else 0.0
        hi = beta_ppf(1 - alpha / 2, k + 0.5, n - k + 0.5) if k < n else 1.0
    else:
        raise ValueError("method must be wilson, agresti-coull, wald, clopper-pearson or jeffreys")
    return p, max(0.0, lo), min(1.0, hi)
