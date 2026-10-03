"""Classical parametric tests: t-tests, one-way ANOVA and Levene's test."""
from __future__ import annotations

import math

import numpy as np

from ._dist import f_sf, t_cdf, t_isf, t_sf, t_sf2
from ._result import TestResult
from ._util import as_groups, as_sample, check_alpha, check_alternative
from .effect_size import cohens_dz_ci, hedges_g_ci


def _t_inference(est, se, df, alternative, alpha):
    """t statistic, p-value and (one- or two-sided) CI for an estimate with standard error."""
    t = est / se
    if alternative == "two-sided":
        p, crit = t_sf2(t, df), t_isf(alpha / 2, df)
        ci = (est - crit * se, est + crit * se)
    elif alternative == "greater":
        p, crit = t_sf(t, df), t_isf(alpha, df)
        ci = (est - crit * se, math.inf)
    else:
        p, crit = t_cdf(t, df), t_isf(alpha, df)
        ci = (-math.inf, est + crit * se)
    return t, min(1.0, p), ci


def ttest_ind(a, b, equal_var=False, alternative="two-sided", alpha=0.05):
    """Two-sample t-test.

    Welch's unequal-variance test is the default (recommended unless variances are known to be
    equal; note SciPy defaults to the pooled Student test). The effect size is Hedges' g with an
    exact non-central-t interval. ``alternative='greater'`` tests ``mean(a) > mean(b)``.
    """
    alternative, alpha = check_alternative(alternative), check_alpha(alpha)
    a, b = as_sample(a, "a"), as_sample(b, "b")
    na, nb = a.size, b.size
    va, vb = a.var(ddof=1), b.var(ddof=1)
    diff = float(a.mean() - b.mean())
    if equal_var:
        df = na + nb - 2
        se = math.sqrt(((na - 1) * va + (nb - 1) * vb) / df * (1 / na + 1 / nb))
    else:
        sa, sb = va / na, vb / nb
        se = math.sqrt(sa + sb)
        df = (sa + sb) ** 2 / (sa ** 2 / (na - 1) + sb ** 2 / (nb - 1)) if se > 0 else math.nan
    if se == 0:
        raise ValueError("both samples have zero variance; the t-test is undefined")
    t, p, ci = _t_inference(diff, se, df, alternative, alpha)
    g, glo, ghi = hedges_g_ci(a, b, alpha)
    return TestResult(
        test="Student two-sample t-test" if equal_var else "Welch two-sample t-test",
        statistic=float(t), p_value=float(p), stat_label="t", df=float(df),
        estimate=diff, estimate_name="mean difference", ci=ci, conf_level=1 - alpha,
        effect_size=g, effect_size_name="Hedges' g", effect_size_ci=(glo, ghi),
        n=na + nb, alternative=alternative,
        details={"mean_a": float(a.mean()), "mean_b": float(b.mean()),
                 "sd_a": float(math.sqrt(va)), "sd_b": float(math.sqrt(vb)),
                 "n_a": na, "n_b": nb, "se": se},
    )


def ttest_rel(a, b, alternative="two-sided", alpha=0.05):
    """Paired t-test on ``a - b``; effect size is Hedges' gz (bias-corrected dz) with exact CI."""
    alternative, alpha = check_alternative(alternative), check_alpha(alpha)
    a, b = as_sample(a, "a"), as_sample(b, "b")
    if a.size != b.size:
        raise ValueError("paired samples must have the same length")
    d = a - b
    sd = float(d.std(ddof=1))
    if sd == 0:
        raise ValueError("differences have zero variance; the paired t-test is undefined")
    n = d.size
    est = float(d.mean())
    se = sd / math.sqrt(n)
    t, p, ci = _t_inference(est, se, n - 1, alternative, alpha)
    g, glo, ghi = cohens_dz_ci(a, b, alpha=alpha, bias_correct=True)
    return TestResult(
        test="Paired t-test", statistic=float(t), p_value=float(p), stat_label="t",
        df=float(n - 1), estimate=est, estimate_name="mean difference", ci=ci,
        conf_level=1 - alpha, effect_size=g, effect_size_name="Hedges' gz",
        effect_size_ci=(glo, ghi), n=n, alternative=alternative,
        details={"sd_diff": sd, "se": se, "dz": est / sd},
    )


def ttest_1samp(x, popmean=0.0, alternative="two-sided", alpha=0.05):
    """One-sample t-test of ``mean(x) == popmean``; effect size is Hedges' g with exact CI."""
    alternative, alpha = check_alternative(alternative), check_alpha(alpha)
    x = as_sample(x, "x")
    sd = float(x.std(ddof=1))
    if sd == 0:
        raise ValueError("sample has zero variance; the t-test is undefined")
    n = x.size
    est = float(x.mean() - popmean)
    se = sd / math.sqrt(n)
    t, p, ci = _t_inference(est, se, n - 1, alternative, alpha)
    g, glo, ghi = cohens_dz_ci(x, None, mu=popmean, alpha=alpha, bias_correct=True)
    return TestResult(
        test="One-sample t-test", statistic=float(t), p_value=float(p), stat_label="t",
        df=float(n - 1), estimate=est, estimate_name="mean minus popmean", ci=ci,
        conf_level=1 - alpha, effect_size=g, effect_size_name="Hedges' g",
        effect_size_ci=(glo, ghi), n=n, alternative=alternative,
        details={"mean": float(x.mean()), "sd": sd, "popmean": float(popmean), "se": se},
    )


def anova_oneway(*groups, equal_var=True):
    """One-way ANOVA across independent groups.

    ``equal_var=False`` gives Welch's ANOVA (robust to unequal variances). The effect size is
    eta squared; omega squared (less biased) and the sums of squares are in ``details``
    (both are computed from the classical decomposition, also for Welch's test).
    """
    gs = as_groups(groups)
    k = len(gs)
    n = np.array([g.size for g in gs], dtype=float)
    total = n.sum()
    means = np.array([g.mean() for g in gs])
    variances = np.array([g.var(ddof=1) for g in gs])
    grand = float(np.sum(n * means) / total)
    ssb = float(np.sum(n * (means - grand) ** 2))
    ssw = float(np.sum((n - 1) * variances))
    msw = ssw / (total - k)
    if equal_var:
        if msw == 0:
            raise ValueError("within-group variance is zero; ANOVA is undefined")
        f_stat = (ssb / (k - 1)) / msw
        df = (k - 1.0, float(total - k))
    else:
        if np.any(variances == 0):
            raise ValueError("Welch's ANOVA needs non-zero variance in every group")
        w = n / variances
        wsum = w.sum()
        weighted_mean = float(np.sum(w * means) / wsum)
        lam = float(np.sum((1 - w / wsum) ** 2 / (n - 1)))
        num = float(np.sum(w * (means - weighted_mean) ** 2)) / (k - 1)
        f_stat = num / (1 + 2 * (k - 2) / (k ** 2 - 1) * lam)
        df = (k - 1.0, (k ** 2 - 1) / (3 * lam))
    p = min(1.0, f_sf(f_stat, *df))
    eta2 = ssb / (ssb + ssw)
    omega2 = max(0.0, (ssb - (k - 1) * msw) / (ssb + ssw + msw))
    return TestResult(
        test="One-way ANOVA" if equal_var else "Welch's one-way ANOVA",
        statistic=float(f_stat), p_value=float(p), stat_label="F", df=df,
        effect_size=eta2, effect_size_name="eta squared", n=int(total),
        details={"omega_squared": omega2, "ss_between": ssb, "ss_within": ssw,
                 "ms_within": msw, "group_means": means, "group_sds": np.sqrt(variances),
                 "group_sizes": n.astype(int)},
    )


def levene(*groups, center="median"):
    """Levene's test for equal variances.

    ``center='median'`` is the robust Brown-Forsythe variant (default); ``'mean'`` is the
    original Levene test.
    """
    gs = as_groups(groups)
    if center not in ("median", "mean"):
        raise ValueError("center must be 'median' or 'mean'")
    dev = [np.abs(g - (np.median(g) if center == "median" else g.mean())) for g in gs]
    k = len(dev)
    n = np.array([d.size for d in dev], dtype=float)
    total = n.sum()
    means = np.array([d.mean() for d in dev])
    grand = float(np.sum(n * means) / total)
    ssb = float(np.sum(n * (means - grand) ** 2))
    ssw = float(sum(np.sum((d - d.mean()) ** 2) for d in dev))
    if ssw == 0:
        raise ValueError("all deviations are identical; Levene's test is undefined")
    w = (ssb / (k - 1)) / (ssw / (total - k))
    df = (k - 1.0, float(total - k))
    return TestResult(
        test="Levene's test (median-centred)" if center == "median" else "Levene's test",
        statistic=float(w), p_value=float(min(1.0, f_sf(w, *df))), stat_label="W", df=df,
        n=int(total),
        details={"group_variances": [float(g.var(ddof=1)) for g in gs]},
    )
