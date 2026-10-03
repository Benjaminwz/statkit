"""Standardised effect sizes, with exact (non-central t) confidence intervals."""
from __future__ import annotations

import math

import numpy as np

from ._dist import nct_ncp_ci
from ._util import as_sample, check_alpha, rankdata


def hedges_j(df):
    """Exact small-sample correction ``J(df) = G(df/2) / (sqrt(df/2) * G((df-1)/2))``."""
    if df < 2:
        return math.nan
    return math.exp(math.lgamma(df / 2.0) - math.lgamma((df - 1) / 2.0)) / math.sqrt(df / 2.0)


def cohens_d(a, b):
    """Cohen's d for two independent samples (pooled standard deviation)."""
    a, b = as_sample(a, "a"), as_sample(b, "b")
    na, nb = a.size, b.size
    pooled_var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if pooled_var == 0:
        raise ValueError("pooled variance is zero; effect size is undefined")
    return float((a.mean() - b.mean()) / math.sqrt(pooled_var))


def hedges_g(a, b):
    """Hedges' g: Cohen's d times the exact small-sample bias correction ``J(n_a + n_b - 2)``."""
    a, b = as_sample(a, "a"), as_sample(b, "b")
    return cohens_d(a, b) * hedges_j(a.size + b.size - 2)


def glass_delta(treatment, control):
    """Glass's delta: mean difference standardised by the *control* group's SD."""
    t, c = as_sample(treatment, "treatment"), as_sample(control, "control")
    sd = c.std(ddof=1)
    if sd == 0:
        raise ValueError("control group has zero variance; delta is undefined")
    return float((t.mean() - c.mean()) / sd)


def _differences(x, y, mu):
    x = as_sample(x, "x")
    if y is not None:
        y = as_sample(y, "y")
        if y.size != x.size:
            raise ValueError("paired samples must have the same length")
        x = x - y
    return x - mu


def cohens_dz(x, y=None, mu=0.0, bias_correct=False):
    """Standardised mean of paired differences (``y`` given) or one-sample effect vs ``mu``.

    ``bias_correct=True`` applies Hedges' correction with ``n - 1`` degrees of freedom.
    """
    diff = _differences(x, y, mu)
    sd = diff.std(ddof=1)
    if sd == 0:
        raise ValueError("differences have zero variance; effect size is undefined")
    dz = float(diff.mean() / sd)
    return dz * hedges_j(diff.size - 1) if bias_correct else dz


def _smd_ci(d, scale, df, alpha):
    """Pivotal CI: invert the non-central t distribution (Steiger & Fouladi, 1997)."""
    lo, hi = nct_ncp_ci(d / scale, df, check_alpha(alpha))
    return lo * scale, hi * scale


def cohens_d_ci(a, b, alpha=0.05):
    """``(d, lower, upper)``: Cohen's d with an exact non-central-t confidence interval."""
    a, b = as_sample(a, "a"), as_sample(b, "b")
    d = cohens_d(a, b)
    lo, hi = _smd_ci(d, math.sqrt(1 / a.size + 1 / b.size), a.size + b.size - 2, alpha)
    return d, lo, hi


def hedges_g_ci(a, b, alpha=0.05):
    """``(g, lower, upper)``: Hedges' g with an exact confidence interval."""
    a, b = as_sample(a, "a"), as_sample(b, "b")
    d, lo, hi = cohens_d_ci(a, b, alpha)
    j = hedges_j(a.size + b.size - 2)
    return d * j, lo * j, hi * j


def cohens_dz_ci(x, y=None, mu=0.0, alpha=0.05, bias_correct=False):
    """``(dz, lower, upper)`` for paired / one-sample designs, exact interval."""
    diff = _differences(x, y, mu)
    dz = cohens_dz(diff, None, 0.0)
    lo, hi = _smd_ci(dz, 1 / math.sqrt(diff.size), diff.size - 1, alpha)
    if bias_correct:
        j = hedges_j(diff.size - 1)
        return dz * j, lo * j, hi * j
    return dz, lo, hi


def prob_superiority(a, b):
    """Probability that a random value of ``a`` exceeds one of ``b`` (ties count half).

    Also known as the common-language effect size or Vargha-Delaney A.
    """
    a, b = as_sample(a, "a", 1), as_sample(b, "b", 1)
    ranks, _ = rankdata(np.concatenate([a, b]))
    u1 = ranks[: a.size].sum() - a.size * (a.size + 1) / 2.0
    return float(u1 / (a.size * b.size))


def cliffs_delta(a, b):
    """Cliff's delta, ``P(a > b) - P(a < b)`` in ``[-1, 1]`` (rank-biserial correlation)."""
    return 2.0 * prob_superiority(a, b) - 1.0
