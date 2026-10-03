"""Reliability and inter-rater agreement."""
from __future__ import annotations

import math

import numpy as np

from ._dist import f_isf, norm_isf
from ._util import check_alpha


def cronbach_alpha(items, alpha=0.05):
    """Cronbach's alpha for an ``n_subjects x k_items`` matrix: ``(alpha, lower, upper)``.

    The interval is Feldt's (1965) F-based interval. Rows with missing values must be removed
    beforehand.
    """
    x = np.asarray(items, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 2:
        raise ValueError("items must be a 2-D array with at least 2 subjects and 2 items")
    if not np.all(np.isfinite(x)):
        raise ValueError("items contains NaN or infinite values")
    alpha = check_alpha(alpha)
    n, k = x.shape
    total_var = float(x.sum(axis=1).var(ddof=1))
    if total_var == 0:
        raise ValueError("the total score has zero variance; alpha is undefined")
    coef = k / (k - 1) * (1.0 - float(x.var(axis=0, ddof=1).sum()) / total_var)
    dfn, dfd = n - 1, (n - 1) * (k - 1)
    lo = 1.0 - (1.0 - coef) * f_isf(alpha / 2, dfn, dfd)
    hi = 1.0 - (1.0 - coef) * f_isf(1.0 - alpha / 2, dfn, dfd)
    return coef, lo, hi


def cohens_kappa(rater1, rater2, weights=None, alpha=0.05):
    """Cohen's kappa for two raters: ``(kappa, lower, upper)``.

    ``weights`` may be ``None`` (nominal categories), ``'linear'`` or ``'quadratic'`` (ordered
    categories, sorted by value). The standard error is the large-sample one of
    Fleiss, Cohen & Everitt (1969) and the interval is ``kappa +/- z * se``.
    """
    r1, r2 = np.asarray(rater1), np.asarray(rater2)
    if r1.ndim != 1 or r1.shape != r2.shape or r1.size < 2:
        raise ValueError("ratings must be two 1-D sequences of equal length (at least 2)")
    alpha = check_alpha(alpha)
    cats = np.unique(np.concatenate([r1, r2]))
    k = cats.size
    if k < 2:
        raise ValueError("kappa is undefined when only one category is used")
    i1, i2 = np.searchsorted(cats, r1), np.searchsorted(cats, r2)
    table = np.zeros((k, k))
    np.add.at(table, (i1, i2), 1.0)
    n = table.sum()
    p = table / n
    row, col = p.sum(axis=1), p.sum(axis=0)
    idx = np.arange(k)
    dist = np.abs(idx[:, None] - idx[None, :]) / (k - 1)
    if weights is None:
        w = np.eye(k)
    elif weights == "linear":
        w = 1.0 - dist
    elif weights == "quadratic":
        w = 1.0 - dist ** 2
    else:
        raise ValueError("weights must be None, 'linear' or 'quadratic'")
    po = float(np.sum(w * p))
    pe = float(np.sum(w * np.outer(row, col)))
    if pe == 1.0:
        raise ValueError("expected agreement is 1; kappa is undefined")
    kappa = (po - pe) / (1.0 - pe)
    wbar_row = w @ col          # sum_j w_ij * p_.j
    wbar_col = w.T @ row        # sum_i w_ij * p_i.
    a = np.sum(p * (w - (wbar_row[:, None] + wbar_col[None, :]) * (1.0 - kappa)) ** 2)
    b = (kappa - pe * (1.0 - kappa)) ** 2
    se = math.sqrt(max(0.0, (a - b) / (n * (1.0 - pe) ** 2)))
    z = norm_isf(alpha / 2)
    return float(kappa), float(max(-1.0, kappa - z * se)), float(min(1.0, kappa + z * se))
