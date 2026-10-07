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


_ICC_KINDS = ("ICC1", "ICC2", "ICC3", "ICC1k", "ICC2k", "ICC3k")


def _icc_parts(ratings):
    x = np.asarray(ratings, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 2:
        raise ValueError("ratings must be a 2-D array: at least 2 subjects x 2 raters")
    if not np.all(np.isfinite(x)):
        raise ValueError("ratings contains NaN or infinite values")
    n, k = x.shape
    grand = x.mean()
    ssr = k * float(((x.mean(axis=1) - grand) ** 2).sum())
    ssc = n * float(((x.mean(axis=0) - grand) ** 2).sum())
    sst = float(((x - grand) ** 2).sum())
    sse = sst - ssr - ssc
    msr, msc = ssr / (n - 1), ssc / (k - 1)
    mse = sse / ((n - 1) * (k - 1))
    msw = (ssc + sse) / (n * (k - 1))
    if msr == 0 and msc == 0 and mse == 0:
        raise ValueError("ratings have zero variance; ICC is undefined")
    return n, k, msr, msc, mse, msw


def icc_table(ratings, alpha=0.05):
    """All six intraclass correlation coefficients of Shrout & Fleiss (1979) / McGraw & Wong (1996).

    ``ratings`` is ``n_subjects x k_raters``. Returns ``{kind: dict}`` for ``ICC1``, ``ICC2``,
    ``ICC3`` (single rater: one-way random, two-way random, two-way mixed / consistency) and
    ``ICC1k``, ``ICC2k``, ``ICC3k`` (mean of ``k`` raters), each with ``icc``, ``ci``, ``F``,
    ``df`` and ``p_value``. The intervals are the F-based ones (Satterthwaite for ICC2 / ICC2k).
    """
    from ._dist import f_sf
    alpha = check_alpha(alpha)
    n, k, msr, msc, mse, msw = _icc_parts(ratings)

    def f_ci(f, dfn, dfd):
        return f / f_isf(alpha / 2, dfn, dfd), f * f_isf(alpha / 2, dfd, dfn)

    out = {}
    # ICC1 / ICC1k: one-way random effects
    f1, d1, d2 = msr / msw, n - 1, n * (k - 1)
    fl, fu = f_ci(f1, d1, d2)
    out["ICC1"] = dict(icc=(msr - msw) / (msr + (k - 1) * msw), F=f1, df=(d1, d2),
                       ci=((fl - 1) / (fl + k - 1), (fu - 1) / (fu + k - 1)))
    out["ICC1k"] = dict(icc=(msr - msw) / msr, F=f1, df=(d1, d2), ci=(1 - 1 / fl, 1 - 1 / fu))
    # ICC3 / ICC3k: two-way mixed, consistency
    f3, e1, e2 = msr / mse, n - 1, (n - 1) * (k - 1)
    gl, gu = f_ci(f3, e1, e2)
    out["ICC3"] = dict(icc=(msr - mse) / (msr + (k - 1) * mse), F=f3, df=(e1, e2),
                       ci=((gl - 1) / (gl + k - 1), (gu - 1) / (gu + k - 1)))
    out["ICC3k"] = dict(icc=(msr - mse) / msr, F=f3, df=(e1, e2), ci=(1 - 1 / gl, 1 - 1 / gu))
    # ICC2 / ICC2k: two-way random, absolute agreement (Satterthwaite interval)
    icc2 = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    icc2k = (msr - mse) / (msr + (msc - mse) / n)
    fj = msc / mse
    a = k * icc2 * fj + n * (1 + (k - 1) * icc2) - k * icc2
    v = ((k - 1) * (n - 1) * a ** 2) / ((n - 1) * k ** 2 * icc2 ** 2 * fj ** 2
                                        + (n * (1 + (k - 1) * icc2) - k * icc2) ** 2)
    f3c, f3u = f_isf(alpha / 2, n - 1, v), f_isf(alpha / 2, v, n - 1)
    denom = k * msc + (k * n - k - n) * mse
    lo = n * (msr - f3c * mse) / (f3c * denom + n * msr)
    hi = n * (f3u * msr - mse) / (denom + n * f3u * msr)
    out["ICC2"] = dict(icc=icc2, F=f3, df=(e1, e2), ci=(lo, hi))
    out["ICC2k"] = dict(icc=icc2k, F=f3, df=(e1, e2),
                        ci=(lo * k / (1 + lo * (k - 1)), hi * k / (1 + hi * (k - 1))))
    for rec in out.values():
        rec["p_value"] = float(min(1.0, f_sf(rec["F"], *rec["df"])))
        rec["icc"], rec["F"] = float(rec["icc"]), float(rec["F"])
        rec["ci"] = (float(rec["ci"][0]), float(rec["ci"][1]))
    return {kind: out[kind] for kind in _ICC_KINDS}


def icc(ratings, kind="ICC2", alpha=0.05):
    """One intraclass correlation coefficient: ``(icc, lower, upper)``. See :func:`icc_table`.

    Rule of thumb for choosing: ``ICC1`` if each subject is rated by different raters, ``ICC2``
    if raters are a random sample and absolute agreement matters, ``ICC3`` if these raters are
    the only ones of interest (consistency). Add ``k`` when you will use the mean of the raters.
    """
    if kind not in _ICC_KINDS:
        raise ValueError(f"kind must be one of {_ICC_KINDS}")
    rec = icc_table(ratings, alpha)[kind]
    return rec["icc"], rec["ci"][0], rec["ci"][1]
