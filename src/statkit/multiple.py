"""Multiple-comparison corrections. Each returns adjusted p-values in input order."""
from __future__ import annotations

import numpy as np


def _check(pvals):
    p = np.asarray(pvals, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("pvals must be a non-empty 1-D sequence")
    if np.any((p < 0) | (p > 1)) or np.any(np.isnan(p)):
        raise ValueError("p-values must lie in [0, 1]")
    return p


def _unsort(order, adjusted):
    out = np.empty(order.size)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def bonferroni(pvals):
    """Bonferroni: multiply by the number of tests (controls the family-wise error rate)."""
    p = _check(pvals)
    return np.minimum(p * p.size, 1.0)


def sidak(pvals):
    """Sidak: ``1 - (1 - p)^m``; slightly less conservative than Bonferroni (independent tests)."""
    p = _check(pvals)
    return np.minimum(-np.expm1(p.size * np.log1p(-p)), 1.0)


def holm(pvals):
    """Holm step-down (uniformly more powerful than Bonferroni)."""
    p = _check(pvals)
    m = p.size
    order = np.argsort(p)
    return _unsort(order, np.maximum.accumulate((m - np.arange(m)) * p[order]))


def holm_sidak(pvals):
    """Holm-Sidak step-down (uniformly more powerful than Holm for independent tests)."""
    p = _check(pvals)
    m = p.size
    order = np.argsort(p)
    adj = -np.expm1((m - np.arange(m)) * np.log1p(-p[order]))
    return _unsort(order, np.maximum.accumulate(adj))


def hochberg(pvals):
    """Hochberg step-up (FWER control for independent or positively dependent tests)."""
    p = _check(pvals)
    m = p.size
    order = np.argsort(p)
    adj = np.minimum.accumulate(((m - np.arange(m)) * p[order])[::-1])[::-1]
    return _unsort(order, adj)


def _bh(p, factor):
    m = p.size
    order = np.argsort(p)[::-1]
    ranked = p[order] * m / (m - np.arange(m)) * factor
    return _unsort(order, np.minimum.accumulate(ranked))


def benjamini_hochberg(pvals):
    """Benjamini-Hochberg FDR control (valid for independent / PRDS tests)."""
    return _bh(_check(pvals), 1.0)


def benjamini_yekutieli(pvals):
    """Benjamini-Yekutieli FDR control (valid under any dependence; more conservative)."""
    p = _check(pvals)
    return _bh(p, float(np.sum(1.0 / np.arange(1, p.size + 1))))


_METHODS = {
    "bonferroni": bonferroni, "sidak": sidak, "holm": holm, "holm-sidak": holm_sidak,
    "hochberg": hochberg, "fdr_bh": benjamini_hochberg, "benjamini-hochberg": benjamini_hochberg,
    "fdr_by": benjamini_yekutieli, "benjamini-yekutieli": benjamini_yekutieli,
}


def adjust_pvalues(pvals, method="holm"):
    """Adjust ``pvals`` with a named method.

    One of ``bonferroni``, ``sidak``, ``holm``, ``holm-sidak``, ``hochberg``, ``fdr_bh``
    (Benjamini-Hochberg) or ``fdr_by`` (Benjamini-Yekutieli); the long names also work.
    """
    try:
        return _METHODS[method](pvals)
    except KeyError:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(_METHODS)}") from None
