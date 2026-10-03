"""Input validation helpers shared by all modules."""
from __future__ import annotations

import numpy as np

_ALTERNATIVES = ("two-sided", "less", "greater")


def as_sample(x, name="sample", min_size=2):
    """Convert input to a 1-D float array and validate it."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size < min_size:
        raise ValueError(f"{name} needs at least {min_size} observations")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def as_groups(groups, min_size=2):
    """Validate the groups passed to k-sample tests."""
    if len(groups) < 2:
        raise ValueError("at least two groups are required")
    return [as_sample(g, f"group {i + 1}", min_size) for i, g in enumerate(groups)]


def check_alternative(alternative):
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative must be one of {_ALTERNATIVES}, got {alternative!r}")
    return alternative


def check_alpha(alpha):
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    return float(alpha)


def rankdata(x):
    """Average ranks (1-based, ties share the mean rank) and the sizes of tie groups."""
    x = np.asarray(x, dtype=float)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    start = np.flatnonzero(np.concatenate(([True], xs[1:] != xs[:-1])))
    counts = np.diff(np.concatenate((start, [n])))
    ranks = np.empty(n)
    ranks[order] = np.repeat(start + (counts + 1) / 2.0, counts)
    return ranks, counts


def tie_term(counts):
    """sum(t^3 - t) over tie groups, the usual tie-correction ingredient."""
    c = np.asarray(counts, dtype=float)
    return float(np.sum(c ** 3 - c))
