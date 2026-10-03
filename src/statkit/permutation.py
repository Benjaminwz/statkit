"""Permutation tests (Monte-Carlo with the add-one correction, or exact enumeration)."""
from __future__ import annotations

import itertools
import math

import numpy as np

from ._util import as_sample, check_alternative

_EXACT_LIMIT = 200_000
_TOL = 1e-12


def _mean_diff(x, y):
    return x.mean() - y.mean()


def _count(stats, observed, alternative):
    """Number of statistics at least as extreme as ``observed`` (with float tolerance)."""
    stats = np.asarray(stats, dtype=float)
    if alternative == "greater":
        return int(np.sum(stats >= observed - _TOL))
    if alternative == "less":
        return int(np.sum(stats <= observed + _TOL))
    return int(np.sum(np.abs(stats) >= abs(observed) - _TOL))


def _exact_stats(a, b, statistic, paired):
    na, nb = a.size, b.size
    if paired:
        if 2 ** na > _EXACT_LIMIT:
            raise ValueError(f"exact paired test needs 2**n <= {_EXACT_LIMIT}; use Monte Carlo")
        out = []
        for flips in itertools.product((False, True), repeat=na):
            mask = np.array(flips)
            out.append(statistic(np.where(mask, b, a), np.where(mask, a, b)))
        return out
    if math.comb(na + nb, na) > _EXACT_LIMIT:
        raise ValueError(f"exact test needs C(n_a + n_b, n_a) <= {_EXACT_LIMIT}; use Monte Carlo")
    pooled = np.concatenate([a, b])
    everyone = np.arange(pooled.size)
    out = []
    for idx in itertools.combinations(everyone, na):
        sel = np.zeros(pooled.size, dtype=bool)
        sel[list(idx)] = True
        out.append(statistic(pooled[sel], pooled[~sel]))
    return out


def permutation_test(a, b, n_perm=10_000, seed=None, *, alternative="two-sided",
                     statistic=None, paired=False, exact=False):
    """Permutation test for a difference between two groups.

    By default tests the difference in means with the add-one correction, so the returned
    p-value is never exactly 0: ``p = (1 + #{|T*| >= |T|}) / (1 + n_perm)``.

    * ``alternative``: ``'two-sided'`` (default), ``'greater'`` (``a`` larger) or ``'less'``.
    * ``statistic``: optional ``f(x, y) -> float`` (for example a difference in medians).
    * ``paired=True``: sign-flip test for paired data (swaps the members of random pairs).
    * ``exact=True``: enumerate every arrangement instead of sampling (small samples only);
      the p-value is then exact and ``n_perm`` and ``seed`` are ignored.

    Returns ``(observed_statistic, p_value)``. With the defaults the results for a given
    ``seed`` are identical to statkit 0.1.
    """
    alternative = check_alternative(alternative)
    statistic = statistic or _mean_diff
    a, b = as_sample(a, "a", 1), as_sample(b, "b", 1)
    if paired and a.size != b.size:
        raise ValueError("paired samples must have the same length")
    observed = float(statistic(a, b))
    if exact:
        stats = _exact_stats(a, b, statistic, paired)
        return observed, float(_count(stats, observed, alternative) / len(stats))
    if n_perm < 100:
        raise ValueError("n_perm must be at least 100")
    rng = np.random.default_rng(seed)
    stats = np.empty(n_perm)
    if paired:
        for i in range(n_perm):
            mask = rng.random(a.size) < 0.5
            stats[i] = statistic(np.where(mask, b, a), np.where(mask, a, b))
    else:
        pooled = np.concatenate([a, b])
        for i in range(n_perm):
            rng.shuffle(pooled)
            stats[i] = statistic(pooled[: a.size], pooled[a.size:])
    return observed, float((1 + _count(stats, observed, alternative)) / (1 + n_perm))
