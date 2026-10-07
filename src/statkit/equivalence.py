"""Equivalence testing with two one-sided tests (TOST, Schuirmann 1987)."""
from __future__ import annotations

import math

from ._dist import t_isf, t_sf
from ._result import TestResult
from ._util import as_sample, check_alpha


def _bounds(low, high):
    low, high = float(low), float(high)
    if not low < high:
        raise ValueError("equivalence bounds must satisfy low < high")
    return low, high


def _tost(diff, se, df, low, high, alpha, name, n, extra):
    t_low = (diff - low) / se                  # H0: diff <= low
    t_up = (diff - high) / se                  # H0: diff >= high
    p_low, p_up = float(t_sf(t_low, df)), float(1.0 - t_sf(t_up, df))
    p = max(p_low, p_up)
    crit = t_isf(alpha, df)                    # (1 - 2 alpha) interval, the TOST-consistent one
    stat = t_low if p_low >= p_up else t_up
    return TestResult(
        test=name, statistic=float(stat), p_value=float(min(1.0, p)), stat_label="t", df=float(df),
        estimate=float(diff), estimate_name="difference", ci=(diff - crit * se, diff + crit * se),
        conf_level=1 - 2 * alpha, n=n, alternative="equivalence",
        details={"low": low, "high": high, "p_lower": p_low, "p_upper": p_up,
                 "equivalent": bool(p < alpha), "alpha": alpha, "se": se, **extra},
    )


def tost_ind(a, b, low, high, equal_var=False, alpha=0.05):
    """TOST for the difference ``mean(a) - mean(b)`` lying inside ``(low, high)`` (raw units).

    Equivalence is declared when both one-sided tests reject, i.e. ``p_value < alpha``; this
    happens exactly when the ``1 - 2*alpha`` confidence interval (``ci``) lies inside the bounds.
    Welch's test is used unless ``equal_var=True``.
    """
    alpha = check_alpha(alpha)
    low, high = _bounds(low, high)
    a, b = as_sample(a, "a"), as_sample(b, "b")
    na, nb = a.size, b.size
    va, vb = a.var(ddof=1), b.var(ddof=1)
    if equal_var:
        df = na + nb - 2.0
        pooled = ((na - 1) * va + (nb - 1) * vb) / df
        se = math.sqrt(pooled * (1 / na + 1 / nb))
    else:
        se2 = va / na + vb / nb
        se = math.sqrt(se2)
        df = float("nan")
        if se2 > 0:
            df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    if se == 0 or math.isnan(df):
        raise ValueError("zero variance in both samples; TOST is undefined")
    name = "TOST equivalence test (" + ("Student" if equal_var else "Welch") + ")"
    return _tost(float(a.mean() - b.mean()), se, df, low, high, alpha, name, na + nb,
                 {"n_a": na, "n_b": nb})


def tost_rel(a, b, low, high, alpha=0.05):
    """TOST for paired samples: the mean of ``a - b`` lies inside ``(low, high)``."""
    alpha = check_alpha(alpha)
    low, high = _bounds(low, high)
    a, b = as_sample(a, "a"), as_sample(b, "b")
    if a.size != b.size:
        raise ValueError("paired samples must have the same length")
    d = a - b
    sd = float(d.std(ddof=1))
    if sd == 0:
        raise ValueError("differences have zero variance; TOST is undefined")
    n = d.size
    return _tost(float(d.mean()), sd / math.sqrt(n), n - 1.0, low, high, alpha,
                 "TOST equivalence test (paired)", n, {"sd_diff": sd})


def tost_1samp(x, popmean, low, high, alpha=0.05):
    """TOST that ``mean(x) - popmean`` lies inside ``(low, high)``."""
    alpha = check_alpha(alpha)
    low, high = _bounds(low, high)
    x = as_sample(x, "x")
    sd = float(x.std(ddof=1))
    if sd == 0:
        raise ValueError("sample has zero variance; TOST is undefined")
    return _tost(float(x.mean() - popmean), sd / math.sqrt(x.size), x.size - 1.0, low, high, alpha,
                 "TOST equivalence test (one-sample)", x.size, {"mean": float(x.mean())})


__all__ = ["tost_ind", "tost_rel", "tost_1samp"]
