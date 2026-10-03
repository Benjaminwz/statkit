"""Shapiro-Wilk normality test (Royston's AS R94 approximation)."""
from __future__ import annotations

import math

import numpy as np

from ._dist import norm_ppf, norm_sf
from ._result import TestResult
from ._util import as_sample

_C1 = (0.0, 0.221157, -0.147981, -2.07119, 4.434685, -2.706056)
_C2 = (0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633)
_C3 = (0.544, -0.39978, 0.025054, -6.714e-4)
_C4 = (1.3822, -0.77857, 0.062767, -0.0020322)
_C5 = (-1.5861, -0.31082, -0.083751, 0.0038915)
_C6 = (-0.4803, -0.082676, 0.0030302)
_G = (-2.273, 0.459)


def _poly(coef, x):
    return sum(c * x ** i for i, c in enumerate(coef))


def _coefficients(n):
    half = n // 2
    if n == 3:
        return np.array([math.sqrt(0.5)])
    m = np.array([norm_ppf((i - 0.375) / (n + 0.25)) for i in range(1, half + 1)])
    summ2 = 2.0 * float(np.sum(m ** 2))
    ssumm2, rsn = math.sqrt(summ2), 1.0 / math.sqrt(n)
    a1 = _poly(_C1, rsn) - m[0] / ssumm2
    if n > 5:
        a2 = _poly(_C2, rsn) - m[1] / ssumm2
        fac = math.sqrt((summ2 - 2 * m[0] ** 2 - 2 * m[1] ** 2) / (1 - 2 * a1 ** 2 - 2 * a2 ** 2))
        return np.concatenate(([a1, a2], -m[2:] / fac))
    fac = math.sqrt((summ2 - 2 * m[0] ** 2) / (1 - 2 * a1 ** 2))
    return np.concatenate(([a1], -m[1:] / fac))


def shapiro(x):
    """Shapiro-Wilk test that ``x`` comes from a normal distribution (3 <= n <= 5000).

    Small p-values are evidence against normality. Royston (1992, 1995) approximation, as used
    by SciPy and R; the p-value is accurate for n up to about 5000.
    """
    x = np.sort(as_sample(x, "x", 3))
    n = x.size
    ssd = float(np.sum((x - x.mean()) ** 2))
    if ssd == 0:
        raise ValueError("all values are identical; the Shapiro-Wilk test is undefined")
    a = _coefficients(n)
    half = n // 2
    w = float(np.sum(a * (x[::-1][:half] - x[:half])) ** 2 / ssd)
    w = min(w, 1.0)
    if n == 3:
        p = max(0.0, 6.0 / math.pi * (math.asin(math.sqrt(w)) - math.pi / 3.0))
    elif w >= 1.0:
        p = 1.0
    else:
        y = math.log1p(-w)
        if n <= 11:
            gamma = _poly(_G, n)
            if y >= gamma:
                p = 1e-99
            else:
                y = -math.log(gamma - y)
                p = norm_sf((y - _poly(_C3, n)) / math.exp(_poly(_C4, n)))
        else:
            xx = math.log(n)
            p = norm_sf((y - _poly(_C5, xx)) / math.exp(_poly(_C6, xx)))
    details = {}
    if n > 5000:
        details["warning"] = "p-value is unreliable for n > 5000"
    return TestResult(test="Shapiro-Wilk normality test", statistic=w, p_value=float(p),
                      stat_label="W", n=n, details=details)
