"""Descriptive statistics in the form reported by journals."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ._dist import t_isf
from ._result import _plain
from ._util import as_sample, check_alpha


@dataclass(frozen=True)
class Description:
    n: int
    mean: float
    sd: float
    sem: float
    median: float
    q1: float
    q3: float
    iqr: float
    minimum: float
    maximum: float
    skewness: float
    kurtosis: float
    ci_mean: Tuple[float, float]
    conf_level: float

    def to_dict(self):
        return {k: _plain(v) for k, v in self.__dict__.items()}


def describe(x, alpha=0.05):
    """Summary statistics of one sample.

    ``skewness`` and ``kurtosis`` (excess) are the sample-size adjusted versions reported by
    SPSS, Excel and R's ``e1071`` type 2 (``G1`` and ``G2``). The CI is the t-based interval for
    the mean. Quartiles use linear interpolation (R type 7 / NumPy default).
    """
    x = as_sample(x, "x")
    alpha = check_alpha(alpha)
    n = x.size
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    sem = sd / math.sqrt(n)
    crit = t_isf(alpha / 2, n - 1)
    dev = x - mean
    m2 = float(np.mean(dev ** 2))
    skew = kurt = math.nan
    if m2 > 0:
        if n >= 3:
            skew = float(np.mean(dev ** 3) / m2 ** 1.5) * math.sqrt(n * (n - 1)) / (n - 2)
        if n >= 4:
            g2 = float(np.mean(dev ** 4) / m2 ** 2) - 3.0
            kurt = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6.0)
    q1, med, q3 = (float(v) for v in np.quantile(x, [0.25, 0.5, 0.75]))
    return Description(
        n=n, mean=mean, sd=sd, sem=sem, median=med, q1=q1, q3=q3, iqr=q3 - q1,
        minimum=float(x.min()), maximum=float(x.max()), skewness=skew, kurtosis=kurt,
        ci_mean=(mean - crit * sem, mean + crit * sem), conf_level=1.0 - alpha,
    )
