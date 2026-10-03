"""Inverse-variance meta-analysis (fixed effect and random effects)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ._dist import bisect, chi2_sf, norm_isf, norm_sf, t_isf
from ._result import _plain, format_p
from ._util import check_alpha


@dataclass(frozen=True)
class MetaResult:
    """Pooled effect with heterogeneity statistics.

    ``i2`` is in percent. ``prediction_interval`` (random effects, k >= 3) is the range
    expected for the effect in a new study (Higgins, Thompson & Spiegelhalter, 2009).
    """

    model: str
    k: int
    estimate: float
    se: float
    ci: Tuple[float, float]
    z: float
    p_value: float
    tau2: float
    i2: float
    h2: float
    q: float
    q_df: int
    q_p_value: float
    weights: np.ndarray
    conf_level: float
    prediction_interval: Optional[Tuple[float, float]] = None
    tau2_method: Optional[str] = None

    def to_dict(self):
        return {k: _plain(v) for k, v in self.__dict__.items()}

    def summary(self, digits: int = 3) -> str:
        pct = f"{self.conf_level * 100:g}%"
        lines = [
            f"{self.model} meta-analysis of {self.k} studies",
            f"  pooled estimate {self.estimate:.{digits}f}  (SE {self.se:.{digits}f}), "
            f"{pct} CI [{self.ci[0]:.{digits}f}, {self.ci[1]:.{digits}f}], "
            f"z = {self.z:.2f}, {format_p(self.p_value)}",
            f"  heterogeneity: Q({self.q_df}) = {self.q:.2f}, {format_p(self.q_p_value)}; "
            f"I² = {self.i2:.1f}%; tau² = {self.tau2:.{digits + 1}f}",
        ]
        if self.prediction_interval is not None:
            lo, hi = self.prediction_interval
            lines.append(f"  prediction interval [{lo:.{digits}f}, {hi:.{digits}f}]")
        return "\n".join(lines)

    __str__ = summary


def _pooled(y, v, tau2):
    w = 1.0 / (v + tau2)
    mu = float(np.sum(w * y) / np.sum(w))
    return w, mu


def meta_analysis(effects, variances=None, ses=None, model="random", tau2_method="DL",
                  alpha=0.05):
    """Combine study effects with inverse-variance weights.

    Give either ``variances`` or ``ses`` (standard errors) per study.
    ``model`` is ``'fixed'`` or ``'random'``; random-effects heterogeneity ``tau2_method``
    is ``'DL'`` (DerSimonian-Laird) or ``'PM'`` (Paule-Mandel, generally less biased). Intervals
    use the normal distribution. Effects can be any consistently-signed measure (standardised
    mean differences, log odds ratios, Fisher-z correlations, ...).
    """
    alpha = check_alpha(alpha)
    y = np.asarray(effects, dtype=float)
    if (variances is None) == (ses is None):
        raise ValueError("give exactly one of variances or ses")
    v = np.asarray(variances if variances is not None else np.square(ses), dtype=float)
    if y.ndim != 1 or v.shape != y.shape or y.size < 2:
        raise ValueError("effects and variances must be 1-D of equal length (at least 2 studies)")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(v)) and np.all(v > 0)):
        raise ValueError("effects must be finite and variances positive")
    if model not in ("fixed", "random"):
        raise ValueError("model must be 'fixed' or 'random'")
    if tau2_method not in ("DL", "PM"):
        raise ValueError("tau2_method must be 'DL' or 'PM'")
    k = y.size
    w_fe, mu_fe = _pooled(y, v, 0.0)
    q = float(np.sum(w_fe * (y - mu_fe) ** 2))
    df = k - 1
    if tau2_method == "DL":
        c = float(w_fe.sum() - np.sum(w_fe ** 2) / w_fe.sum())
        tau2_hat = max(0.0, (q - df) / c)
    else:
        def q_gen(t2):
            w, mu = _pooled(y, v, t2)
            return float(np.sum(w * (y - mu) ** 2)) - df
        if q_gen(0.0) <= 0:
            tau2_hat = 0.0
        else:
            hi = max(1.0, float(np.var(y)) * 4)
            while q_gen(hi) > 0:
                hi *= 2.0
            tau2_hat = bisect(q_gen, 0.0, hi)
    random = model == "random"
    tau2 = tau2_hat if random else 0.0
    w, mu = _pooled(y, v, tau2)
    se = math.sqrt(1.0 / float(w.sum()))
    z = mu / se
    crit = norm_isf(alpha / 2)
    pred = None
    if random and k >= 3:
        half = t_isf(alpha / 2, k - 2) * math.sqrt(tau2 + se ** 2)
        pred = (mu - half, mu + half)
    return MetaResult(
        model="Random-effects" if random else "Fixed-effect", k=k, estimate=mu, se=se,
        ci=(mu - crit * se, mu + crit * se), z=z, p_value=float(min(1.0, 2 * norm_sf(abs(z)))),
        tau2=tau2_hat, i2=max(0.0, (q - df) / q * 100.0) if q > 0 else 0.0,
        h2=q / df, q=q, q_df=df, q_p_value=float(chi2_sf(q, df)), weights=w / w.sum(),
        conf_level=1 - alpha, prediction_interval=pred, tau2_method=tau2_method if random else None,
    )
