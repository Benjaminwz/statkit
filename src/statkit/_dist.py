"""Distribution functions built from ``math`` only, so statkit needs nothing but NumPy.

Everything here is cross-checked against SciPy in the test-suite. The functions are scalar
and aimed at p-values and confidence intervals, not at speed on huge arrays.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

_EPS = 1e-15
_TINY = 1e-300
_NORMAL = NormalDist()
_SQRT2 = math.sqrt(2.0)


# --------------------------------------------------------------------------- normal
def norm_cdf(x):
    return 0.5 * math.erfc(-x / _SQRT2)


def norm_sf(x):
    return 0.5 * math.erfc(x / _SQRT2)


def norm_ppf(p):
    return _NORMAL.inv_cdf(p)


def norm_isf(q):
    return -_NORMAL.inv_cdf(q)


# ------------------------------------------------------- incomplete beta / gamma
def _betacf(a, b, x):
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > _TINY else _TINY)
    h = d
    for m in range(1, 20000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > _TINY else _TINY)
        c = 1.0 + aa / c
        c = c if abs(c) > _TINY else _TINY
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > _TINY else _TINY)
        c = 1.0 + aa / c
        c = c if abs(c) > _TINY else _TINY
        de = d * c
        h *= de
        if abs(de - 1.0) < _EPS:
            break
    return h


def betainc(a, b, x):
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def gammaincc(a, x):
    """Regularised upper incomplete gamma function Q(a, x)."""
    if x <= 0.0:
        return 1.0
    lg = math.lgamma(a)
    if x < a + 1.0:                                   # series for P, return 1 - P
        ap, s = a, 1.0 / a
        d = s
        for _ in range(100000):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * _EPS:
                break
        return 1.0 - s * math.exp(-x + a * math.log(x) - lg)
    b = x + 1.0 - a                                   # continued fraction for Q
    c = 1.0 / _TINY
    d = 1.0 / b
    h = d
    for i in range(1, 100000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = d if abs(d) > _TINY else _TINY
        c = b + an / c
        c = c if abs(c) > _TINY else _TINY
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < _EPS:
            break
    return math.exp(-x + a * math.log(x) - lg) * h


# ------------------------------------------------------------------ root finding
def bisect(f, lo, hi, iters=300):
    """Bisection for a sign change of ``f`` on ``[lo, hi]`` (monotone or not)."""
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if (flo < 0) == (fhi < 0):
        raise ValueError("root is not bracketed")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if fm == 0.0:
            return mid
        if (fm < 0) == (flo < 0):
            lo, flo = mid, fm
        else:
            hi = mid
        if hi - lo <= 1e-15 * max(abs(lo), abs(hi)) + _TINY:
            break
    return 0.5 * (lo + hi)


def _upper_bracket(sf, q, start=1.0):
    hi = start
    for _ in range(2000):
        if sf(hi) <= q:
            return hi
        hi *= 2.0
    raise ValueError("could not bracket the quantile")


# ---------------------------------------------------------------------- Student t
def t_sf2(t, df):
    """Two-sided tail probability P(|T| >= |t|)."""
    if math.isinf(df):
        return 2.0 * norm_sf(abs(t))
    return betainc(df / 2.0, 0.5, df / (df + t * t))


def t_sf(t, df):
    half = 0.5 * t_sf2(t, df)
    return half if t > 0 else 1.0 - half


def t_cdf(t, df):
    return t_sf(-t, df)


def t_isf(q, df):
    """Inverse survival function: the t with P(T > t) = q."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    if q == 0.5:
        return 0.0
    if q > 0.5:
        return -t_isf(1.0 - q, df)
    if math.isinf(df):
        return norm_isf(q)
    hi = _upper_bracket(lambda t: t_sf(t, df), q)
    return bisect(lambda t: t_sf(t, df) - q, 0.0, hi)


def t_ppf(p, df):
    return -t_isf(p, df)


# --------------------------------------------------------------- chi-square, F, beta
def chi2_sf(x, df):
    return gammaincc(df / 2.0, x / 2.0)


def chi2_isf(q, df):
    hi = _upper_bracket(lambda x: chi2_sf(x, df), q, start=max(df, 1.0))
    return bisect(lambda x: chi2_sf(x, df) - q, 0.0, hi)


def f_sf(x, d1, d2):
    if x <= 0.0:
        return 1.0
    return betainc(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * x))


def f_isf(q, d1, d2):
    hi = _upper_bracket(lambda x: f_sf(x, d1, d2), q)
    return bisect(lambda x: f_sf(x, d1, d2) - q, 0.0, hi)


def beta_ppf(p, a, b):
    return bisect(lambda x: betainc(a, b, x) - p, 0.0, 1.0)


# -------------------------------------------------------- non-central t (for power, CIs)
_GL_X, _GL_W = np.polynomial.legendre.leggauss(24)


def nct_cdf(t, df, ncp):
    """CDF of the non-central t distribution, by Gauss-Legendre quadrature.

    ``P(T <= t) = E[ Phi(t * S / sqrt(df) - ncp) ]`` with ``S ~ chi(df)``.
    """
    if df > 1e8:                                       # chi(df)/sqrt(df) -> 1
        return norm_cdf(t - ncp)
    mu = math.sqrt(max(df - 0.5, 0.25))
    lo, hi = max(0.0, mu - 9.0), mu + 9.0
    panels = max(1, int(math.ceil((hi - lo) / 0.5)))
    edges = np.linspace(lo, hi, panels + 1)
    half = 0.5 * (edges[1:] - edges[:-1])
    mid = 0.5 * (edges[1:] + edges[:-1])
    s = (mid[:, None] + half[:, None] * _GL_X[None, :]).ravel()
    w = (half[:, None] * _GL_W[None, :]).ravel()
    logg = (df - 1.0) * np.log(s) - 0.5 * s * s - (df / 2.0 - 1.0) * math.log(2.0) \
        - math.lgamma(df / 2.0)
    z = (t * s / math.sqrt(df) - ncp) / _SQRT2
    phi = 0.5 * np.array([math.erfc(-v) for v in z])
    return float(min(1.0, max(0.0, np.sum(w * np.exp(logg) * phi))))


def nct_ncp_ci(t_obs, df, alpha):
    """Non-centrality parameters whose non-central t has t_obs at the 1-alpha/2 and alpha/2
    quantiles (the pivotal method used for exact CIs of standardised mean differences)."""
    def solve(target):
        lo, hi = t_obs - 10.0, t_obs + 10.0
        f = lambda d: nct_cdf(t_obs, df, d) - target   # decreasing in d  # noqa: E731
        for _ in range(60):
            if f(lo) > 0:
                break
            lo -= 2 * (hi - lo)
        for _ in range(60):
            if f(hi) < 0:
                break
            hi += 2 * (hi - lo)
        return bisect(f, lo, hi)
    return solve(1.0 - alpha / 2.0), solve(alpha / 2.0)
