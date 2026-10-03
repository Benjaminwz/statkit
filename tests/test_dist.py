"""The internal distribution functions are checked against SciPy."""
import math

import pytest

from statkit import _dist as d

st = pytest.importorskip("scipy.stats")

DFS = [1, 2, 3.7, 10, 30, 200, 5000]


@pytest.mark.parametrize("df", DFS)
def test_student_t(df):
    for t in (0.0, 0.3, 1.5, 4.0, 12.0, 40.0):
        assert d.t_sf(t, df) == pytest.approx(st.t.sf(t, df), rel=1e-9, abs=1e-300)
        assert d.t_sf2(t, df) == pytest.approx(2 * st.t.sf(t, df), rel=1e-9, abs=1e-300)
        assert d.t_cdf(-t, df) == pytest.approx(st.t.cdf(-t, df), rel=1e-9, abs=1e-300)
    for q in (0.5, 0.1, 0.025, 1e-6, 0.9, 0.975):
        # older SciPy (e.g. 1.10) returns t quantiles that are off by about 2e-9
        assert d.t_isf(q, df) == pytest.approx(st.t.isf(q, df), rel=1e-7, abs=1e-12)


@pytest.mark.parametrize("df", DFS)
def test_chi_square(df):
    for x in (0.01, 1, 5, 20, 80):
        assert d.chi2_sf(x, df) == pytest.approx(st.chi2.sf(x, df), rel=1e-9)
    for q in (0.5, 0.05, 1e-5):
        assert d.chi2_isf(q, df) == pytest.approx(st.chi2.isf(q, df), rel=1e-9)


@pytest.mark.parametrize("d1,d2", [(1, 5), (3, 20), (4, 150), (2.5, 7.3)])
def test_f_and_beta(d1, d2):
    for x in (0.1, 1, 3, 10, 50):
        assert d.f_sf(x, d1, d2) == pytest.approx(st.f.sf(x, d1, d2), rel=1e-9)
    assert d.f_isf(0.05, d1, d2) == pytest.approx(st.f.isf(0.05, d1, d2), rel=1e-9)
    assert d.beta_ppf(0.9, d1, d2) == pytest.approx(st.beta.ppf(0.9, d1, d2), rel=1e-9)


@pytest.mark.parametrize("df", [1, 2, 5, 18, 98, 998])
def test_noncentral_t(df):
    for t, nc in [(0, 0), (1, 0.5), (2.1, 2), (-1, 1), (5, 3), (30, 4)]:
        assert d.nct_cdf(t, df, nc) == pytest.approx(st.nct.cdf(t, df, nc), rel=1e-8, abs=1e-12)


def test_ncp_ci_inverts_the_cdf():
    lo, hi = d.nct_ncp_ci(2.5, 18, 0.05)
    assert st.nct.cdf(2.5, 18, lo) == pytest.approx(0.975, abs=1e-9)
    assert st.nct.cdf(2.5, 18, hi) == pytest.approx(0.025, abs=1e-9)


def test_normal_helpers():
    assert d.norm_cdf(1.96) == pytest.approx(0.9750021048517795)
    assert d.norm_isf(0.025) == pytest.approx(1.959963984540054)
    assert d.norm_sf(40) == pytest.approx(st.norm.sf(40), rel=1e-9)
    assert math.isclose(d.norm_ppf(0.5), 0.0, abs_tol=1e-15)
