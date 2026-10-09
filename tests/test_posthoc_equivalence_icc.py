"""Post-hoc tests, TOST, ICC and the studentized-range distribution.

Reference values come from SciPy (studentized range, Tukey HSD, t-tests), statsmodels (TOST),
scikit-posthocs (Dunn) and the worked example of Shrout & Fleiss (1979) for the ICC.
"""
import math

import numpy as np
import pytest

import statkit as sk
from statkit import _dist

st = pytest.importorskip("scipy.stats")

PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


@pytest.fixture(scope="module")
def groups():
    rng = np.random.default_rng(5)
    return (rng.normal(0, 1, 12), rng.normal(0.8, 1.4, 15),
            rng.normal(1.5, 0.7, 9), rng.normal(0.3, 2, 20))


# ------------------------------------------------------------- studentized range
@pytest.mark.parametrize("k", [2, 3, 6, 15])
@pytest.mark.parametrize("df", [3, 12, 60, 2000])
@pytest.mark.parametrize("q", [0.8, 2.5, 4.0, 6.5])
def test_studentized_range_cdf(k, df, q):
    assert _dist.studentized_range_cdf(q, k, df) == pytest.approx(
        st.studentized_range.cdf(q, k, df), abs=1e-9)


def test_studentized_range_critical_value():
    # Tukey table value q(0.05; k=3, df=12) = 3.773
    assert _dist.studentized_range_isf(0.05, 3, 12) == pytest.approx(3.7729289658, abs=1e-8)
    inf = _dist.studentized_range_isf(0.05, 3, float("inf"))
    assert inf == pytest.approx(st.studentized_range.isf(0.05, 3, 1e9), abs=1e-6)


# ------------------------------------------------------------------------ posthoc
def test_tukey_hsd_matches_scipy(groups):
    res = sk.tukey_hsd(*groups)
    ref = st.tukey_hsd(*groups)
    ci = ref.confidence_interval()
    for row, (i, j) in zip(res.rows, PAIRS):
        assert row["difference"] == pytest.approx(groups[i].mean() - groups[j].mean())
        assert row["p_value"] == pytest.approx(ref.pvalue[i, j], abs=1e-9)
        assert row["ci"][0] == pytest.approx(ci.low[i, j], abs=1e-8)
        assert row["ci"][1] == pytest.approx(ci.high[i, j], abs=1e-8)


def test_tukey_ci_excludes_zero_exactly_when_significant(groups):
    for row in sk.tukey_hsd(*groups).rows:
        assert (row["p_value"] < 0.05) == (row["ci"][0] > 0 or row["ci"][1] < 0)


def test_games_howell_matches_definition(groups):
    a, b, c = groups[:3]
    res = sk.games_howell(a, b, c, labels="ABC")
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    df = (va + vb) ** 2 / (va ** 2 / (a.size - 1) + vb ** 2 / (b.size - 1))
    q = abs(a.mean() - b.mean()) / math.sqrt((va + vb) / 2)
    row = res.rows[0]
    assert (row["group_a"], row["group_b"]) == ("A", "B")
    assert row["df"] == pytest.approx(df)
    assert row["p_value"] == pytest.approx(st.studentized_range.sf(q, 3, df), abs=1e-9)


def test_pairwise_ttests_match_scipy(groups):
    res = sk.pairwise_ttests(*groups, adjust="none")
    for row, (i, j) in zip(res.rows, PAIRS):
        ref = st.ttest_ind(groups[i], groups[j], equal_var=False)
        assert row["p_raw"] == pytest.approx(ref.pvalue, rel=1e-8)
        assert row["statistic"] == pytest.approx(ref.statistic, rel=1e-8)
    adjusted = sk.pairwise_ttests(*groups, adjust="holm")
    raw = [r["p_raw"] for r in adjusted.rows]
    assert np.allclose(adjusted.pvalues(), sk.adjust_pvalues(raw, "holm"))


def test_dunn_matches_hand_computation():
    # three groups, no ties: z = (R_i - R_j) / sqrt(N(N+1)/12 * (1/n_i + 1/n_j))
    a, b, c = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0]
    res = sk.dunn(a, b, c, adjust="none")
    n = 10
    ranks = {"a": 2.0, "b": 5.5, "c": 9.0}
    z = (ranks["a"] - ranks["b"]) / math.sqrt(n * (n + 1) / 12 * (1 / 3 + 1 / 4))
    assert res.rows[0]["statistic"] == pytest.approx(z)
    assert res.rows[0]["p_value"] == pytest.approx(2 * st.norm.sf(abs(z)))


def test_dunn_with_ties_and_adjustment(groups):
    rounded = [np.round(g) for g in groups]
    res = sk.dunn(*rounded, adjust="holm")
    raw = [r["p_raw"] for r in res.rows]
    assert np.allclose(res.pvalues(), sk.adjust_pvalues(raw, "holm"))
    assert all(0 <= p <= 1 for p in raw)


def test_posthoc_validation_and_output(groups):
    with pytest.raises(ValueError):
        sk.tukey_hsd(groups[0])
    with pytest.raises(ValueError):
        sk.tukey_hsd(*groups, labels=["a", "b"])
    with pytest.raises(ValueError):
        sk.tukey_hsd([1, 1, 1], [2, 2, 2])
    res = sk.tukey_hsd(*groups, labels="WXYZ")
    assert "W vs X" in str(res) and "95% CI" in str(res)
    assert res.to_dict()["rows"][0]["group_a"] == "W"
    assert all(isinstance(p, tuple) for p in res.significant())


# ------------------------------------------------------------------------- TOST
@pytest.mark.parametrize("equal_var", [False, True])
def test_tost_ind_matches_statsmodels(groups, equal_var):
    sw = pytest.importorskip("statsmodels.stats.weightstats")
    a, b = groups[0], groups[1]
    ref = sw.ttost_ind(a, b, -1.5, 1.0, usevar="pooled" if equal_var else "unequal")
    res = sk.tost_ind(a, b, -1.5, 1.0, equal_var=equal_var)
    assert res.p_value == pytest.approx(ref[0], rel=1e-8)
    assert res.details["p_lower"] == pytest.approx(ref[1][1], rel=1e-8)
    assert res.details["p_upper"] == pytest.approx(ref[2][1], rel=1e-8)


def test_tost_rel_matches_statsmodels(groups):
    sw = pytest.importorskip("statsmodels.stats.weightstats")
    x = groups[0][:10]
    y = x + np.random.default_rng(1).normal(0.1, 0.4, 10)
    assert sk.tost_rel(x, y, -0.5, 0.5).p_value == pytest.approx(
        sw.ttost_paired(x, y, -0.5, 0.5)[0], rel=1e-8)


def test_tost_decision_matches_confidence_interval(groups):
    a, b = groups[0], groups[1]
    for bounds in [(-3, 3), (-1, 1), (-0.2, 0.2), (-2, -0.1)]:
        r = sk.tost_ind(a, b, *bounds)
        inside = bounds[0] < r.ci[0] and r.ci[1] < bounds[1]
        assert r.details["equivalent"] == inside == (r.p_value < 0.05)
        assert r.conf_level == pytest.approx(0.90)


def test_tost_one_sample_and_validation():
    x = np.random.default_rng(3).normal(10.1, 1, 40)
    r = sk.tost_1samp(x, 10.0, -0.5, 0.5)
    assert r.alternative == "equivalence"
    with pytest.raises(ValueError):
        sk.tost_1samp(x, 10.0, 0.5, -0.5)
    with pytest.raises(ValueError):
        sk.tost_ind([1, 1, 1], [1, 1, 1], -1, 1)


# --------------------------------------------------------------------------- ICC
SHROUT_FLEISS = [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8],
                 [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]]


def test_icc_shrout_fleiss_example():
    t = sk.icc_table(SHROUT_FLEISS)
    expected = {"ICC1": (0.166, -0.13, 0.72), "ICC2": (0.290, 0.02, 0.76),
                "ICC3": (0.715, 0.34, 0.95), "ICC1k": (0.443, -0.88, 0.91),
                "ICC2k": (0.620, 0.07, 0.93), "ICC3k": (0.909, 0.68, 0.99)}
    for kind, (val, lo, hi) in expected.items():
        assert t[kind]["icc"] == pytest.approx(val, abs=5e-4)
        assert t[kind]["ci"][0] == pytest.approx(lo, abs=5e-3)
        assert t[kind]["ci"][1] == pytest.approx(hi, abs=5e-3)
    assert t["ICC1"]["F"] == pytest.approx(1.79, abs=5e-3)
    assert t["ICC3"]["F"] == pytest.approx(11.03, abs=5e-3)


def test_icc_single_matches_table_and_validation():
    assert sk.icc(SHROUT_FLEISS, "ICC3k") == pytest.approx(
        (0.909, 0.68, 0.99), abs=5e-3)
    perfect = np.tile(np.arange(1.0, 9.0)[:, None], (1, 3)) + 0.0
    noisy = perfect + np.random.default_rng(0).normal(0, 1e-3, perfect.shape)
    assert sk.icc(noisy, "ICC2")[0] > 0.999
    with pytest.raises(ValueError):
        sk.icc(SHROUT_FLEISS, "ICC9")
    with pytest.raises(ValueError):
        sk.icc([[1, 2, 3]])
    with pytest.raises(ValueError):
        sk.icc([[1, np.nan], [2, 3]])


# --------------------------------------------------------------------------- CLI
def test_cli_posthoc_tost_icc(tmp_path, capsys, groups):
    import json

    from statkit.cli import main

    long = tmp_path / "long.csv"
    long.write_text("g,y\n" + "".join(f"{lab},{v}\n" for lab, g in zip("abcd", groups) for v in g))
    main(["posthoc", str(long), "--group-col", "g", "--value-col", "y", "--method", "tukey"])
    out = json.loads(capsys.readouterr().out)
    ref = sk.tukey_hsd(*groups, labels="abcd")
    assert out["groups"] == list("abcd")
    assert [r["p_value"] for r in out["rows"]] == pytest.approx(list(ref.pvalues()), rel=1e-9)
    main(["posthoc", str(long), "--group-col", "g", "--value-col", "y",
          "--method", "dunn", "--format", "text"])
    assert "Dunn's test" in capsys.readouterr().out

    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("y\n" + "\n".join(map(str, groups[0])) + "\n")
    b.write_text("y\n" + "\n".join(map(str, groups[1])) + "\n")
    main(["tost", str(a), str(b), "--column", "y", "--low", "-3", "--high", "3"])
    out = json.loads(capsys.readouterr().out)
    assert out["verdict"] == "equivalent" and out["alternative"] == "equivalence"

    rat = tmp_path / "r.csv"
    rat.write_text("r1,r2,r3,r4\n" + "\n".join(",".join(map(str, r)) for r in SHROUT_FLEISS) + "\n")
    main(["icc", str(rat), "--columns", "r1", "r2", "r3", "r4"])
    out = json.loads(capsys.readouterr().out)
    assert out["icc"]["ICC3"]["icc"] == pytest.approx(0.715, abs=5e-4)
    assert out["n_subjects"] == 6


def test_cli_errors_are_readable(tmp_path):
    from statkit.cli import main

    f = tmp_path / "x.csv"
    f.write_text("g,y\na,1\na,2\n")
    with pytest.raises(SystemExit, match="at least two groups"):
        main(["posthoc", str(f), "--group-col", "g", "--value-col", "y"])
    with pytest.raises(SystemExit, match="not found"):
        main(["posthoc", str(f), "--group-col", "nope", "--value-col", "y"])
