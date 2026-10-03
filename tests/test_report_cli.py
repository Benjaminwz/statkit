import json
import re
from pathlib import Path

import pytest

import statkit as sk
from statkit.cli import main


def test_version_matches_pyproject():
    text = (Path(__file__).parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^version = "([^"]+)"', text, re.M).group(1) == sk.__version__


def test_format_p_follows_apa():
    assert sk.format_p(0.0004) == "p < .001"
    assert sk.format_p(0.026) == "p = .026"
    assert sk.format_p(0.5) == "p = .500"
    assert sk.format_p(1.0) == "p = 1.000"


def test_apa_strings():
    r = sk.TestResult(test="t", statistic=2.314, p_value=0.0263, stat_label="t", df=37.82,
                      ci=(0.1, 1.2), effect_size=0.72, effect_size_name="Hedges' g",
                      effect_size_ci=(0.08, 1.35))
    assert r.apa() == ("t(37.82) = 2.31, p = .026, 95% CI [0.10, 1.20], "
                       "g = 0.72, 95% CI [0.08, 1.35]")
    u = sk.TestResult(test="u", statistic=12.0, p_value=0.03, stat_label="U", effect_size=0.5,
                      effect_size_name="rank-biserial correlation")
    assert u.apa() == "U = 12, p = .030, r_rb = .50"
    f = sk.TestResult(test="f", statistic=4.5, p_value=0.01, stat_label="F", df=(2.0, 27.0),
                      effect_size=0.25, effect_size_name="eta squared")
    assert f.apa() == "F(2, 27) = 4.50, p = .010, η² = .25"
    r = sk.TestResult(test="r", statistic=-0.4567, p_value=0.0001, stat_label="r", df=28.0)
    assert r.apa() == "r(28) = -.46, p < .001"


def test_results_are_serialisable_and_printable():
    res = sk.ttest_ind([1, 2, 3, 4, 5.5], [3, 4, 5, 6, 9], alternative="less")  # CI has -inf
    d = json.loads(json.dumps(res.to_dict()))
    assert d["test"].startswith("Welch") and d["details"]["n_a"] == 5
    text = str(res)
    assert "Welch" in text and "n_a: 5\n" in text + "\n" and "n_a: 5.0" not in text
    anova = sk.anova_oneway([1, 2, 3], [2, 3, 4], [5, 6, 8])
    assert json.dumps(anova.to_dict())
    assert sk.chi2_contingency([[10, 20], [30, 25]]).to_dict()["details"]["expected"]


def test_summary_of_a_known_ttest():
    r = sk.ttest_ind([5.1, 4.9, 6.2, 5.8, 5.5], [4.1, 3.9, 5.2, 4.4, 4.6], equal_var=True)
    assert r.df == 8 and r.p_value < 0.05 and r.effect_size > 1
    assert r.apa().startswith("t(8) = ")


def _csvs(tmp_path, a_vals, b_vals, header="score"):
    for name, vals in (("a.csv", a_vals), ("b.csv", b_vals)):
        (tmp_path / name).write_text(header + "\n" + "\n".join(vals) + "\n", encoding="utf-8")
    return str(tmp_path / "a.csv"), str(tmp_path / "b.csv")


def test_cli_json_has_old_and_new_fields(tmp_path, capsys):
    a, b = _csvs(tmp_path, ["1", "2", "3", "4", "5.5", "7"], ["6", "7", "8", "9", "10", "12"])
    main([a, b, "--column", "score", "--n-boot", "200"])
    out = json.loads(capsys.readouterr().out)
    assert out["n_a"] == 6 and out["mean_difference"] == pytest.approx(3.75 - 52 / 6, abs=1e-9)
    assert {"permutation_p", "hedges_g", "ci95_a", "ci95_b", "seed"} <= set(out)
    assert out["parametric_test"]["test"].startswith("Welch")
    assert out["rank_test"]["test"].startswith("Mann-Whitney")
    assert out["hedges_g_ci95"][0] < out["hedges_g"] < out["hedges_g_ci95"][1]
    assert out["versions"]["statkit"] == sk.__version__


def test_cli_text_paired_and_alternative(tmp_path, capsys):
    a, b = _csvs(tmp_path, ["5.1", "", "6.0", "5.5", "4.9", "6.2", "5.7"],
                 ["4.7", "4.9", "5.1", "5.0", "4.6", "5.8", "5.2"])
    main([a, b, "--column", "score", "--paired", "--alternative", "greater", "--format", "text",
          "--n-boot", "200"])
    out = capsys.readouterr().out
    assert "Paired t-test" in out and "Wilcoxon" in out and "Hedges" in out
    main([a, b, "--column", "score", "--paired", "--n-boot", "200"])
    assert json.loads(capsys.readouterr().out)["n_a"] == 6               # incomplete pair dropped


def test_cli_reports_bad_input_plainly(tmp_path):
    a, b = _csvs(tmp_path, ["1", "2", "x"], ["1", "2", "3"])
    with pytest.raises(SystemExit, match="line 4"):
        main([a, b, "--column", "score"])
    a, b = _csvs(tmp_path, ["1", "2", "3"], ["1", "2", "3"])
    with pytest.raises(SystemExit, match="not found"):
        main([a, b, "--column", "nope"])
    a, b = _csvs(tmp_path, ["1", "1", "1"], ["1", "1", "1"])
    with pytest.raises(SystemExit, match="cannot analyse"):
        main([a, b, "--column", "score"])


def test_cli_reads_excel_style_csv_with_bom(tmp_path, capsys):
    f = tmp_path / "d.csv"
    f.write_bytes("﻿score\n1\n2\n3\n4\n".encode("utf-8"))
    main(["describe", str(f), "--column", "score"])
    assert json.loads(capsys.readouterr().out)["mean"] == 2.5


def test_cli_subcommands(capsys):
    main(["adjust", "0.01", "0.04", "0.03", "--method", "bonferroni"])
    assert json.loads(capsys.readouterr().out)["adjusted"] == pytest.approx([0.03, 0.12, 0.09])
    main(["power", "--effect-size", "0.5"])
    out = json.loads(capsys.readouterr().out)
    assert out["n_per_group"] == 64 and out["achieved_power"] >= 0.8
    main(["power", "--effect-size", "0.5", "--n", "64"])
    assert json.loads(capsys.readouterr().out)["power"] == pytest.approx(0.8015, abs=1e-3)
    with pytest.raises(SystemExit):
        main(["adjust", "0.01", "1.5"])


def test_python_dash_m_entry_point_exists():
    assert (Path(sk.__file__).parent / "__main__.py").exists()
