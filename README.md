# statkit

**Reproducible statistics for research, in pure NumPy.** The tests, effect sizes, intervals and
reports a paper needs, with APA-style output and no dependency beyond NumPy.

[![CI](https://github.com/Benjaminwz/statkit/actions/workflows/ci.yml/badge.svg)](https://github.com/Benjaminwz/statkit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Every p-value, confidence interval and effect size is checked in the test-suite against
SciPy and statsmodels (relative error 1e-8 or better) and against published G\*Power values.
The same seed gives the same numbers on every machine. [繁體中文說明](README.zh-TW.md)

## Install

```bash
pip install git+https://github.com/Benjaminwz/statkit
```

Needs Python 3.9+ and NumPy 1.22+. For development: `pip install -e ".[dev]"`.

## Quick start

```python
import statkit as sk

control = [5.1, 4.9, 6.2, 5.8, 5.5, 4.7, 5.3, 6.0]
treated = [6.4, 7.1, 6.8, 7.5, 6.2, 7.9, 6.6, 7.2, 6.9]

res = sk.ttest_ind(control, treated)          # Welch's t-test by default
print(res.apa())
# t(14.78) = -5.84, p < .001, 95% CI [-2.07, -0.96], g = -2.69, 95% CI [-3.98, -1.36]
```

Every test returns a `TestResult` with the statistic, p-value, degrees of freedom, confidence
interval and effect size (with its own exact interval). `res.apa()` is a one-line APA 7 summary,
`print(res)` a readable report and `res.to_dict()` a plain dictionary for JSON or pandas.

```python
sk.mannwhitneyu(control, treated).apa()          # 'U = 0.50, p < .001, r_rb = -.99'
sk.shapiro(control).apa()                        # 'W = 0.97, p = .863'
sk.hedges_g_ci(treated, control)                 # (2.69, 1.36, 3.98)  exact non-central t interval
sk.bootstrap_diff_ci(treated, control, method="bca", seed=1)
                                                 # (1.518, 1.060, 2.014)
sk.permutation_test(treated, control, alternative="greater", seed=1)
                                                 # (1.518, 9.999e-05)
sk.adjust_pvalues([0.001, 0.02, 0.04, 0.3], "holm")   # [0.004 0.06  0.08  0.3 ]
sk.sample_size_ttest(0.5)                        # 64 per group for 80 % power (G*Power: 64)
sk.fisher_exact([[8, 2], [1, 5]]).apa()          # 'OR = 20.00, p = .035, 95% CI [1.42, 282.45]'
sk.proportion_ci(18, 60)                         # Wilson: (0.30, 0.199, 0.425)
```

Meta-analysis:

```python
print(sk.meta_analysis([0.30, 0.15, 0.62, -0.05, 0.41], ses=[0.2, 0.14, 0.3, 0.17, 0.22]))
# Random-effects meta-analysis of 5 studies
#   pooled estimate 0.219  (SE 0.100), 95% CI [0.022, 0.416], z = 2.18, p = .029
#   heterogeneity: Q(4) = 5.41, p = .248; I² = 26.1%; tau² = 0.0131
#   prediction interval [-0.266, 0.704]
```

## What is inside

| Area | Functions |
| --- | --- |
| Descriptive | `describe` (mean, SD, SEM, median, IQR, skewness, kurtosis, CI of the mean) |
| Parametric tests | `ttest_ind` (Welch / Student), `ttest_rel`, `ttest_1samp`, `anova_oneway` (classic / Welch), `levene` (Brown-Forsythe) |
| Rank-based tests | `mannwhitneyu`, `wilcoxon` (exact or asymptotic, tie-corrected), `kruskal` |
| Normality | `shapiro` (Shapiro-Wilk, Royston 1995) |
| Counts and proportions | `chi2_contingency` (Yates, Cramér's V), `fisher_exact`, `proportion_ci` (Wilson, Clopper-Pearson, Jeffreys, Agresti-Coull) |
| Correlation | `pearsonr` and `spearmanr` (Fisher z intervals), `kendalltau` (tau-b) |
| Effect sizes | `cohens_d`, `hedges_g`, `glass_delta`, `cohens_dz`, exact intervals `cohens_d_ci` / `hedges_g_ci` / `cohens_dz_ci`, `cliffs_delta`, `prob_superiority`; eta², omega², epsilon², Cramér's V and rank-biserial r come with the tests |
| Resampling | `bootstrap_ci` (percentile, basic, BCa), `bootstrap_diff_ci` (independent or paired), `permutation_test` (one- or two-sided, paired, exact enumeration, custom statistic) |
| Multiple comparisons | `bonferroni`, `sidak`, `holm`, `holm_sidak`, `hochberg`, `benjamini_hochberg`, `benjamini_yekutieli`, `adjust_pvalues` |
| Power and sample size | `power_ttest`, `sample_size_ttest`, `min_detectable_effect` (exact, via the non-central t) |
| Meta-analysis | `meta_analysis` (fixed / random effects, DerSimonian-Laird or Paule-Mandel, Q, I², tau², prediction interval) |
| Reliability | `cronbach_alpha` (Feldt interval), `cohens_kappa` (nominal, linear or quadratic weights, with CI) |
| Reporting | `TestResult.apa()`, `.summary()`, `.to_dict()`, `format_p` |

## Choices worth knowing

- **Welch's t-test is the default** for two groups (it loses almost nothing when variances are
  equal and is valid when they are not; Delacre et al., 2017). Pass `equal_var=True` for Student's test.
  SciPy defaults to Student, so compare with `equal_var=False` there.
- **Effect-size intervals are exact**: they invert the non-central t distribution
  (Steiger & Fouladi, 1997), not a large-sample approximation. Hedges' g uses the exact gamma-function correction.
- **Rank tests** use the exact null distribution when there are no ties and the sample is small, and the
  tie-corrected normal approximation (with continuity correction) otherwise. `method=` overrides.
- **Monte-Carlo p-values** use the add-one correction, so they are never exactly 0.
- **One-sided tests** (`alternative="less"` / `"greater"`) also return one-sided confidence bounds.
- Not included on purpose: regression, GLMs, mixed models. Use statsmodels for those.

## Reproducibility

All randomised functions take a `seed`. With the default options, `bootstrap_ci` and `permutation_test`
return exactly the same numbers as statkit 0.1 for the same seed. The JSON output of the command line
includes the statkit, NumPy and Python versions so a number in a paper can be traced back.

## Command line

```bash
statkit control.csv treated.csv --column score --format text
```
```
n = 8 / 9   mean difference = -1.5181
Welch two-sample t-test: t(14.78) = -5.84, p < .001, 95% CI [-2.07, -0.96], g = -2.69, 95% CI [-3.98, -1.36]
Mann-Whitney U test: U = 0.50, p < .001, r_rb = -.99
Permutation test (two-sided, seed 0): p = 0.0002
Hedges' g (pooled SD) = -2.690, 95% CI [-3.983, -1.355]
Bootstrap 95% CI: group A [5.1, 5.775], group B [6.633, 7.3]
```

Options: `--paired`, `--alternative less|greater`, `--alpha`, `--seed` (default 0), `--n-boot`, and the
default `--format json`. Other commands:

```bash
statkit describe data.csv --column score
statkit adjust 0.001 0.02 0.04 0.3 --method holm
statkit power --effect-size 0.5            # per-group n for 80 % power; add --n 64 for the power instead
```

CSV files exported from Excel (UTF-8 with BOM) work. `python -m statkit ...` is equivalent.

## How it is validated

`pytest` compares statkit with SciPy on randomised data for every test above (t, ANOVA, Levene, Mann-Whitney
exact and asymptotic, Wilcoxon, Kruskal-Wallis, Shapiro-Wilk, chi-square, Fisher, correlations,
proportion intervals, exact permutation tests, BCa bootstrap), and with statsmodels for multiple-comparison
corrections, meta-analysis and kappa. Power calculations are checked against G\*Power. The suite passes on
Python 3.9 with the oldest supported NumPy and on current releases.

```bash
pip install -e ".[dev]"
pytest && ruff check .
```

## Citing

See [CITATION.cff](CITATION.cff) (GitHub shows a "Cite this repository" button).

## License

MIT
