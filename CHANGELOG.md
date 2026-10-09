# Changelog

## 0.3.0 (2026-10-07)

### Added
- Post-hoc comparisons returning a `PosthocResult` table: `tukey_hsd` (Tukey-Kramer, simultaneous
  intervals), `games_howell`, `dunn` (tie-corrected, adjustable) and `pairwise_ttests`.
- Studentized-range distribution (`cdf`, `sf`, `isf`) in pure NumPy, accurate to about 1e-10 against SciPy.
- Equivalence tests with two one-sided tests: `tost_ind`, `tost_rel`, `tost_1samp`.
- Intraclass correlation: `icc` and `icc_table` (ICC1, ICC2, ICC3 and their k-rater forms, F-based intervals).
- Command line: `posthoc`, `tost` and `icc` commands.
- Tests against SciPy (Tukey HSD, studentized range), statsmodels (TOST), scikit-posthocs (Dunn) and the
  Shrout & Fleiss (1979) worked example (ICC).

## 0.2.0 (2026-10-03)

A large step toward what papers and reviews expect. Still NumPy only at runtime.

### Added
- Hypothesis tests returning a `TestResult` (statistic, p, df, CI, effect size with exact CI, `.apa()`):
  `ttest_ind` (Welch / Student), `ttest_rel`, `ttest_1samp`, `anova_oneway` (classic / Welch),
  `levene`, `mannwhitneyu`, `wilcoxon`, `kruskal`, `shapiro`, `chi2_contingency`, `fisher_exact`,
  `pearsonr`, `spearmanr`, `kendalltau`.
- `describe`, `proportion_ci` (Wilson, Clopper-Pearson, Jeffreys, Agresti-Coull, Wald).
- Effect sizes: `glass_delta`, `cohens_dz`, exact non-central-t intervals (`cohens_d_ci`,
  `hedges_g_ci`, `cohens_dz_ci`), `cliffs_delta`, `prob_superiority`.
- Bootstrap: `method="bca"` / `"basic"` for `bootstrap_ci`; new `bootstrap_diff_ci` (independent or paired).
- Permutation test: `alternative=`, `paired=True`, `exact=True`, custom `statistic=`.
- Multiple comparisons: `sidak`, `holm_sidak`, `hochberg`, `benjamini_yekutieli`, `adjust_pvalues`.
- Power analysis: `power_ttest`, `sample_size_ttest`, `min_detectable_effect` (exact, non-central t).
- `meta_analysis` (fixed / random effects, DerSimonian-Laird or Paule-Mandel, prediction interval),
  `cronbach_alpha`, `cohens_kappa`.
- Command line: `--paired`, `--alternative`, `--alpha`, `--format text`; new `describe`, `adjust`
  and `power` commands; JSON output now includes the t-test, rank test, Hedges' g interval and
  library versions. `python -m statkit` works.
- Traditional Chinese README, this changelog, pinned lint rules, tests against SciPy / statsmodels / G*Power.

### Changed
- `hedges_g` uses the exact gamma-function correction instead of the approximation `1 - 3/(4 df - 1)`.
  Values change by about 1e-7 relative or less.
- `hedges_g` now validates its input like `cohens_d` (NaN, 2-D arrays and too-short samples were accepted before).
- Bootstrap resamples are generated in blocks, so memory stays bounded for large samples.
  Results for a given seed are unchanged.

### Fixed
- Command line: CSV files exported from Excel (UTF-8 with BOM) were rejected as "column not found";
  non-numeric cells and degenerate data now give a plain error message instead of a traceback.

With default options, `bootstrap_ci` and `permutation_test` give the same numbers as 0.1.0 for the same seed.

## 0.1.0

First release: Cohen's d, Hedges' g, percentile bootstrap CI, permutation test, Bonferroni / Holm /
Benjamini-Hochberg, command line for two CSV files.
