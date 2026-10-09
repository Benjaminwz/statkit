"""statkit: reproducible statistics for research, in pure NumPy."""
from ._result import TestResult, format_p
from .bootstrap import bootstrap_ci, bootstrap_diff_ci
from .categorical import chi2_contingency, fisher_exact, proportion_ci
from .correlation import kendalltau, pearsonr, spearmanr
from .describe import Description, describe
from .effect_size import (
                          cliffs_delta,
                          cohens_d,
                          cohens_d_ci,
                          cohens_dz,
                          cohens_dz_ci,
                          glass_delta,
                          hedges_g,
                          hedges_g_ci,
                          prob_superiority,
)
from .equivalence import tost_1samp, tost_ind, tost_rel
from .meta import MetaResult, meta_analysis
from .multiple import (
                          adjust_pvalues,
                          benjamini_hochberg,
                          benjamini_yekutieli,
                          bonferroni,
                          hochberg,
                          holm,
                          holm_sidak,
                          sidak,
)
from .nonparametric import kruskal, mannwhitneyu, wilcoxon
from .normality import shapiro
from .parametric import anova_oneway, levene, ttest_1samp, ttest_ind, ttest_rel
from .permutation import permutation_test
from .posthoc import PosthocResult, dunn, games_howell, pairwise_ttests, tukey_hsd
from .power import min_detectable_effect, power_ttest, sample_size_ttest
from .reliability import cohens_kappa, cronbach_alpha, icc, icc_table

__all__ = [
    # descriptive & reporting
    "describe", "Description", "TestResult", "format_p",
    # parametric tests
    "ttest_ind", "ttest_rel", "ttest_1samp", "anova_oneway", "levene",
    # rank-based tests and normality
    "mannwhitneyu", "wilcoxon", "kruskal", "shapiro",
    # categorical
    "chi2_contingency", "fisher_exact", "proportion_ci",
    # correlation
    "pearsonr", "spearmanr", "kendalltau",
    # resampling
    "bootstrap_ci", "bootstrap_diff_ci", "permutation_test",
    # effect sizes
    "cohens_d", "hedges_g", "glass_delta", "cohens_dz", "cohens_d_ci", "hedges_g_ci",
    "cohens_dz_ci", "cliffs_delta", "prob_superiority",
    # multiple comparisons
    "bonferroni", "sidak", "holm", "holm_sidak", "hochberg", "benjamini_hochberg",
    "benjamini_yekutieli", "adjust_pvalues",
    # power, meta-analysis, reliability
    "power_ttest", "sample_size_ttest", "min_detectable_effect",
    "meta_analysis", "MetaResult", "cronbach_alpha", "cohens_kappa", "icc", "icc_table",
    # post-hoc comparisons
    "tukey_hsd", "games_howell", "dunn", "pairwise_ttests", "PosthocResult",
    # equivalence
    "tost_ind", "tost_rel", "tost_1samp",
]
__version__ = "0.3.0"
