"""Post-hoc pairwise comparisons: Tukey HSD, Games-Howell, Dunn and pairwise t-tests."""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ._dist import norm_sf, studentized_range_isf, studentized_range_sf
from ._result import format_p
from ._util import as_groups, check_alpha, rankdata, tie_term
from .multiple import adjust_pvalues
from .parametric import ttest_ind


@dataclass(frozen=True)
class PosthocResult:
    """Table of pairwise comparisons. Each row is a dict (see ``columns``)."""

    method: str
    rows: List[Dict[str, Any]]
    conf_level: Optional[float] = None
    adjust: Optional[str] = None

    __test__ = False

    def to_dict(self) -> Dict[str, Any]:
        return {"method": self.method, "conf_level": self.conf_level, "adjust": self.adjust,
                "rows": [dict(r) for r in self.rows]}

    def pvalues(self) -> np.ndarray:
        return np.array([r["p_value"] for r in self.rows])

    def significant(self, alpha: float = 0.05) -> List[tuple]:
        """Group pairs whose (adjusted) p-value is below ``alpha``."""
        return [(r["group_a"], r["group_b"]) for r in self.rows if r["p_value"] < alpha]

    def summary(self, digits: int = 3) -> str:
        has_ci = "ci" in self.rows[0] if self.rows else False
        head = f"{self.method}" + (f"  (p adjusted: {self.adjust})" if self.adjust else "")
        lines = [head]
        for r in self.rows:
            line = (f"  {r['group_a']} vs {r['group_b']}: diff = {r['difference']:.{digits}f}, "
                    f"{format_p(r['p_value'])}")
            if has_ci:
                lo, hi = r["ci"]
                line += f", {self.conf_level * 100:g}% CI [{lo:.{digits}f}, {hi:.{digits}f}]"
            lines.append(line)
        return "\n".join(lines)

    __str__ = summary


def _labels(labels, k):
    if labels is None:
        return [str(i + 1) for i in range(k)]
    if len(labels) != k:
        raise ValueError("labels must have one entry per group")
    return [str(x) for x in labels]


def tukey_hsd(*groups, labels: Optional[Sequence] = None, alpha: float = 0.05) -> PosthocResult:
    """Tukey HSD (Tukey-Kramer for unequal sizes) with studentized-range p-values and intervals.

    Assumes equal variances. ``difference`` is mean(group_a) - mean(group_b) and ``ci`` the
    simultaneous ``1 - alpha`` interval for it.
    """
    alpha = check_alpha(alpha)
    gs = as_groups(groups)
    k = len(gs)
    names = _labels(labels, k)
    n = np.array([g.size for g in gs], dtype=float)
    df = float(n.sum() - k)
    if df < 1:
        raise ValueError("not enough observations for Tukey HSD")
    means = np.array([g.mean() for g in gs])
    msw = float(sum(((g - g.mean()) ** 2).sum() for g in gs) / df)
    if msw == 0:
        raise ValueError("within-group variance is zero; Tukey HSD is undefined")
    q_crit = studentized_range_isf(alpha, k, df)
    rows = []
    for i, j in combinations(range(k), 2):
        diff = float(means[i] - means[j])
        se = math.sqrt(msw / 2.0 * (1 / n[i] + 1 / n[j]))
        q = abs(diff) / se
        rows.append({"group_a": names[i], "group_b": names[j], "difference": diff,
                     "statistic": float(q), "df": df,
                     "p_value": float(min(1.0, studentized_range_sf(q, k, df))),
                     "ci": (diff - q_crit * se, diff + q_crit * se)})
    return PosthocResult("Tukey HSD", rows, conf_level=1 - alpha)


def games_howell(*groups, labels: Optional[Sequence] = None, alpha: float = 0.05) -> PosthocResult:
    """Games-Howell: Tukey-type comparisons without assuming equal variances or sizes."""
    alpha = check_alpha(alpha)
    gs = as_groups(groups)
    k = len(gs)
    names = _labels(labels, k)
    rows = []
    for i, j in combinations(range(k), 2):
        a, b = gs[i], gs[j]
        va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
        if va + vb == 0:
            raise ValueError("both groups have zero variance; comparison is undefined")
        diff = float(a.mean() - b.mean())
        se = math.sqrt((va + vb) / 2.0)
        df = (va + vb) ** 2 / (va ** 2 / (a.size - 1) + vb ** 2 / (b.size - 1))
        q = abs(diff) / se
        q_crit = studentized_range_isf(alpha, k, df)
        rows.append({"group_a": names[i], "group_b": names[j], "difference": diff,
                     "statistic": float(q), "df": float(df),
                     "p_value": float(min(1.0, studentized_range_sf(q, k, df))),
                     "ci": (diff - q_crit * se, diff + q_crit * se)})
    return PosthocResult("Games-Howell", rows, conf_level=1 - alpha)


def dunn(*groups, labels: Optional[Sequence] = None, adjust: str = "holm") -> PosthocResult:
    """Dunn's test after Kruskal-Wallis (tie-corrected, two-sided), with p-value adjustment.

    ``difference`` is the difference in mean ranks (group_a - group_b); ``statistic`` is z.
    """
    gs = as_groups(groups, min_size=1)
    k = len(gs)
    names = _labels(labels, k)
    n = np.array([g.size for g in gs], dtype=float)
    total = float(n.sum())
    ranks, counts = rankdata(np.concatenate(gs))
    bounds = np.concatenate(([0], np.cumsum(n.astype(int))))
    mean_rank = np.array([ranks[bounds[i]:bounds[i + 1]].mean() for i in range(k)])
    var_term = total * (total + 1) / 12.0 - tie_term(counts) / (12.0 * (total - 1))
    if var_term <= 0:
        raise ValueError("all observations are tied; Dunn's test is undefined")
    pairs = list(combinations(range(k), 2))
    z, raw = [], []
    for i, j in pairs:
        zi = (mean_rank[i] - mean_rank[j]) / math.sqrt(var_term * (1 / n[i] + 1 / n[j]))
        z.append(zi)
        raw.append(min(1.0, 2.0 * norm_sf(abs(zi))))
    adj = adjust_pvalues(raw, adjust) if adjust not in (None, "none") else np.array(raw)
    rows = [{"group_a": names[i], "group_b": names[j],
             "difference": float(mean_rank[i] - mean_rank[j]), "statistic": float(zv),
             "p_raw": float(pr), "p_value": float(pa)}
            for (i, j), zv, pr, pa in zip(pairs, z, raw, adj)]
    return PosthocResult("Dunn's test", rows, adjust=adjust if adjust != "none" else None)


def pairwise_ttests(*groups, labels: Optional[Sequence] = None, equal_var: bool = False,
                    adjust: str = "holm", alpha: float = 0.05) -> PosthocResult:
    """All pairwise t-tests (Welch by default) with p-value adjustment.

    The confidence intervals are the unadjusted ones; use ``p_value`` (adjusted) for decisions.
    """
    alpha = check_alpha(alpha)
    gs = as_groups(groups)
    names = _labels(labels, len(gs))
    pairs = list(combinations(range(len(gs)), 2))
    results = [ttest_ind(gs[i], gs[j], equal_var=equal_var, alpha=alpha) for i, j in pairs]
    raw = [r.p_value for r in results]
    adj = adjust_pvalues(raw, adjust) if adjust not in (None, "none") else np.array(raw)
    rows = [{"group_a": names[i], "group_b": names[j], "difference": float(r.estimate),
             "statistic": float(r.statistic), "df": float(r.df), "p_raw": float(r.p_value),
             "p_value": float(pa), "ci": tuple(map(float, r.ci))}
            for (i, j), r, pa in zip(pairs, results, adj)]
    return PosthocResult("Pairwise " + ("Student" if equal_var else "Welch") + " t-tests", rows,
                         conf_level=1 - alpha, adjust=adjust if adjust != "none" else None)
