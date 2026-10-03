"""Result container shared by the hypothesis tests, with APA-style reporting."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

# Statistics that cannot exceed 1 in absolute value are written without a leading zero (APA 7).
_BOUNDED = {"r", "rs", "tau", "r²", "η²", "ω²", "ε²", "V", "φ", "κ", "α", "r_rb"}
_SYMBOLS = {
    "Cohen's d": "d", "Hedges' g": "g", "Cohen's dz": "dz", "Hedges' gz": "gz",
    "Glass's delta": "Δ", "eta squared": "η²", "omega squared": "ω²",
    "epsilon squared": "ε²", "Cramér's V": "V", "rank-biserial correlation": "r_rb",
    "Pearson r": "r", "odds ratio": "OR",
}
_INTEGER_STATS = {"U", "W", "T"}


def format_p(p: float, digits: int = 3) -> str:
    """APA style p-value: ``p = .026`` or ``p < .001`` (no leading zero)."""
    floor = 10.0 ** -digits
    if p < floor:
        return f"p < {floor:.{digits}f}".replace("0.", ".", 1)
    return f"p = {p:.{digits}f}".replace("0.", ".", 1) if p < 1 else f"p = {p:.{digits}f}"


def _num(x: float, digits: int = 2, bounded: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NaN"
    if math.isinf(x):
        return "∞" if x > 0 else "-∞"
    s = f"{x:.{digits}f}"
    if bounded and abs(x) < 1:
        s = s.replace("0.", ".", 1)
    return s


def _df(x) -> str:
    x = float(x)
    return str(int(x)) if x == int(x) else f"{x:.2f}"


def _plain(v):
    """Convert numpy scalars/arrays (also inside containers) to JSON-friendly Python values."""
    if isinstance(v, np.ndarray):
        return [_plain(i) for i in v.tolist()]
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, dict):
        return {str(k): _plain(i) for k, i in v.items()}
    if isinstance(v, (tuple, list)):
        return [_plain(i) for i in v]
    return v


@dataclass(frozen=True)
class TestResult:
    """Outcome of a hypothesis test.

    ``estimate`` / ``ci`` describe the effect on the original scale (for example the mean
    difference); ``effect_size`` is the standardised effect with its own interval.
    ``details`` holds test-specific extras (group means, ranks, assumptions, ...).
    """

    __test__ = False  # not a pytest test class

    test: str
    statistic: float
    p_value: float
    stat_label: str = "t"
    df: Optional[Union[float, Tuple[float, float]]] = None
    estimate: Optional[float] = None
    estimate_name: Optional[str] = None
    ci: Optional[Tuple[float, float]] = None
    conf_level: float = 0.95
    effect_size: Optional[float] = None
    effect_size_name: Optional[str] = None
    effect_size_ci: Optional[Tuple[float, float]] = None
    n: Optional[int] = None
    alternative: str = "two-sided"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Plain-Python dictionary (safe for ``json.dumps``)."""
        return {k: _plain(v) for k, v in self.__dict__.items()}

    def apa(self, digits: int = 2) -> str:
        """One-line APA 7 style summary, e.g. ``t(37.82) = 2.31, p = .026, g = 0.72``."""
        label = self.stat_label
        bounded = label in _BOUNDED
        if self.df is None:
            head = label
        elif isinstance(self.df, tuple):
            head = f"{label}({', '.join(_df(d) for d in self.df)})"
        else:
            head = f"{label}({_df(self.df)})"
        stat_digits = 0 if (label in _INTEGER_STATS and float(self.statistic).is_integer()) \
            else digits
        parts = [f"{head} = {_num(self.statistic, stat_digits, bounded)}", format_p(self.p_value)]
        pct = f"{self.conf_level * 100:g}% CI"
        if self.ci is not None:
            lo, hi = self.ci
            parts.append(f"{pct} [{_num(lo, digits, bounded)}, {_num(hi, digits, bounded)}]")
        if self.effect_size is not None:
            sym = _SYMBOLS.get(self.effect_size_name or "", self.effect_size_name or "effect")
            b = sym in _BOUNDED
            es = f"{sym} = {_num(self.effect_size, digits, b)}"
            if self.effect_size_ci is not None:
                lo, hi = self.effect_size_ci
                es += f", {pct} [{_num(lo, digits, b)}, {_num(hi, digits, b)}]"
            parts.append(es)
        return ", ".join(parts)

    def summary(self, digits: int = 4) -> str:
        """Multi-line human readable summary."""
        lines = [f"{self.test}  ({self.alternative})", f"  {self.apa(digits=digits)}"]
        if self.estimate is not None:
            name = self.estimate_name or "estimate"
            lines.append(f"  {name}: {_num(self.estimate, digits)}")
        for key, val in self.details.items():
            if isinstance(val, (str, bool, int, np.integer)):
                lines.append(f"  {key}: {val}")
            elif isinstance(val, (float, np.floating)):
                lines.append(f"  {key}: {_num(float(val), digits)}")
        return "\n".join(lines)

    __str__ = summary
