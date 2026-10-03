"""Command line interface.

    statkit a.csv b.csv --column score          compare two groups (the original command)
    statkit describe data.csv --column score    descriptive statistics
    statkit adjust 0.01 0.04 0.03 --method holm multiple-comparison correction
    statkit power --effect-size 0.5             sample size for 80 % power (or --n for power)
"""
import argparse
import csv
import json
import math
import platform
import sys

import numpy as np

from . import (
    __version__,
    adjust_pvalues,
    bootstrap_ci,
    cliffs_delta,
    describe,
    hedges_g,
    hedges_g_ci,
    mannwhitneyu,
    permutation_test,
    power_ttest,
    sample_size_ttest,
    ttest_ind,
    ttest_rel,
    wilcoxon,
)

_COMMANDS = ("describe", "adjust", "power")


def _read_column(path, column, keep_blank=False):
    """Read one numeric column; blank cells are dropped (or kept as ``None``)."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if column not in (reader.fieldnames or []):
            raise SystemExit(f"column '{column}' not found in {path}")
        values = []
        for line, row in enumerate(reader, start=2):
            cell = (row[column] or "").strip()
            if cell == "":
                if keep_blank:
                    values.append(None)
                continue
            try:
                values.append(float(cell))
            except ValueError:
                raise SystemExit(f"{path}, line {line}: '{cell}' is not a number") from None
        return values


def _json_safe(obj):
    """JSON has no Infinity/NaN: map them to null."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _emit(result, as_text=False, text=""):
    if as_text:
        print(text)
    else:
        json.dump(_json_safe(result), sys.stdout, indent=2)
        print()


def _versions():
    return {"statkit": __version__, "numpy": np.__version__, "python": platform.python_version()}


def _compare(argv):
    ap = argparse.ArgumentParser(prog="statkit", description="Compare two groups.")
    ap.add_argument("group_a")
    ap.add_argument("group_b")
    ap.add_argument("--column", required=True)
    ap.add_argument("--paired", action="store_true",
                    help="rows are matched pairs (paired t / Wilcoxon / sign-flip tests)")
    ap.add_argument("--alternative", choices=["two-sided", "less", "greater"],
                    default="two-sided")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (default 0, for reproducibility)")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--format", choices=["json", "text"], default="json")
    args = ap.parse_args(argv)

    if args.paired:
        pairs = [(x, y) for x, y in zip(_read_column(args.group_a, args.column, True),
                                        _read_column(args.group_b, args.column, True))
                 if x is not None and y is not None]
        if not pairs:
            raise SystemExit("no complete pairs found")
        a, b = [p[0] for p in pairs], [p[1] for p in pairs]
    else:
        a = _read_column(args.group_a, args.column)
        b = _read_column(args.group_b, args.column)

    try:
        diff, p = permutation_test(a, b, seed=args.seed, alternative=args.alternative,
                                   paired=args.paired)
        parametric = (ttest_rel if args.paired else ttest_ind)(
            a, b, alternative=args.alternative, alpha=args.alpha)
        rank_test = wilcoxon(a, b, alternative=args.alternative) if args.paired \
            else mannwhitneyu(a, b, alternative=args.alternative)
        g = hedges_g(a, b)
        g_ci = hedges_g_ci(a, b, args.alpha)[1:]
        ci_a = bootstrap_ci(a, n_boot=args.n_boot, alpha=args.alpha, seed=args.seed)[1:]
        ci_b = bootstrap_ci(b, n_boot=args.n_boot, alpha=args.alpha, seed=args.seed)[1:]
    except ValueError as err:
        raise SystemExit(f"cannot analyse these data: {err}") from None

    level = f"{100 * (1 - args.alpha):g}"
    if args.format == "text":
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        text = "\n".join([
            f"n = {len(a)} / {len(b)}   mean difference = {diff:.4f}",
            f"{parametric.test}: {parametric.apa()}",
            f"{rank_test.test}: {rank_test.apa()}",
            f"Permutation test ({args.alternative}, seed {args.seed}): p = {p:.4f}",
            f"Hedges' g (pooled SD) = {g:.3f}, {level}% CI [{g_ci[0]:.3f}, {g_ci[1]:.3f}]",
            f"Bootstrap {level}% CI: group A [{ci_a[0]:.4g}, {ci_a[1]:.4g}], "
            f"group B [{ci_b[0]:.4g}, {ci_b[1]:.4g}]",
        ])
        _emit(None, True, text)
        return
    result = {
        "n_a": len(a), "n_b": len(b),
        "mean_difference": diff,
        "permutation_p": p,
        "hedges_g": g,
        f"hedges_g_ci{level}": list(g_ci),
        "cliffs_delta": cliffs_delta(a, b),
        f"ci{level}_a": list(ci_a),
        f"ci{level}_b": list(ci_b),
        "parametric_test": parametric.to_dict(),
        "rank_test": rank_test.to_dict(),
        "paired": args.paired, "alternative": args.alternative, "alpha": args.alpha,
        "seed": args.seed, "versions": _versions(),
    }
    _emit(result)


def _describe(argv):
    ap = argparse.ArgumentParser(prog="statkit describe", description="Descriptive statistics.")
    ap.add_argument("file")
    ap.add_argument("--column", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args(argv)
    try:
        _emit(describe(_read_column(args.file, args.column), args.alpha).to_dict())
    except ValueError as err:
        raise SystemExit(f"cannot analyse these data: {err}") from None


def _adjust(argv):
    ap = argparse.ArgumentParser(prog="statkit adjust",
                                 description="Multiple-comparison correction.")
    ap.add_argument("pvalues", nargs="+", type=float)
    ap.add_argument("--method", default="holm")
    args = ap.parse_args(argv)
    try:
        adjusted = adjust_pvalues(args.pvalues, args.method)
    except ValueError as err:
        raise SystemExit(str(err)) from None
    _emit({"method": args.method, "p": args.pvalues, "adjusted": adjusted.tolist()})


def _power(argv):
    ap = argparse.ArgumentParser(prog="statkit power",
                                 description="Power / sample size for t-tests.")
    ap.add_argument("--effect-size", type=float, required=True, help="Cohen's d (dz if paired)")
    ap.add_argument("--n", type=int, help="size of group 1 (or pairs): report the power")
    ap.add_argument("--power", type=float, default=0.8, help="target power (default 0.8)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--kind", choices=["two-sample", "one-sample", "paired"], default="two-sample")
    ap.add_argument("--alternative", choices=["two-sided", "less", "greater"],
                    default="two-sided")
    args = ap.parse_args(argv)
    common = dict(alpha=args.alpha, kind=args.kind, alternative=args.alternative)
    try:
        if args.n is not None:
            out = {"power": power_ttest(args.effect_size, args.n, **common), "n": args.n}
        else:
            n = sample_size_ttest(args.effect_size, args.power, **common)
            out = {"n_per_group" if args.kind == "two-sample" else "n": n,
                   "achieved_power": power_ttest(args.effect_size, n, **common)}
    except ValueError as err:
        raise SystemExit(str(err)) from None
    _emit({**out, "effect_size": args.effect_size, **common})


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in _COMMANDS:
        {"describe": _describe, "adjust": _adjust, "power": _power}[argv[0]](argv[1:])
    else:
        _compare(argv)


if __name__ == "__main__":
    main()
