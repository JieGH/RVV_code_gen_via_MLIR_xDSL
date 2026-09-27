#!/usr/bin/env python3
"""
Issues #35/#36: summarise the per-rep logs that `--save-reps` writes (reps_*.txt).

For each case (a row label and the nc it ran at) of each launch: GOPs by mean time (the
reference's formula, total ops over total time) and by median rep, the CV of the reps'
GOPs, the best rep, the share of slow reps, and whether the slow reps come clumped
together or scattered. Then one line per packer and case across launches.

Slow reps: the reps' GOPs are cut into two groups at the point that separates them best
(1-D two-means). If the slower group's mean is under 80% of the faster group's, the case
has two speeds and the slower group's reps are slow (B3's two speeds are ~4.4 and ~9.9
GOPs); otherwise it has one speed and no rep is slow. This works however few of the reps
are fast. Rep 0 is left out of the split, since under the reference protocol it is the
cold rep; it stays inside the mean.

Fast start: the reps (and seconds) from the start of the case up to its first slow rep.

Clumping = the number of runs of consecutive slow reps, over the number expected if the
same count of slow reps were placed at random: s*(n-s+1)/n. Near 1 is random, well under 1
is clumped (slow stretches), over 1 is alternating.

Reads reps_*.txt or reps_*.txt.gz (compile_macro_bench.sh gzips them on fetch).

Usage: reps_report.py <run dir or reps_*.txt[.gz]> [...]
"""

from __future__ import annotations

import csv
import gzip
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

TWO_SPEEDS_BELOW = 0.8  # slow group's mean under this fraction of the fast group's


@dataclass
class Case:
    ops: float
    seconds: list[float] = field(default_factory=list[float])


@dataclass
class CaseStats:
    n: int
    gops_mean_time: float
    gops_median: float
    cv_pct: float
    best: float
    slow_share: float
    slow_runs: int
    clumping: float
    fast_start_reps: int
    fast_start_s: float


def _slow_cut(gops: list[float]) -> float:
    """GOPs at or under which a rep is slow; -inf when the reps have one speed."""
    ranked = sorted(gops)
    n = len(ranked)
    if n < 2:
        return float("-inf")
    prefix = [0.0]
    for g in ranked:
        prefix.append(prefix[-1] + g)
    best_i, best_score = 0, -1.0
    for i in range(1, n):  # low group = ranked[:i]
        lo, hi = prefix[i] / i, (prefix[n] - prefix[i]) / (n - i)
        score = i * (n - i) * (hi - lo) ** 2  # between-group variance, up to a constant
        if score > best_score:
            best_i, best_score = i, score
    lo_mean = prefix[best_i] / best_i
    hi_mean = (prefix[n] - prefix[best_i]) / (n - best_i)
    return ranked[best_i - 1] if lo_mean < TWO_SPEEDS_BELOW * hi_mean else float("-inf")


def read_reps(path: Path) -> dict[tuple[str, int], Case]:
    """Rows of one reps_*.txt, grouped by (Layer, NC) in file order, reps in the order they ran."""
    cases: dict[tuple[str, int], Case] = {}
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as fh:
        for row in csv.DictReader(fh):
            key = (row["Layer"], int(row["NC"]))
            ops = 2.0 * int(row["M"]) * int(row["N"]) * int(row["K"])
            cases.setdefault(key, Case(ops)).seconds.append(float(row["Seconds"]))
    return cases


def case_stats(ops: float, seconds: list[float]) -> CaseStats:
    gops = [ops / (s * 1e9) for s in seconds]
    mean_g = statistics.fmean(gops)
    cv = statistics.pstdev(gops) / mean_g * 100 if len(gops) > 1 else 0.0

    warm = gops[1:] if len(gops) > 1 else gops
    cut = _slow_cut(warm)
    slow = [g <= cut for g in warm]
    first_slow = next((i + 1 for i, x in enumerate(slow) if x), len(gops)) if len(gops) > 1 else len(gops)
    s, n = sum(slow), len(slow)
    runs = sum(1 for i, x in enumerate(slow) if x and (i == 0 or not slow[i - 1]))
    expected = s * (n - s + 1) / n if s else 0.0
    return CaseStats(
        n=len(seconds),
        gops_mean_time=len(seconds) * ops / (sum(seconds) * 1e9),
        gops_median=statistics.median(gops),
        cv_pct=cv,
        best=max(gops),
        slow_share=s / n,
        slow_runs=runs,
        clumping=runs / expected if expected else float("nan"),
        fast_start_reps=first_slow,
        fast_start_s=sum(seconds[:first_slow]),
    )


@dataclass
class LaunchRow:
    packer: str
    layer: str
    nc: int
    launches: int
    gops_mean_time_avg: float
    gops_mean_time_lo: float
    gops_mean_time_hi: float
    gops_median_avg: float
    cv_pct_avg: float
    best: float
    slow_share_avg: float
    clumping_avg: float
    fast_start_s_avg: float


def _reps_files(target: Path) -> list[Path]:
    return [target] if target.is_file() else sorted(target.rglob("reps_*.txt*"))


def launch_table(targets: list[Path]) -> list[LaunchRow]:
    """One row per (packer, layer, nc), in first-seen order, averaged over the launches given."""
    per: dict[tuple[str, str, int], list[CaseStats]] = {}
    for target in targets:
        for f in _reps_files(target):
            packer = "scalar" if "_scalarpack" in f.name else "vector"
            for (layer, nc), case in read_reps(f).items():
                per.setdefault((packer, layer, nc), []).append(case_stats(case.ops, case.seconds))
    rows: list[LaunchRow] = []
    for (packer, layer, nc), sts in per.items():
        means = [st.gops_mean_time for st in sts]
        clumps = [st.clumping for st in sts if st.clumping == st.clumping]
        rows.append(LaunchRow(
            packer, layer, nc, len(sts),
            statistics.fmean(means), min(means), max(means),
            statistics.fmean(st.gops_median for st in sts),
            statistics.fmean(st.cv_pct for st in sts),
            max(st.best for st in sts),
            statistics.fmean(st.slow_share for st in sts),
            statistics.fmean(clumps) if clumps else float("nan"),
            statistics.fmean(st.fast_start_s for st in sts),
        ))
    return rows


def format_table(rows: list[LaunchRow]) -> str:
    out = [
        "| packer | layer | nc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | CV % | best rep | slow reps % | clumping | fast start (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| {r.packer} | {r.layer} | {r.nc} | {r.launches} | "
            f"{r.gops_mean_time_avg:.2f} ({r.gops_mean_time_lo:.2f}-{r.gops_mean_time_hi:.2f}) | "
            f"{r.gops_median_avg:.2f} | {r.cv_pct_avg:.1f} | {r.best:.2f} | "
            f"{100 * r.slow_share_avg:.0f} | {r.clumping_avg:.2f} | {r.fast_start_s_avg:.2f} |"
        )
    return "\n".join(out)


def main(argv: list[str]) -> int:
    if not argv:
        print("Usage: reps_report.py <run dir or reps_*.txt[.gz]> [...]", file=sys.stderr)
        return 2
    print(format_table(launch_table([Path(a) for a in argv])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
