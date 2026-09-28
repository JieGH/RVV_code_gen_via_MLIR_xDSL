#!/usr/bin/env python3
"""
Issues #35/#36: summarise the per-rep logs that `--save-reps` writes (reps_*.txt).

For each case (a row label and the nc it ran at) of each launch: GOPs by mean time (the
reference's formula, total ops over total time) and by median rep, the CV of the reps'
GOPs, the best rep, the share of slow reps, and whether the slow reps come clumped
together or scattered. Then one line per packer and case across launches (a launch with
--zero-pack-lines, issue #43, counts as its own packer: "vector+zeroline").

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

Issue #42: when the per-rep log carries counts (the instrumented build's Pack_Cycles /
Compute_Cycles, and a Pack_<event> / Compute_<event> pair per --raw-events event), a second
table puts fast reps beside slow reps: per case and counter, the mean per rep over the fast
reps and over the slow reps of every launch given, and slow/fast. The split is the one above,
made per launch, rep 0 left out. A count that is nan is left out; a column that is nan in
every rep (an uninstrumented log) is dropped.

Issue #39's house rules are checked on the way: every run folder's build, protocol, board
and core are printed (from its provenance.txt), a run on core 0 is flagged, and so is any case
seen in fewer than 3 launches (one launch is not a result).

Usage: reps_report.py <run dir or reps_*.txt[.gz]> [...]
"""

from __future__ import annotations

import csv
import gzip
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

MIN_LAUNCHES = 3  # issue #39, house rule 2
TWO_SPEEDS_BELOW = 0.8  # slow group's mean under this fraction of the fast group's


@dataclass
class Case:
    ops: float
    seconds: list[float] = field(default_factory=list[float])
    # Issue #42: each counter column's per-rep counts, reps in the order they ran.
    counters: dict[str, list[float]] = field(default_factory=dict[str, list[float]])


def _is_counter(column: str) -> bool:
    return column.startswith(("Pack_", "Compute_")) and not column.endswith("_Seconds")


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
        reader = csv.DictReader(fh)
        columns = [c for c in reader.fieldnames or [] if _is_counter(c)]
        for row in reader:
            key = (row["Layer"], int(row["NC"]))
            ops = 2.0 * int(row["M"]) * int(row["N"]) * int(row["K"])
            case = cases.setdefault(key, Case(ops))
            case.seconds.append(float(row["Seconds"]))
            for c in columns:
                case.counters.setdefault(c, []).append(float(row[c]))
    return cases


def slow_reps(ops: float, seconds: list[float]) -> list[bool]:
    """Which reps are slow, rep 0 included as never slow (it is left out of the split)."""
    gops = [ops / (s * 1e9) for s in seconds]
    warm = gops[1:]
    cut = _slow_cut(warm)
    return [False] + [g <= cut for g in warm] if gops else []


def case_stats(ops: float, seconds: list[float]) -> CaseStats:
    gops = [ops / (s * 1e9) for s in seconds]
    mean_g = statistics.fmean(gops)
    cv = statistics.pstdev(gops) / mean_g * 100 if len(gops) > 1 else 0.0

    mask = slow_reps(ops, seconds)
    slow = mask[1:] if len(mask) > 1 else mask  # rep 0 left out, unless it is the only rep
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


def _packer(f: Path) -> str:
    """The packer a per-rep log was made with, and whether its pack lines were cleared (#43)."""
    return ("scalar" if "_scalarpack" in f.name else "vector") + ("+zeroline" if "_zeroline" in f.name else "")


def _reps_files(target: Path) -> list[Path]:
    return [target] if target.is_file() else sorted(target.rglob("reps_*.txt*"))


def launch_table(targets: list[Path]) -> list[LaunchRow]:
    """One row per (packer, layer, nc), in first-seen order, averaged over the launches given."""
    per: dict[tuple[str, str, int], list[CaseStats]] = {}
    for target in targets:
        for f in _reps_files(target):
            packer = _packer(f)
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


@dataclass
class CounterRow:
    packer: str
    layer: str
    nc: int
    launches: int
    column: str
    n_fast: int
    n_slow: int
    fast_mean: float
    slow_mean: float

    @property
    def ratio(self) -> float:
        return self.slow_mean / self.fast_mean if self.n_slow and self.fast_mean else float("nan")


def counter_table(targets: list[Path]) -> list[CounterRow]:
    """Issue #42: one row per (packer, layer, nc, counter column), in first-seen order, over the
    launches given: the mean count per rep over fast reps and over slow reps. Each launch is
    split on its own; rep 0 of each launch is in neither group."""
    fast: dict[tuple[str, str, int, str], list[float]] = {}
    slow: dict[tuple[str, str, int, str], list[float]] = {}
    launches: dict[tuple[str, str, int], int] = {}
    for target in targets:
        for f in _reps_files(target):
            packer = _packer(f)
            for (layer, nc), case in read_reps(f).items():
                launches[(packer, layer, nc)] = launches.get((packer, layer, nc), 0) + 1
                is_slow = slow_reps(case.ops, case.seconds)
                for column, counts in case.counters.items():
                    key = (packer, layer, nc, column)
                    fast.setdefault(key, [])
                    slow.setdefault(key, [])
                    for r in range(1, len(counts)):
                        if counts[r] == counts[r]:  # not nan
                            (slow if is_slow[r] else fast)[key].append(counts[r])
    rows: list[CounterRow] = []
    for key in fast:
        f_counts, s_counts = fast[key], slow[key]
        if not f_counts and not s_counts:
            continue  # nan in every rep: not counted in this log
        rows.append(CounterRow(
            key[0], key[1], key[2], launches[key[:3]], key[3], len(f_counts), len(s_counts),
            statistics.fmean(f_counts) if f_counts else float("nan"),
            statistics.fmean(s_counts) if s_counts else float("nan"),
        ))
    return rows


def format_counter_table(rows: list[CounterRow]) -> str:
    out = [
        "| packer | layer | nc | launches | counter | fast reps: mean per rep (n) | slow reps: mean per rep (n) | slow/fast |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        slow = f"{r.slow_mean:.4g} ({r.n_slow})" if r.n_slow else "one speed, no slow reps"
        ratio = f"{r.ratio:.2f}" if r.n_slow else "-"
        out.append(
            f"| {r.packer} | {r.layer} | {r.nc} | {r.launches} | {r.column} | "
            f"{r.fast_mean:.4g} ({r.n_fast}) | {slow} | {ratio} |"
        )
    return "\n".join(out)


@dataclass
class RunConditions:
    board: str
    build: str
    protocol: str
    core: str
    raw_events: str = "unknown"
    busy_share: str = "unknown"


def run_conditions(run_dir: Path) -> RunConditions:
    """What a run folder was timed under, from the provenance.txt compile_macro_bench.sh
    writes; "unknown" for anything it does not say."""
    fields: dict[str, str] = {}
    prov = run_dir / "provenance.txt"
    if prov.is_file():
        for line in prov.read_text().splitlines():
            key, sep, value = line.partition(" : ")
            if sep and key.strip() not in fields:
                fields[key.strip()] = value.strip()
    def get(key: str) -> str:
        return fields.get(key) or "unknown"

    return RunConditions(get("board"), get("build"), get("protocol"), get("cpu pin"),
                         get("raw events"), get("busy share"))


def conditions_lines(targets: list[Path]) -> list[str]:
    """One line per run folder: board, build, protocol and core (issue #39, rule 4), and a
    WARNING for a run timed on core 0 (rule 1)."""
    out: list[str] = []
    for target in targets:
        run_dir = target if target.is_dir() else next(
            (p for p in target.parents if (p / "provenance.txt").is_file()), target.parent
        )
        c = run_conditions(run_dir)
        out.append(f"- {run_dir.name}: {c.board}, {c.build.split(' --')[0]}, {c.protocol.split(' --')[0]}, core {c.core}, "
                   f"raw events {c.raw_events}, busy {c.busy_share}")
        if c.core == "0":
            out.append(f"  WARNING: {run_dir.name} was timed on core 0, which takes every device interrupt (issue #39)")
    return out


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
    targets = [Path(a) for a in argv]
    rows = launch_table(targets)
    print("\n".join(conditions_lines(targets)))
    print()
    print(format_table(rows))
    counters = counter_table(targets)
    if counters:
        print()
        print(format_counter_table(counters))
    few = [f"{r.packer} {r.layer}" for r in rows if r.launches < MIN_LAUNCHES]
    if few:
        print(f"\nWARNING: {len(few)} case(s) seen in fewer than {MIN_LAUNCHES} launches: not a result under "
              f"issue #39's house rules (one launch is not a result): {', '.join(few[:8])}"
              + (" ..." if len(few) > 8 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
