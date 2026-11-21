#!/usr/bin/env python3
"""
Issue #48: read `run_issue48.sh` series, the B3 microkernel-dimension sweep.

  1. Speed (the uninstrumented arms, `vec` and `scalar`), per board, shape, packer and tile: GOPs by mean
     time (average over launches, with the lowest and highest launch) and by median rep, the share of slow
     reps, the fast start and the CV, with the blocking (kc) the cache model picked for the tile.
  2. Pack and compute apart (the instrumented arms, `vec-instr` and `scalar-instr`), per board, shape,
     packer and tile: pack's share of pack + compute, and the mean pack time and compute time of a rep, in
     fast reps and in slow reps (the split of reps_report.py, made per launch, rep 0 left out). Instrumented
     numbers explain where the time goes; they are not quoted as speed (house rule 4).

The run conditions of every run folder and any RED NOTE (another user's job on the board) come first, and a
WARNING for any row seen in fewer than 3 launches (house rule 2). More than one series folder can be given:
the launches of all of them are pooled.

Usage: issue48_report.py <logs/issue48_<stamp> series folder> [...]
"""

from __future__ import annotations

import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

from reps_report import (  # noqa: E402
    MIN_LAUNCHES,
    CaseStats,
    case_stats,
    conditions_lines,
    load_notes,
    read_reps,
    reps_files,
    slow_reps,
)

LAYER_ORDER = ["B3", "B3_K256", "B2", "R24"]  # B3 is the shape under test, the rest are its controls


@dataclass
class Sample:
    """One tile of one shape in one run (one launch of one arm on one board)."""

    board: str
    launch: int
    arm: str
    layer: str
    nc: int
    tile: tuple[int, int]
    mc: int
    kc: int
    stats: CaseStats
    fast_pack: list[float]  # seconds per rep, reps 1.. split into fast and slow
    slow_pack: list[float]
    fast_compute: list[float]
    slow_compute: list[float]

    @property
    def instrumented(self) -> bool:
        return self.arm.endswith("-instr")

    @property
    def packer(self) -> str:
        return "scalar" if self.arm.startswith("scalar") else "vector"


def _tile(variant: str) -> tuple[int, int]:
    m = re.fullmatch(r"MR(\d+)_NR(\d+)", variant)
    if not m:
        raise ValueError(f"cannot read a tile from variant {variant!r}")
    return int(m.group(1)), int(m.group(2))


def _finite(xs: list[float]) -> list[float]:
    return [x for x in xs if x == x]


def read_runs(series: Path) -> list[tuple[str, int, str, Path]]:
    """(board, launch, arm, run folder) of the runs of a series that finished (exit 0), from its runs.tsv."""
    out: list[tuple[str, int, str, Path]] = []
    for line in series.joinpath("runs.tsv").read_text().splitlines()[1:]:
        board, launch, _pos, arm, folder, code = line.split("\t")
        if code == "0" and folder != "-":
            out.append((board, int(launch), arm, series.parent / Path(folder).name))
    return out


def collect(series: list[Path]) -> list[Sample]:
    samples: list[Sample] = []
    for sd in series:
        for board, launch, arm, folder in read_runs(sd):
            for f in reps_files(folder):
                for (layer, nc), case in read_reps(f).items():
                    slow = slow_reps(case.ops, case.seconds)
                    groups: dict[bool, tuple[list[float], list[float]]] = {False: ([], []), True: ([], [])}
                    for r in range(1, len(case.seconds)):
                        groups[slow[r]][0].append(case.pack_seconds[r])
                        groups[slow[r]][1].append(case.compute_seconds[r])
                    samples.append(Sample(
                        board, launch, arm, layer, nc, _tile(case.variant), case.mc, case.kc,
                        case_stats(case.ops, case.seconds),
                        _finite(groups[False][0]), _finite(groups[True][0]),
                        _finite(groups[False][1]), _finite(groups[True][1]),
                    ))
    return samples


def _order(layer: str) -> tuple[int, str]:
    return (LAYER_ORDER.index(layer) if layer in LAYER_ORDER else len(LAYER_ORDER), layer)


@dataclass
class SpeedRow:
    board: str
    packer: str
    layer: str
    nc: int
    tile: tuple[int, int]
    mc: int
    kc: int
    launches: int
    gops_mean_time_avg: float
    gops_mean_time_lo: float
    gops_mean_time_hi: float
    gops_median_avg: float
    slow_share: float
    fast_start_s: float
    cv_pct: float


def speed_rows(samples: list[Sample]) -> list[SpeedRow]:
    """One row per (board, packer, shape, tile) of the uninstrumented arms, averaged over the launches given."""
    per: dict[tuple[str, str, str, int, tuple[int, int]], list[Sample]] = {}
    for s in samples:
        if not s.instrumented:
            per.setdefault((s.board, s.packer, s.layer, s.nc, s.tile), []).append(s)
    rows: list[SpeedRow] = []
    for (board, packer, layer, nc, tile), ss in per.items():
        means = [s.stats.gops_mean_time for s in ss]
        rows.append(SpeedRow(
            board, packer, layer, nc, tile, ss[0].mc, ss[0].kc, len(ss),
            statistics.fmean(means), min(means), max(means),
            statistics.fmean(s.stats.gops_median for s in ss),
            statistics.fmean(s.stats.slow_share for s in ss),
            statistics.fmean(s.stats.fast_start_s for s in ss),
            statistics.fmean(s.stats.cv_pct for s in ss),
        ))
    rows.sort(key=lambda r: (r.board, _order(r.layer), r.packer, r.tile))
    return rows


@dataclass
class SplitRow:
    board: str
    packer: str
    layer: str
    nc: int
    tile: tuple[int, int]
    kc: int
    launches: int
    n_fast: int
    n_slow: int
    pack_share_pct: float  # pack / (pack + compute), over every rep after rep 0
    fast_pack_ms: float
    fast_compute_ms: float
    slow_pack_ms: float  # nan when the case has no slow reps
    slow_compute_ms: float


def _mean_ms(xs: list[float]) -> float:
    return 1e3 * statistics.fmean(xs) if xs else float("nan")


def split_rows(samples: list[Sample]) -> list[SplitRow]:
    """One row per (board, packer, shape, tile) of the instrumented arms, reps pooled over the launches given."""
    per: dict[tuple[str, str, str, int, tuple[int, int]], list[Sample]] = {}
    for s in samples:
        if s.instrumented:
            per.setdefault((s.board, s.packer, s.layer, s.nc, s.tile), []).append(s)
    rows: list[SplitRow] = []
    for (board, packer, layer, nc, tile), ss in per.items():
        fp = [x for s in ss for x in s.fast_pack]
        sp = [x for s in ss for x in s.slow_pack]
        fc = [x for s in ss for x in s.fast_compute]
        sc = [x for s in ss for x in s.slow_compute]
        total_pack, total_compute = sum(fp) + sum(sp), sum(fc) + sum(sc)
        rows.append(SplitRow(
            board, packer, layer, nc, tile, ss[0].kc, len(ss), len(fp), len(sp),
            100 * total_pack / (total_pack + total_compute) if total_pack + total_compute else float("nan"),
            _mean_ms(fp), _mean_ms(fc), _mean_ms(sp), _mean_ms(sc),
        ))
    rows.sort(key=lambda r: (r.board, _order(r.layer), r.packer, r.tile))
    return rows


def _ms(x: float) -> str:
    return f"{x:.3f}" if x == x else "-"


def format_speed(rows: list[SpeedRow]) -> str:
    out = [
        "| tile | kc | packer | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| {r.tile[0]}x{r.tile[1]} | {r.kc} | {r.packer} | {r.launches} | "
            f"{r.gops_mean_time_avg:.2f} ({r.gops_mean_time_lo:.2f}-{r.gops_mean_time_hi:.2f}) | "
            f"{r.gops_median_avg:.2f} | {100 * r.slow_share:.0f} | {r.fast_start_s:.2f} | {r.cv_pct:.1f} |"
        )
    return "\n".join(out)


def format_split(rows: list[SplitRow]) -> str:
    out = [
        "| tile | kc | packer | launches | fast / slow reps | pack share % | fast reps: pack / compute (ms) | slow reps: pack / compute (ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        slow = f"{_ms(r.slow_pack_ms)} / {_ms(r.slow_compute_ms)}" if r.n_slow else "no slow reps"
        out.append(
            f"| {r.tile[0]}x{r.tile[1]} | {r.kc} | {r.packer} | {r.launches} | {r.n_fast} / {r.n_slow} | "
            f"{r.pack_share_pct:.0f} | {_ms(r.fast_pack_ms)} / {_ms(r.fast_compute_ms)} | {slow} |"
        )
    return "\n".join(out)


Row = TypeVar("Row", SpeedRow, SplitRow)


def _by_board_and_layer(rows: list[Row]) -> dict[tuple[str, str], list[Row]]:
    groups: dict[tuple[str, str], list[Row]] = {}
    for r in rows:
        groups.setdefault((r.board, r.layer), []).append(r)
    return groups


def report(series: list[Path]) -> str:
    samples = collect(series)
    folders = list(dict.fromkeys(folder for sd in series for *_, folder in read_runs(sd)))
    speed, split = speed_rows(samples), split_rows(samples)
    out = [f"# Issue #48 series {', '.join(sd.name for sd in series)}", "", "## Run conditions", "",
           *conditions_lines(folders), *load_notes(folders), ""]
    out += ["## 1. Speed and slow-rep share per tile (uninstrumented, reference protocol, vector and scalar packer)", ""]
    for (board, layer), rows in _by_board_and_layer(speed).items():
        out += [f"### {board}, {layer}", "", format_speed(rows), ""]
    out += ["## 2. Pack and compute apart per tile (instrumented: debug numbers, not speed)", ""]
    for (board, layer), rows in _by_board_and_layer(split).items():
        out += [f"### {board}, {layer}", "", format_split(rows), ""]
    few = [f"{r.board} {r.packer} {r.layer} {r.tile[0]}x{r.tile[1]}" for r in speed if r.launches < MIN_LAUNCHES]
    few += [f"{r.board} {r.packer} {r.layer} {r.tile[0]}x{r.tile[1]} (instrumented)" for r in split if r.launches < MIN_LAUNCHES]
    if few:
        out += [f"WARNING: {len(few)} row(s) seen in fewer than {MIN_LAUNCHES} launches: not a result under issue #39's "
                f"house rules (one launch is not a result): {', '.join(few[:6])}" + (" ..." if len(few) > 6 else ""), ""]
    return "\n".join(out)


def main(argv: list[str]) -> int:
    if not argv:
        print("Usage: issue48_report.py <logs/issue48_<stamp> series folder> [...]", file=sys.stderr)
        return 2
    print(report([Path(a) for a in argv]), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
