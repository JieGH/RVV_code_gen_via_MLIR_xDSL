#!/usr/bin/env python3
"""
Issue #46: read a run_issue46.sh series.

  1. Control (plan step 2): the reference run with the per-rep record against the unchanged one,
     per board, built tile and case, launch by launch, by the mean GFLOPS the driver itself printed.
     They must agree within noise before the per-rep times are believed: every launch under 3%, and
     not the same sign in every launch (a gap that always points one way, over 1% on average and
     over the plain reference's own launch-to-launch spread, is a systematic effect of the patch, not
     noise). A case that moves more than 3% between launches of the plain reference alone (a two-speed
     case) is held to its own spread instead. Under 3 launches: no verdict.
  2. Per-rep (plan step 4): reps_report.py's table for ours and the reference's per-rep files,
     per board: slow-rep share, fast start, clumping and CV, with the run conditions and any RED NOTE
     for another user's load.

Usage: issue46_report.py <logs/issue46_<stamp> series folder>
"""

from __future__ import annotations

import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "ref_baseline"))

from ref_per_rep import load_labels  # noqa: E402  # pyright: ignore[reportMissingImports, reportUnknownVariableType]
from reps_report import MIN_LAUNCHES, conditions_lines, format_table, launch_table, load_notes  # noqa: E402

AGREE_UNDER_PCT = 3.0  # issue #46 plan step 2
SYSTEMATIC_OVER_PCT = 1.0


@dataclass
class Run:
    board: str
    launch: int
    step: str
    folder: Path


@dataclass
class ControlRow:
    board: str
    tile: str
    case: str
    launches: int
    plain: list[float]
    per_rep: list[float]
    diff_pct: list[float]  # (per-rep run - plain run) / plain run, in %, per launch
    verdict: str
    plain_spread_pct: float = 0.0  # the plain reference's own launch-to-launch range, (max - min) / mean, in %


def read_runs(series: Path) -> list[Run]:
    """The runs of a series that finished (exit 0), from its runs.tsv."""
    runs: list[Run] = []
    for line in series.joinpath("runs.tsv").read_text().splitlines()[1:]:
        board, launch, _pos, step, folder, code = line.split("\t")
        if code == "0" and folder != "-":
            runs.append(Run(board, int(launch), step, series.parent / Path(folder).name))
    return runs


def _driver_gflops(folder: Path) -> dict[tuple[str, str], float]:
    """{(tile, case label): the driver's mean GFLOPS} from a reference run's result_<tile>_<variant>.csv."""
    labels: dict[int, str] = load_labels(folder / "cases.tsv")  # pyright: ignore[reportUnknownVariableType]
    out: dict[tuple[str, str], float] = {}
    for f in sorted(folder.glob("result_*.csv")):
        tile = f.name.split("_")[1]
        for line in f.read_text().splitlines():
            parts = line.split(";")
            if len(parts) == 5 and parts[0].isdigit() and int(parts[0]) in labels:
                out[(tile, labels[int(parts[0])])] = float(parts[4])
    return out


def _spread_pct(xs: list[float]) -> float:
    return 100 * (max(xs) - min(xs)) / statistics.fmean(xs)


def _verdict(diffs: list[float], plain: list[float]) -> str:
    """Plan step 2, with the case's own noise as a second yardstick: a two-speed case (B3) moves ~10%
    between launches of the plain reference alone, so a 3% gate would fail it with no patch at all."""
    if len(diffs) < MIN_LAUNCHES:
        return f"no verdict ({len(diffs)} launches)"
    spread = _spread_pct(plain)
    mean = statistics.fmean(diffs)
    if (all(d > 0 for d in diffs) or all(d < 0 for d in diffs)) and abs(mean) > max(SYSTEMATIC_OVER_PCT, spread):
        return f"SYSTEMATIC: same sign in every launch, mean {mean:+.1f}% (plain spread {spread:.0f}%)"
    worst = max(diffs, key=abs)
    if abs(worst) < AGREE_UNDER_PCT:
        return "agree"
    if abs(worst) <= spread:
        return f"agree within the case's own noise (plain reference spread {spread:.0f}%, worst gap {worst:+.1f}%)"
    return f"DISAGREE: a launch is over {AGREE_UNDER_PCT:.0f}% ({worst:+.1f}%), more than the plain spread {spread:.0f}%"


def control(series: Path) -> list[ControlRow]:
    runs = read_runs(series)
    plain = {(r.board, r.launch): r.folder for r in runs if r.step == "ref"}
    patched = {(r.board, r.launch): r.folder for r in runs if r.step == "ref-per-rep"}
    acc: dict[tuple[str, str, str], tuple[list[float], list[float]]] = {}
    for key in sorted(plain.keys() & patched.keys()):
        p, q = _driver_gflops(plain[key]), _driver_gflops(patched[key])
        for tile_case in p.keys() & q.keys():
            slot = acc.setdefault((key[0], *tile_case), ([], []))
            slot[0].append(p[tile_case])
            slot[1].append(q[tile_case])
    rows: list[ControlRow] = []
    for (board, tile, case), (ps, qs) in sorted(acc.items()):
        diffs = [100 * (q - p) / p for p, q in zip(ps, qs)]
        rows.append(ControlRow(board, tile, case, len(ps), ps, qs, diffs, _verdict(diffs, ps),
                               _spread_pct(ps)))
    return rows


def format_control(rows: list[ControlRow]) -> str:
    out = ["| board | built tile | case | launches | plain reference GFLOPS (per launch) | with per-rep record | gap % (per launch) | plain spread % | verdict |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r.board} | {r.tile} | {r.case} | {r.launches} | {' / '.join(f'{x:.2f}' for x in r.plain)} | "
                   f"{' / '.join(f'{x:.2f}' for x in r.per_rep)} | {' / '.join(f'{x:+.1f}' for x in r.diff_pct)} | {r.plain_spread_pct:.0f} | {r.verdict} |")
    return "\n".join(out)


def report(series: Path) -> str:
    runs = read_runs(series)
    out = [f"# Issue #46 series {series.name}", "", "## 1. Control: the reference with the per-rep record against the unchanged one", "",
           "By the mean GFLOPS the driver printed (3 digits). Agree = every launch under "
           f"{AGREE_UNDER_PCT:.0f}%, or inside the plain reference's own launch-to-launch spread, and not one sign in every launch. 16x14 and 16x16 are the two even-NR builds that bracket our 16x15.", "",
           format_control(control(series)), "", "## 2. Per-rep: ours and the reference, per board", ""]
    for board in dict.fromkeys(r.board for r in runs):
        targets = [r.folder for r in runs if r.board == board and r.step in ("ours", "ref-per-rep")]
        out += [f"### {board}", "", *conditions_lines(targets), *load_notes(targets), "", format_table(launch_table(targets)), ""]
    return "\n".join(out)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("Usage: issue46_report.py <logs/issue46_<stamp> series folder>", file=sys.stderr)
        return 2
    print(report(Path(argv[0])), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
