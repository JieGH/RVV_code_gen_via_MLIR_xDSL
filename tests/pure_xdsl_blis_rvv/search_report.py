#!/usr/bin/env python3
"""
Issue #60: read a Tile Search, one compile_macro_bench.sh run folder per launch of a search config (one pinned row per
Candidate Tile, e.g. config_yolo_search.json), and name the fastest tile per shape.

  1. Per shape: the best tile by GOPs by mean time (average over launches, with the lowest and highest launch) and the
     best by GOPs by median rep, both of the whole call (packing included), and an optional compare tile.
  2. The best single tile over the whole dataset: the tile with the least total time for one pass over the shapes,
     each shape counted once (only tiles that ran every shape).
  3. The best tile per n, by the same total time.

Reads the per-rep logs (reps_*.txt[.gz], on by default; a run made with --no-save-reps has none and is an error).
Prints each run folder's conditions, a RED NOTE for a run made under another user's load, and a WARNING for any
shape seen in fewer than 3 launches (house rule 2, issue #39).

Usage: search_report.py <run folder> [...] [--compare TILE] [--compare n=N:TILE]
  --compare 16x8 --compare n=4:16x4   compare with 16x8, and with 16x4 on shapes where n = 4
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from reps_report import (  # noqa: E402
    MIN_LAUNCHES,
    CaseStats,
    case_stats,
    conditions_lines,
    load_notes,
    packer_of,
    read_reps,
    red,
    reps_files,
)

Shape = tuple[int, int, int]


def _tile_label(variant: str) -> str:
    mr, nr = variant.removeprefix("MR").split("_NR")
    return f"{int(mr)}x{int(nr)}"


def collect(targets: list[Path]) -> tuple[dict[str, Shape], dict[str, dict[str, list[CaseStats]]]]:
    """Shapes by label, and per label and tile one CaseStats per launch (run folder)."""
    shapes: dict[str, Shape] = {}
    stats: dict[str, dict[str, list[CaseStats]]] = {}
    packers: set[str] = set()
    for target in targets:
        files = reps_files(target)
        if not files:
            sys.exit(f"search_report.py: no per-rep logs (reps_*.txt) under {target}: was it run with --no-save-reps?")
        for f in files:
            packers.add(packer_of(f))
            for (layer, _nc), case in read_reps(f).items():
                shapes[layer] = (case.m, case.n, case.k)
                stats.setdefault(layer, {}).setdefault(_tile_label(case.variant), []).append(case_stats(case.ops, case.seconds))
    if len(packers) > 1:
        sys.exit(f"search_report.py: the run folders mix packers ({', '.join(sorted(packers))}); give one packer's runs")
    return shapes, stats


def _mean(sts: list[CaseStats]) -> float:
    return statistics.fmean(s.gops_mean_time for s in sts)


def _median(sts: list[CaseStats]) -> float:
    return statistics.fmean(s.gops_median for s in sts)


def _compare_tile(compare: list[str], n: int) -> str | None:
    """The last --compare that applies to n: "n=4:16x4" applies only when n = 4, a bare tile always."""
    pick = None
    for c in compare:
        cond, _, tile = c.rpartition(":")
        if not cond or cond == f"n={n}":
            pick = tile
    return pick


def _best_by_total(shapes: dict[str, Shape], stats: dict[str, dict[str, list[CaseStats]]], labels: list[str]
                   ) -> list[tuple[str, float, float, float]]:
    """(tile, ms per pass by mean time, GOPs over the pass, ms per pass by median rep), fastest first.
    Only tiles that ran every label."""
    tiles = set(stats[labels[0]]).intersection(*(stats[lb] for lb in labels[1:]))
    ops = {lb: 2.0 * shapes[lb][0] * shapes[lb][1] * shapes[lb][2] for lb in labels}
    total_ops = sum(ops.values())
    rows: list[tuple[str, float, float, float]] = []
    for t in tiles:
        ns = sum(ops[lb] / _mean(stats[lb][t]) for lb in labels)
        ns_med = sum(ops[lb] / _median(stats[lb][t]) for lb in labels)
        rows.append((t, ns / 1e6, total_ops / ns, ns_med / 1e6))
    return sorted(rows, key=lambda r: r[1])


def report(targets: list[Path], compare: list[str]) -> str:
    shapes, stats = collect(targets)
    notes = load_notes(targets)
    out = ["# Tile Search", "", *conditions_lines(targets), *notes, ""]

    out += ["## Best tile per shape (whole call, packing included)", "",
            "| shape | m | n | k | launches | best by mean time | GOPs by mean time (avg, lo-hi) | best by median rep | "
            "GOPs by median rep | compare tile | its GOPs by mean time |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    few: list[str] = []
    uneven: list[str] = []
    for lb, per_tile in stats.items():
        m, n, k = shapes[lb]
        counts = {len(sts) for sts in per_tile.values()}
        launches = min(counts)
        if launches < MIN_LAUNCHES:
            few.append(lb)
        if len(counts) > 1:
            uneven.append(lb)
        best = max(per_tile, key=lambda t: _mean(per_tile[t]))
        best_med = max(per_tile, key=lambda t: _median(per_tile[t]))
        means = [s.gops_mean_time for s in per_tile[best]]
        ct = _compare_tile(compare, n)
        ct_gops = f"{_mean(per_tile[ct]):.2f}" if ct in per_tile else ("not run" if ct else "")
        out.append(f"| {lb} | {m} | {n} | {k} | {launches} | {best} | {_mean(per_tile[best]):.2f} "
                   f"({min(means):.2f}-{max(means):.2f}) | {best_med} | {_median(per_tile[best_med]):.2f} | "
                   f"{ct or ''} | {ct_gops} |")

    labels = list(stats)
    out += ["", f"## Best single tile over the dataset ({len(labels)} shapes, each counted once)", "",
            "| tile | ms per pass (by mean time) | GOPs over the pass | ms per pass (by median rep) |", "|---|---|---|---|"]
    out += [f"| {t} | {ms:.4f} | {g:.2f} | {ms_med:.4f} |" for t, ms, g, ms_med in _best_by_total(shapes, stats, labels)]

    out += ["", "## Best tile per n (least total time over that n's shapes)", "",
            "| n | shapes | best tile | ms per pass | GOPs |", "|---|---|---|---|---|"]
    for n in sorted({shapes[lb][1] for lb in labels}):
        group = [lb for lb in labels if shapes[lb][1] == n]
        ranked = _best_by_total(shapes, stats, group)
        if not ranked:
            out.append(f"| {n} | {len(group)} | no tile ran every shape | | |")
            continue
        t, ms, g, _ = ranked[0]
        out.append(f"| {n} | {len(group)} | {t} | {ms:.4f} | {g:.2f} |")

    if few:
        out.append(f"\nWARNING: {len(few)} shape(s) seen in fewer than {MIN_LAUNCHES} launches: not a result under "
                   f"issue #39's house rules: {', '.join(few[:8])}" + (" ..." if len(few) > 8 else ""))
    if uneven:
        out.append(f"\nWARNING: {len(uneven)} shape(s) where the tiles ran a different number of launches (a tile "
                   f"missing from a run folder): their tiles are not compared like for like: {', '.join(uneven[:8])}"
                   + (" ..." if len(uneven) > 8 else ""))
    if notes:
        out.append("\n" + red("🔴 RED NOTE: some run folders above were made under another user's load."))
    return "\n".join(out)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="compile_macro_bench.sh run folders, one per launch")
    ap.add_argument("--compare", action="append", default=[], help="TILE, or n=N:TILE for shapes with that n")
    args = ap.parse_args(argv)
    print(report(args.runs, args.compare))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
