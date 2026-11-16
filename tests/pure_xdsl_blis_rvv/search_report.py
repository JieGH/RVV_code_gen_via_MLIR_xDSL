#!/usr/bin/env python3
# pyright: reportUnknownMemberType=false
# (matplotlib ships partial types; the heatmap is its only use here)
"""
Issue #60: read a Tile Search, one compile_macro_bench.sh run folder per launch of a search config (one pinned row per
Candidate Tile, e.g. config_yolo_search.json), and name the fastest tile per shape.

  1. Per shape: the best tile by GOPs by mean time (average over launches, with the lowest and highest launch) and the
     best by GOPs by median rep, both of the whole call (packing included), and an optional compare tile.
  2. The best single tile over the whole dataset: the tile with the least total time for one pass over the shapes,
     each shape counted once (only tiles that ran every shape).
  3. The best tile per n, by the same total time.
  4. A grid of every tile's GOPs by mean time per shape, shapes sorted by n, m, k; --heatmap draws it as a PNG.

Reads the per-rep logs (reps_*.txt[.gz], on by default; a run made with --no-save-reps has none and is an error).
Prints each run folder's conditions, a RED NOTE for a run made under another user's load, and a WARNING for any
shape seen in fewer than 3 launches (house rule 2, issue #39).

Usage: search_report.py <run folder> [...] [--compare TILE] [--compare n=N:TILE] [--heatmap PNG]
  --compare 16x8 --compare n=4:16x4   compare with 16x8, and with 16x4 on shapes where n = 4
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import NamedTuple

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
Stats = dict[str, dict[str, list[CaseStats]]]  # shape label -> tile label -> one CaseStats per launch


class PassTime(NamedTuple):
    """One tile's time for one pass over a set of shapes, each shape once."""
    tile: str
    ms_by_mean: float
    gops: float
    ms_by_median: float


def _tile_label(variant: str) -> str:
    mr, nr = variant.removeprefix("MR").split("_NR")
    return f"{int(mr)}x{int(nr)}"


def collect(targets: list[Path]) -> tuple[dict[str, Shape], Stats]:
    """Shapes by label, and per label and tile one CaseStats per launch (run folder)."""
    shapes: dict[str, Shape] = {}
    stats: Stats = {}
    packers: set[str] = set()
    for target in targets:
        files = reps_files(target)
        if not files:
            sys.exit(f"search_report.py: no per-rep logs (reps_*.txt) under {target}: was it run with --no-save-reps?")
        for f in files:
            packers.add(packer_of(f))
            for (label, _nc), case in read_reps(f).items():
                shape = (case.m, case.n, case.k)
                if shapes.setdefault(label, shape) != shape:
                    sys.exit(f"search_report.py: shape label {label} is {shapes[label]} in one log and {shape} in "
                             f"{f}: give runs of one dataset")
                stats.setdefault(label, {}).setdefault(_tile_label(case.variant), []).append(
                    case_stats(case.ops, case.seconds))
    if len(packers) > 1:
        sys.exit(f"search_report.py: the run folders mix packers ({', '.join(sorted(packers))}); give one packer's runs")
    return shapes, stats


def _mean(sts: list[CaseStats]) -> float:
    return statistics.fmean(s.gops_mean_time for s in sts)


def _median(sts: list[CaseStats]) -> float:
    return statistics.fmean(s.gops_median for s in sts)


def _spread(values: list[float]) -> str:
    return f"{statistics.fmean(values):.2f} ({min(values):.2f}-{max(values):.2f})"


def _compare_tile(compare: list[str], n: int) -> str | None:
    """The last --compare that applies to n: "n=4:16x4" applies only when n = 4, a bare tile always."""
    pick = None
    for c in compare:
        cond, _, tile = c.rpartition(":")
        if not cond or cond == f"n={n}":
            pick = tile
    return pick


def _pass_times(shapes: dict[str, Shape], stats: Stats, labels: list[str]) -> list[PassTime]:
    """Each tile's time for one pass over the labels, fastest first. Only tiles that ran every label.
    Time per shape = ops / GOPs (ns), so the ranking is the same as by total FLOPs over total time."""
    tiles = set(stats[labels[0]]).intersection(*(stats[label] for label in labels[1:]))
    ops = {label: 2.0 * shapes[label][0] * shapes[label][1] * shapes[label][2] for label in labels}
    rows: list[PassTime] = []
    for tile in tiles:
        ns_by_mean = sum(ops[label] / _mean(stats[label][tile]) for label in labels)
        ns_by_median = sum(ops[label] / _median(stats[label][tile]) for label in labels)
        rows.append(PassTime(tile, ns_by_mean / 1e6, sum(ops.values()) / ns_by_mean, ns_by_median / 1e6))
    return sorted(rows, key=lambda r: r.ms_by_mean)


def _per_shape_table(shapes: dict[str, Shape], stats: Stats, compare: list[str]) -> tuple[list[str], list[str], list[str]]:
    """The per-shape table, and the labels seen in fewer than 3 launches and with uneven launch counts."""
    out = ["## Best tile per shape (whole call, packing included)", "",
           "| shape | m | n | k | launches | best by mean time | GOPs by mean time (avg, lo-hi) | best by median rep | "
           "GOPs by median rep (avg, lo-hi) | compare tile | its GOPs by mean time |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    few: list[str] = []
    uneven: list[str] = []
    for label, per_tile in stats.items():
        m, n, k = shapes[label]
        counts = {len(sts) for sts in per_tile.values()}
        if min(counts) < MIN_LAUNCHES:
            few.append(label)
        if len(counts) > 1:
            uneven.append(label)
        best = max(per_tile, key=lambda tile: _mean(per_tile[tile]))
        best_med = max(per_tile, key=lambda tile: _median(per_tile[tile]))
        ct = _compare_tile(compare, n)
        ct_gops = f"{_mean(per_tile[ct]):.2f}" if ct in per_tile else ("not run" if ct else "")
        out.append(f"| {label} | {m} | {n} | {k} | {min(counts)} | {best} | "
                   f"{_spread([s.gops_mean_time for s in per_tile[best]])} | {best_med} | "
                   f"{_spread([s.gops_median for s in per_tile[best_med]])} | {ct or ''} | {ct_gops} |")
    return out, few, uneven


def _dataset_table(shapes: dict[str, Shape], stats: Stats) -> list[str]:
    labels = list(stats)
    out = [f"## Best single tile over the dataset ({len(labels)} shapes, each counted once)", "",
           "| tile | ms per pass (by mean time) | GOPs over the pass | ms per pass (by median rep) |", "|---|---|---|---|"]
    return out + [f"| {r.tile} | {r.ms_by_mean:.4f} | {r.gops:.2f} | {r.ms_by_median:.4f} |"
                  for r in _pass_times(shapes, stats, labels)]


def _per_n_table(shapes: dict[str, Shape], stats: Stats) -> list[str]:
    out = ["## Best tile per n (least total time over that n's shapes)", "",
           "| n | shapes | best tile | ms per pass | GOPs |", "|---|---|---|---|---|"]
    for n in sorted({shapes[label][1] for label in stats}):
        group = [label for label in stats if shapes[label][1] == n]
        ranked = _pass_times(shapes, stats, group)
        if not ranked:
            out.append(f"| {n} | {len(group)} | no tile ran every shape | | |")
        else:
            out.append(f"| {n} | {len(group)} | {ranked[0].tile} | {ranked[0].ms_by_mean:.4f} | {ranked[0].gops:.2f} |")
    return out


def _grid(shapes: dict[str, Shape], stats: Stats) -> tuple[list[str], list[str], list[list[float | None]]]:
    """Shape labels sorted by n, m, k; tiles sorted by MR, NR; GOPs by mean time per (shape, tile), None if not run."""
    labels = sorted(stats, key=lambda label: (shapes[label][1], shapes[label][0], shapes[label][2], label))
    tiles = sorted({t for per_tile in stats.values() for t in per_tile}, key=lambda t: tuple(map(int, t.split("x"))))
    return labels, tiles, [[_mean(stats[label][t]) if t in stats[label] else None for t in tiles] for label in labels]


def _grid_table(shapes: dict[str, Shape], stats: Stats) -> list[str]:
    labels, tiles, gops = _grid(shapes, stats)
    out = ["## GOPs grid (by mean time, whole call; best per shape in bold)", "",
           "| shape | m | n | k | " + " | ".join(tiles) + " |", "|---|---|---|---|" + "---|" * len(tiles)]
    for label, row in zip(labels, gops):
        best = max(g for g in row if g is not None)
        cells = ["" if g is None else f"**{g:.2f}**" if g == best else f"{g:.2f}" for g in row]
        m, n, k = shapes[label]
        out.append(f"| {label} | {m} | {n} | {k} | " + " | ".join(cells) + " |")
    return out


def write_heatmap(shapes: dict[str, Shape], stats: Stats, png: Path) -> None:
    """The grid as a heatmap: colour = GOPs by mean time, each cell labelled, the best tile per shape boxed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    labels, tiles, gops = _grid(shapes, stats)
    data = [[float("nan") if g is None else g for g in row] for row in gops]
    fig, ax = plt.subplots(figsize=(1.0 + 0.6 * len(tiles), 1.5 + 0.28 * len(labels)))
    top = max(g for row in gops for g in row if g is not None)
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=top)
    ax.set_xticks(range(len(tiles)), tiles, rotation=90, fontsize=8)
    ax.set_yticks(range(len(labels)), [f"{lb} {shapes[lb][0]}x{shapes[lb][1]}x{shapes[lb][2]}" for lb in labels],
                  fontsize=7)
    ax.set_xlabel("tile (MR x NR)")
    ax.set_ylabel("shape (m x n x k), sorted by n")
    for i, row in enumerate(gops):
        best = max(g for g in row if g is not None)
        for j, g in enumerate(row):
            if g is None:
                continue
            ax.text(j, i, f"{g:.1f}", ha="center", va="center", fontsize=6,
                    color="black" if g > 0.6 * top else "white")
            if g == best:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=1.5))
    fig.colorbar(im, ax=ax, label="GOPs by mean time (whole call)")
    ax.set_title("Tile Search: GOPs per shape and tile (red box = best)")
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)


def _listed(labels: list[str]) -> str:
    return ", ".join(labels[:8]) + (" ..." if len(labels) > 8 else "")


def report(targets: list[Path], compare: list[str]) -> str:
    shapes, stats = collect(targets)
    notes = load_notes(targets)
    per_shape, few, uneven = _per_shape_table(shapes, stats, compare)
    out = ["# Tile Search", "", *conditions_lines(targets), *notes, "",
           *per_shape, "", *_dataset_table(shapes, stats), "", *_per_n_table(shapes, stats), "",
           *_grid_table(shapes, stats)]
    if few:
        out.append(f"\nWARNING: {len(few)} shape(s) seen in fewer than {MIN_LAUNCHES} launches: not a result under "
                   f"issue #39's house rules: {_listed(few)}")
    if uneven:
        out.append(f"\nWARNING: {len(uneven)} shape(s) where the tiles ran a different number of launches (a tile "
                   f"missing from a run folder): their tiles are not compared like for like: {_listed(uneven)}")
    if notes:
        out.append("\n" + red("🔴 RED NOTE: some run folders above were made under another user's load."))
    return "\n".join(out)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="compile_macro_bench.sh run folders, one per launch")
    ap.add_argument("--compare", action="append", default=[], help="TILE, or n=N:TILE for shapes with that n")
    ap.add_argument("--heatmap", type=Path, help="also draw the GOPs grid as a heatmap PNG")
    args = ap.parse_args(argv)
    print(report(args.runs, args.compare))
    if args.heatmap:
        write_heatmap(*collect(args.runs), args.heatmap)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
