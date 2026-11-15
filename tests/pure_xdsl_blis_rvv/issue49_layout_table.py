#!/usr/bin/env python3
"""
Issue #49 step 1: the desk table, no board. For every #48 sweep tile, at K=384 (B3) and at
K=256 (B3_K256): the B stream step in bytes (ldb * 4, ldb = k today), how many distinct byte
offsets mod 4096 the NR stream heads take, how many 4 KiB pages one k-step of one panel
touches, kc, the number of K blocks and the last block's size, Bc panel bytes -- then the
measured slow-rep share of both boards (vector packer) from the #48 report
(logs/issue48_2026-09-30_report.md, "jbpi, B3"/"jbpi, B3_K256"/"jbpi2, B3"/"jbpi2, B3_K256").

Looks for one quantity that puts every steady case (slow-rep share == 0 on both boards and
both K) on one side and every slow case on the other, in both K columns -- the question #49
asks before any board time is spent on the decisive ldb-padding test.

Usage: python3 issue49_layout_table.py [--config config_issue48_sweep.json]
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cache_config import get_optim_mc_nc_kc  # noqa: E402
from pack_dims_table import ARCH, ELEM, sweep_tiles  # noqa: E402

PAGE = 4096  # bytes, the page size candidate for H2
M, N = 64, 384  # B3 / B3_K256's own m, n (pack_dims_table.py's defaults)

# Measured slow-rep share (%), vector packer, from logs/issue48_2026-09-30_report.md.
# (board, shape) -> {tile: slow_rep_pct}
SLOW_REP_PCT: dict[tuple[str, str], dict[tuple[int, int], int]] = {
    ("jbpi", "B3"): {
        (4, 4): 0, (4, 8): 82, (4, 16): 76, (8, 4): 0, (8, 8): 73, (8, 14): 93, (8, 15): 90,
        (8, 16): 81, (8, 24): 53, (8, 31): 84, (12, 8): 82, (16, 4): 0, (16, 6): 0, (16, 7): 79,
        (16, 8): 89, (16, 12): 88, (16, 14): 81, (16, 15): 89, (16, 16): 25, (24, 6): 0,
        (24, 8): 87, (24, 9): 89, (32, 4): 0, (32, 6): 0, (32, 7): 87,
    },
    ("jbpi", "B3_K256"): {
        (4, 4): 0, (4, 8): 93, (4, 16): 76, (8, 4): 0, (8, 8): 95, (8, 14): 94, (8, 15): 88,
        (8, 16): 96, (8, 24): 92, (8, 31): 92, (12, 8): 0, (16, 4): 0, (16, 6): 48, (16, 7): 79,
        (16, 8): 0, (16, 12): 95, (16, 14): 93, (16, 15): 94, (16, 16): 77, (24, 6): 81,
        (24, 8): 0, (24, 9): 97, (32, 4): 0, (32, 6): 19, (32, 7): 59,
    },
    ("jbpi2", "B3"): {
        (4, 4): 0, (4, 8): 78, (4, 16): 81, (8, 4): 0, (8, 8): 88, (8, 14): 89, (8, 15): 92,
        (8, 16): 88, (8, 24): 54, (8, 31): 75, (12, 8): 72, (16, 4): 0, (16, 6): 0, (16, 7): 68,
        (16, 8): 94, (16, 12): 78, (16, 14): 86, (16, 15): 92, (16, 16): 67, (24, 6): 0,
        (24, 8): 89, (24, 9): 83, (32, 4): 0, (32, 6): 0, (32, 7): 83,
    },
    ("jbpi2", "B3_K256"): {
        (4, 4): 0, (4, 8): 89, (4, 16): 87, (8, 4): 0, (8, 8): 89, (8, 14): 95, (8, 15): 94,
        (8, 16): 94, (8, 24): 85, (8, 31): 91, (12, 8): 0, (16, 4): 0, (16, 6): 58, (16, 7): 81,
        (16, 8): 0, (16, 12): 97, (16, 14): 93, (16, 15): 88, (16, 16): 88, (24, 6): 85,
        (24, 8): 0, (24, 9): 93, (32, 4): 0, (32, 6): 30, (32, 7): 62,
    },
}


@dataclass
class LayoutRow:
    mr: int
    nr: int
    k: int
    kc: int
    k_blocks: int
    last_block: int
    step_bytes: int
    distinct_offsets: int
    pages_per_kstep: int
    bc_panel_bytes: int
    mc: int
    a_panels: int
    slow_pct: dict[str, int]  # board -> pct, this (mr,nr) at this k


def distinct_mod(step: int, count: int, modulus: int) -> int:
    """How many distinct values {0, step, 2*step, ..., (count-1)*step} take mod modulus."""
    return len({(i * step) % modulus for i in range(count)})


def pages_touched(step: int, count: int, page: int = PAGE) -> int:
    """How many page-size windows the `count` stream heads 0, step, 2*step, ... fall into,
    counting from the first head's own page (i.e. relative to head 0, not absolute address)."""
    if count == 0:
        return 0
    offs = [i * step for i in range(count)]
    return len({o // page for o in offs})


def build_rows(tiles: list[tuple[int, int]]) -> list[LayoutRow]:
    rows = []
    for mr, nr in tiles:
        row_by_k: dict[int, LayoutRow] = {}
        for k, shape in ((384, "B3"), (256, "B3_K256")):
            mc, nc, kc = get_optim_mc_nc_kc(ARCH, ELEM, mr, nr, M, N, k)
            k_blocks = math.ceil(k / kc)
            last_block = k - (k_blocks - 1) * kc
            step_bytes = k * ELEM  # ldb = k today; the B packer's vlse32 stride
            distinct = distinct_mod(step_bytes, nr, PAGE)
            pages = pages_touched(step_bytes, nr)
            bc_panel_bytes = nr * kc * ELEM
            a_panels = math.ceil(mc / mr)
            slow = {board: SLOW_REP_PCT[(board, shape)][(mr, nr)] for board in ("jbpi", "jbpi2")}
            row_by_k[k] = LayoutRow(mr, nr, k, kc, k_blocks, last_block, step_bytes, distinct,
                                     pages, bc_panel_bytes, mc, a_panels, slow)
        rows.append(row_by_k[384])
        rows.append(row_by_k[256])
    return rows


def steady(row: LayoutRow) -> bool:
    return max(row.slow_pct.values()) == 0


def format_table(rows: list[LayoutRow]) -> str:
    out = [
        "| tile | K | kc | K blocks | last block | step B | distinct offsets mod 4096 | pages/k-step "
        "| Bc panel B | slow% jbpi | slow% jbpi2 | steady? |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| {r.mr}x{r.nr} | {r.k} | {r.kc} | {r.k_blocks} | {r.last_block} | {r.step_bytes} "
            f"| {r.distinct_offsets} | {r.pages_per_kstep} | {r.bc_panel_bytes} "
            f"| {r.slow_pct['jbpi']} | {r.slow_pct['jbpi2']} | {'yes' if steady(r) else 'no'} |"
        )
    return "\n".join(out)


def find_splitting_quantities(rows: list[LayoutRow]) -> list[str]:
    """Which single-valued quantities put every steady row on one side and every slow row on
    the other, across both K columns at once (the question #49 step 1 asks)."""
    candidates = {
        "distinct_offsets": lambda r: r.distinct_offsets,
        "pages_per_kstep": lambda r: r.pages_per_kstep,
        "nr": lambda r: r.nr,
        "mr": lambda r: r.mr,
        "k_blocks": lambda r: r.k_blocks,
        "step_bytes_pow2": lambda r: (r.step_bytes & (r.step_bytes - 1)) == 0,
        "bc_panel_bytes_mod_4096": lambda r: r.bc_panel_bytes % PAGE,
        "mc": lambda r: r.mc,
        "a_panels": lambda r: r.a_panels,
        "nr_and_mc_ge_12": lambda r: (r.nr, r.mc >= 384) if r.nr >= 7 else ("small_nr",),
    }
    hits = []
    for name, f in candidates.items():
        steady_vals = {f(r) for r in rows if steady(r)}
        slow_vals = {f(r) for r in rows if not steady(r)}
        if not (steady_vals & slow_vals):
            hits.append(f"{name}: steady={sorted(steady_vals, key=str)} slow={sorted(slow_vals, key=str)}")
    return hits


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config_issue48_sweep.json"))
    args = ap.parse_args(argv)

    tiles = sweep_tiles(Path(args.config))
    rows = build_rows(tiles)
    print(format_table(rows))
    print()
    hits = find_splitting_quantities(rows)
    if hits:
        print("Splitting quantities found (steady and slow never share a value):")
        for h in hits:
            print(f"  {h}")
    else:
        print("No single quantity among the candidates tried puts every steady row on one side"
              " and every slow row on the other, in both K columns.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
