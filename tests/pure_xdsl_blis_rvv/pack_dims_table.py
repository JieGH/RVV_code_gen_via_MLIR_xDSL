#!/usr/bin/env python3
"""
Issue #48, step 1: what each vector packer does per MR and NR, written down before any run.

The shipped AXPY vector packers (pack_builder.py, the `_if` pair the driver calls) are:

  pack_RB_v_if (A, column-major source):  for each MR-row panel, for each k, for each chunk of 8 rows:
      vle32 (contiguous source, one column of A) + vse32 into Ac[panel * kc + k * MR + chunk * 8].
  pack_CB_v_if (B, column-major source):  for each NR-column panel, for each k, for each chunk of 8 columns:
      vlse32 (strided source, stride ldb * 4 bytes: one element from each of up to 8 columns of B)
      + vse32 into Bc[panel * kc + k * NR + chunk * 8].

A chunk's vector length is min(avl, chunk end) - chunk start, clamped at 0, so a panel narrower than a
chunk still issues the chunk's ops with vl = 0. Everything here is worked out for one block (mc x nc x kc),
the way the driver's loop nest packs it; `pack_plan` is checked against the real builders in
test_pack_dims_table.py.

Usage: pack_dims_table.py [--config experiments/issue48/config_issue48_sweep.json] [--m 64 --n 384 --k 384]
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
from config import BLISProjectConfig  # noqa: E402

VLEN = 8  # fp32 lanes of one vector register (VLEN 256)
ELEM = 4  # bytes per fp32
LINE = 64  # cache line, bytes
VREGS = 32
ARCH = "jbpi2"  # both boards use the jbpi2 cache model in the configs


def chunk_vls(width: int, panel: int, vlen: int = VLEN) -> tuple[int, ...]:
    """The vector length each chunk of a panel gets when `width` of its `panel` slots are valid
    (width == panel for a full panel). Mirrors pack_builder.get_safe_chunk_avl."""
    return tuple(
        max(min(width, (v + 1) * vlen), v * vlen) - v * vlen for v in range(math.ceil(panel / vlen))
    )


def _stores_crossing(row_bytes: int, chunks: tuple[int, ...], rows: int, vlen: int = VLEN) -> float:
    """Share of the vector stores of a panel that straddle two 64-byte lines, for a panel whose base is on a
    line and whose rows are `row_bytes` apart. Stores with vl = 0 write nothing and count for nothing."""
    crossing = total = 0
    for k in range(rows):
        for v, vl in enumerate(chunks):
            if vl == 0:
                continue
            start = k * row_bytes + v * vlen * ELEM
            total += 1
            crossing += (start % LINE) + vl * ELEM > LINE
    return crossing / total if total else 0.0


@dataclass
class PackPlan:
    mr: int
    nr: int
    mc: int
    nc: int
    kc: int
    # Ac, the A packer
    a_chunks: tuple[int, ...]
    a_row_bytes: int
    a_panels: int
    a_fringe_rows: int
    a_fringe_chunks: tuple[int, ...]
    a_panel_bytes: int
    a_panel_line_aligned: bool
    a_stores_crossing: float
    # Bc, the B packer
    b_chunks: tuple[int, ...]
    b_row_bytes: int
    b_panels: int
    b_full_panels: int
    b_fringe_cols: int
    b_fringe_chunks: tuple[int, ...]
    b_panel_bytes: int
    b_panel_line_aligned: bool
    b_stores_crossing: float
    # the microkernel and the tiles of the block
    live_vregs: int
    tiles: int
    fringe_tiles: int

    @property
    def a_vectors_per_k(self) -> int:
        return len(self.a_chunks)

    @property
    def b_vectors_per_k(self) -> int:
        return len(self.b_chunks)

    @property
    def over_register_budget(self) -> bool:
        return self.live_vregs > VREGS


def pack_plan(mr: int, nr: int, mc: int, nc: int, kc: int) -> PackPlan:
    a_panels, b_panels = math.ceil(mc / mr), math.ceil(nc / nr)
    a_fringe, b_fringe = mc % mr, nc % nr
    a_chunks, b_chunks = chunk_vls(mr, mr), chunk_vls(nr, nr)
    a_bytes, b_bytes = mr * kc * ELEM, nr * kc * ELEM
    # Tile registers of the microkernel: MR/8 accumulator columns of NR, plus the A vectors (#32's budget).
    live = math.ceil(mr / VLEN) * (1 + nr)
    # In the pinned build every tile with cur_mr < MR or cur_nr < NR goes through the temp-buffer path.
    tiles = a_panels * b_panels
    fringe_tiles = tiles - (mc // mr) * (nc // nr)
    return PackPlan(
        mr, nr, mc, nc, kc,
        a_chunks, mr * ELEM, a_panels, a_fringe, chunk_vls(a_fringe, mr) if a_fringe else (),
        a_bytes, a_bytes % LINE == 0, _stores_crossing(mr * ELEM, a_chunks, kc),
        b_chunks, nr * ELEM, b_panels, nc // nr, b_fringe, chunk_vls(b_fringe, nr) if b_fringe else (),
        b_bytes, b_bytes % LINE == 0, _stores_crossing(nr * ELEM, b_chunks, kc),
        live, tiles, fringe_tiles,
    )


def sweep_tiles(config: Path) -> list[tuple[int, int]]:
    """The (MR, NR) of every row of a sweep config, in file order."""
    return [row.mr_nr_family[0] for row in BLISProjectConfig.from_json(str(config))]


def _vls(chunks: tuple[int, ...]) -> str:
    return "+".join(str(c) for c in chunks) if chunks else "-"


def format_table(tiles: list[tuple[int, int]], m: int, n: int, k: int) -> str:
    """One markdown row per tile: the blocking the cache model picks for m x n x k, then the A packer, the B
    packer and the tile counts of one mc x nc block."""
    out = [
        "| tile | mc | nc | kc | K blocks | A: vectors per k (vl) | A panels (fringe rows, vl) | Ac row B | A stores over a line "
        "| B: strided loads per k (vl) | B panels (fringe cols, vl) | Bc row B | B stores over a line "
        "| Bc panel B, on a line | live vregs | fringe tiles / tiles |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for mr, nr in tiles:
        mc, nc, kc = get_optim_mc_nc_kc(ARCH, ELEM, mr, nr, m, n, k)
        p = pack_plan(mr, nr, mc, nc, kc)
        a_fringe = f"{p.a_fringe_rows}, {_vls(p.a_fringe_chunks)}" if p.a_fringe_rows else "none"
        b_fringe = f"{p.b_fringe_cols}, {_vls(p.b_fringe_chunks)}" if p.b_fringe_cols else "none"
        regs = f"{p.live_vregs} (over)" if p.over_register_budget else str(p.live_vregs)
        out.append(
            f"| {mr}x{nr} | {mc} | {nc} | {kc} | {math.ceil(k / kc)} "
            f"| {p.a_vectors_per_k} ({_vls(p.a_chunks)}) | {p.a_panels} ({a_fringe}) | {p.a_row_bytes} "
            f"| {100 * p.a_stores_crossing:.0f}% "
            f"| {p.b_vectors_per_k} ({_vls(p.b_chunks)}) | {p.b_panels} ({b_fringe}) | {p.b_row_bytes} "
            f"| {100 * p.b_stores_crossing:.0f}% "
            f"| {p.b_panel_bytes} {'yes' if p.b_panel_line_aligned else 'NO'} | {regs} "
            f"| {p.fringe_tiles} / {p.tiles} |"
        )
    return "\n".join(out)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "experiments/issue48/config_issue48_sweep.json"))
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--n", type=int, default=384)
    ap.add_argument("--k", type=int, default=384)
    args = ap.parse_args(argv)
    print(format_table(sweep_tiles(Path(args.config)), args.m, args.n, args.k))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
