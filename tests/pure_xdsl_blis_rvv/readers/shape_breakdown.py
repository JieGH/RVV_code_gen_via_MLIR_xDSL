"""Shape-side numbers for one GEMM case, no board time needed (issue #34 §3.3).

B4 (4096x384x1024) runs slower than every square case on the identical 32x7 tile. The
tile is not the cause (issue #31), so this looks at what the *shape* changes: the
blocking the driver picks, how often each packed buffer is reused, how much packing
each flop pays for, and where the columns of one C tile land in the L1 cache.

The driver sets `lda = ldc = m` (`templates/driver_benchmark.cpp`), so the C tile's
columns sit `m` floats apart. When that stride is a multiple of the L1's set period
(sets x line size), every column maps to the same few sets.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the project folder, for cache_config

from cache_config import ARCH_CONFIGS, get_optim_mc_nc_kc  # noqa: E402

DATA_SIZE = 4  # fp32
ARCH = "jbpi2"


def c_tile_lines_per_set(ld: int, mr: int, nr: int, arch: str = ARCH) -> int:
    """Most L1 lines one mr x nr column-major C tile puts into a single set, with its
    columns `ld` elements apart and the tile starting on a line boundary. More than the
    L1's ways means the tile cannot stay in L1 as a whole."""
    cfg = ARCH_CONFIGS[arch]
    line, sets = cfg["CL1"], cfg["NL1"]
    per_set: Counter[int] = Counter()
    for j in range(nr):
        start = j * ld * DATA_SIZE
        end = start + mr * DATA_SIZE - 1
        for ln in range(start // line, end // line + 1):
            per_set[ln % sets] += 1
    return max(per_set.values())


@dataclass(frozen=True)
class ShapeBreakdown:
    mc: int
    nc: int
    kc: int
    a_packs: int
    """Times the whole of A is packed: once per jc block, ceil(N/nc)."""
    ac_reuse: int
    """Tiles that use one packed Ac micro-panel (the jr loop): ceil(nc/nr)."""
    bc_reuse: int
    """Tiles that use one packed Bc micro-panel (the ir loop): ceil(mc/mr)."""
    pack_per_kflop: float
    """Elements the packers copy per 1000 flops of the GEMM."""
    c_lines_per_set: int
    """c_tile_lines_per_set for this shape's ldc (= M)."""
    l1_ways: int


def shape_breakdown(m: int, n: int, k: int, mr: int, nr: int, arch: str = ARCH) -> ShapeBreakdown:
    """The shape-side numbers for one case, with the blocking the driver picks
    (`get_optim_mc_nc_kc` clamped to the problem, no nc override)."""
    mc, nc, kc = get_optim_mc_nc_kc(arch, DATA_SIZE, mr, nr, mmax=m, nmax=n, kmax=k)
    a_packs = -(-n // nc)
    packed = a_packs * m * k + k * n
    return ShapeBreakdown(
        mc=mc,
        nc=nc,
        kc=kc,
        a_packs=a_packs,
        ac_reuse=-(-nc // nr),
        bc_reuse=-(-mc // mr),
        pack_per_kflop=1000.0 * packed / (2.0 * m * n * k),
        c_lines_per_set=c_tile_lines_per_set(m, mr, nr, arch),
        l1_ways=ARCH_CONFIGS[arch]["WL1"],
    )


def _read_dataset(path: str) -> list[tuple[str, int, int, int]]:
    """label, M, N, K per row, the four columns the driver parses."""
    rows: list[tuple[str, int, int, int]] = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4 or parts[0].startswith("#"):
                continue
            rows.append((parts[0], int(parts[1]), int(parts[2]), int(parts[3])))
    return rows


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Shape-side numbers per dataset row (issue #34).")
    ap.add_argument("dataset", help="a datasets/*.dat file")
    ap.add_argument("--mr", type=int, required=True)
    ap.add_argument("--nr", type=int, required=True)
    args = ap.parse_args()

    print(
        "| case | M×N×K | mc/nc/kc | A packed | Ac reuse | Bc reuse "
        "| packed elems / kflop | C-tile lines per L1 set (ways) |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for label, m, n, k in _read_dataset(args.dataset):
        b = shape_breakdown(m, n, k, args.mr, args.nr)
        over = " **over**" if b.c_lines_per_set > b.l1_ways else ""
        print(
            f"| {label} | {m}×{n}×{k} | {b.mc}/{b.nc}/{b.kc} | {b.a_packs}× "
            f"| {b.ac_reuse} | {b.bc_reuse} | {b.pack_per_kflop:.3f} "
            f"| {b.c_lines_per_set} ({b.l1_ways}){over} |"
        )


if __name__ == "__main__":
    main()
