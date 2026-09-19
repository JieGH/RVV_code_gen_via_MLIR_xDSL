"""How much extra multiply-add work our fringe tiles do compared with the reference.

Our dispatcher (`gemm_dispatch_xdsl.cpp`) runs the full MRxNR kernel on every tile,
including the partial ones at the edge of a block. The reference calls a kernel sized
to the tile (`ukrmatrix[mr][nr][be]`, `gemm_blis.c:119`). Both do the same real work;
ours also multiplies the zero padding.

The jc and ic loops cut N and M into blocks of at most nc and mc, and every block pads
its own last tile, so the figure depends on the blocking, including any nc override in
config.json. K is not padded there.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from cache_config import get_optim_mc_nc_kc  # noqa: E402
from config import BLISProjectConfig  # noqa: E402

DATA_SIZE = 4  # fp32
CONFIG_JSON = HERE.parent / "config.json"


def _padded_extent(total: int, block: int, tile: int) -> int:
    """Length of `total` after cutting it into blocks of `block` and padding each
    block's last tile out to `tile`."""
    full, rest = divmod(total, block)
    per_block = -(-block // tile) * tile
    last = -(-rest // tile) * tile
    return full * per_block + last


def fringe_extra_macs(m: int, n: int, mr: int, nr: int, mc: int, nc: int) -> float:
    """Extra multiply-adds our loop nest does over the real work, as a fraction
    (0.055 means +5.5%)."""
    padded = _padded_extent(m, mc, mr) * _padded_extent(n, nc, nr)
    return padded / (m * n) - 1.0


def nc_override_for(
    case: str, mr: int, nr: int, configs: list[BLISProjectConfig] | None = None
) -> int | None:
    """The nc our binary is forced to for `case`, or None when it keeps the heuristic's.

    config.json rows carry `{"label-prefix": nc}` overrides (issue #1 pins B3 to 128);
    the driver applies the longest matching prefix, and only within the row whose pinned
    MR/NR built that binary."""
    if configs is None:
        configs = BLISProjectConfig.from_json(str(CONFIG_JSON))
    best_len, best_nc = 0, None
    for row in configs:
        if (mr, nr) not in row.mr_nr_family:
            continue
        for prefix, nc in (row.nc_overrides or {}).items():
            if case.startswith(prefix) and len(prefix) > best_len:
                best_len, best_nc = len(prefix), nc
    return best_nc


def fringe_for_case(
    m: int, n: int, k: int, mr: int, nr: int, nc_override: int | None = None
) -> float:
    """fringe_extra_macs using the blocking our own binary picks for this shape and
    pinned MR/NR. `nc_override` replaces the heuristic's nc, clamped to N as the
    driver does."""
    mc, nc, _kc = get_optim_mc_nc_kc("jbpi2", DATA_SIZE, mr, nr, mmax=m, nmax=n, kmax=k)
    if nc_override is not None:
        nc = min(n, nc_override)
    return fringe_extra_macs(m, n, mr, nr, mc, nc)


def our_fringe(case: str, m: int, n: int, k: int, mr: int, nr: int) -> float:
    """fringe_for_case for a named case, with the nc override config.json gives it."""
    return fringe_for_case(m, n, k, mr, nr, nc_override_for(case, mr, nr))


def load_case_shapes(cases_tsv: Path = HERE / "cases.tsv") -> dict[str, float]:
    """Case name -> our_fringe, for every row of cases.tsv."""
    out: dict[str, float] = {}
    for line in cases_tsv.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _numid, m, n, k, mr, nr = line.split()
        out[name] = our_fringe(name, int(m), int(n), int(k), int(mr), int(nr))
    return out
