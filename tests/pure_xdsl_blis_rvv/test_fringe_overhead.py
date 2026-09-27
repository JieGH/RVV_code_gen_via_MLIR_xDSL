"""
Issue #15: our dispatcher runs the full MRxNR kernel on every fringe tile, where the
reference calls a right-sized one. `fringe_extra_macs` puts a number on that: the
fraction of extra multiply-adds our loop nest does on a given shape.

The jc/ic loops split M and N into blocks of at most nc/mc, and every block pads its
own last tile, so the answer depends on the blocking and not just on M, N, MR, NR.
K is never padded (`kc_pad == cur_kc` in gemm_dispatch_xdsl.cpp).
"""

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "ref_baseline"))

from cache_config import get_optim_mc_nc_kc  # noqa: E402
from config import BLISProjectConfig  # noqa: E402
import fringe  # noqa: E402
from fringe import (  # noqa: E402
    fringe_extra_macs,
    fringe_for_case,
    load_case_shapes,
    nc_override_for,
    our_fringe,
)


def _walk_the_loop_nest(m, n, mr, nr, mc, nc):
    """Tile-by-tile count, mirroring the jc/ic/jr/ir loops in the dispatcher."""
    done = padded = 0
    for jc in range(0, n, nc):
        cur_nc = min(n - jc, nc)
        for ic in range(0, m, mc):
            cur_mc = min(m - ic, mc)
            for jr in range(0, cur_nc, nr):
                for ir in range(0, cur_mc, mr):
                    done += min(cur_mc - ir, mr) * min(cur_nc - jr, nr)
                    padded += mr * nr
    return padded / done - 1.0


def test_a_shape_that_tiles_exactly_has_no_extra_work():
    assert fringe_extra_macs(m=64, n=48, mr=16, nr=12, mc=64, nc=48) == 0.0


def test_one_short_column_tile_is_counted_in_full():
    # N=7 with NR=6 is a 6-wide tile and a 1-wide tile that runs as 6 wide: 12/7.
    assert fringe_extra_macs(m=32, n=7, mr=32, nr=6, mc=32, nc=7) == pytest.approx(12 / 7 - 1)


def test_each_block_pads_its_own_last_tile():
    # nc=5, NR=4: two blocks of 5 columns, each padded to 8, so 16 columns of work
    # for 10 real ones. Ignoring the block edge would say 12/10.
    assert fringe_extra_macs(m=8, n=10, mr=8, nr=4, mc=8, nc=5) == pytest.approx(16 / 10 - 1)
    assert fringe_extra_macs(m=8, n=10, mr=8, nr=4, mc=8, nc=10) == pytest.approx(12 / 10 - 1)


def test_m_and_n_padding_multiply():
    # 8 rows padded to 16 (x2) and 5 columns padded to 8 (x1.6) is 3.2x the work.
    assert fringe_extra_macs(m=8, n=5, mr=16, nr=8, mc=8, nc=5) == pytest.approx(2 * 1.6 - 1)


@pytest.mark.parametrize(
    "m,n,mr,nr,mc,nc",
    [
        (64, 384, 16, 15, 64, 384),
        (3000, 3000, 32, 7, 1792, 512),
        (1000, 1000, 32, 7, 1000, 524),
        (100, 37, 32, 6, 48, 10),
    ],
)
def test_it_agrees_with_walking_the_loop_nest_tile_by_tile(m, n, mr, nr, mc, nc):
    assert fringe_extra_macs(m, n, mr, nr, mc, nc) == pytest.approx(
        _walk_the_loop_nest(m, n, mr, nr, mc, nc)
    )


def test_the_case_table_yields_one_figure_per_case_using_the_pinned_kernel_blocking():
    shapes = load_case_shapes()

    assert list(shapes) == ["B1", "B2", "B3", "B4", "B5", "S1", "S2", "S3", "S4", "S5"]
    # B2 shares B3's pinning and dataset row but carries no nc override.
    mc, nc, _kc = get_optim_mc_nc_kc("jbpi2", 4, 16, 15, mmax=384, nmax=384, kmax=64)
    assert shapes["B2"] == pytest.approx(fringe_extra_macs(384, 384, 16, 15, mc, nc))


def _row(mr_nr, overrides):
    return BLISProjectConfig(
        datatype="fp32", method="AXPY_FP32_Pack", mr_nr_family=mr_nr, vlen=256,
        arch="jbpi2", nc_overrides=overrides,
    )


def test_config_json_ships_b3_at_the_heuristics_nc():
    # Issue #35 dropped B3's nc=128 override (issue #1): B3 now runs at the heuristic's
    # nc=384, like the reference, and pads once, to 26 tiles of 15: +1.56%.
    assert nc_override_for("B3", 16, 15) is None
    assert load_case_shapes()["B3"] == pytest.approx(390 / 384 - 1)
    # nc=128 would pad each of 3 blocks to 135: 405 columns of work for 384, +5.47%.
    assert fringe_for_case(64, 384, 384, 16, 15, nc_override=128) == pytest.approx(405 / 384 - 1)


def test_an_nc_override_is_clamped_to_n_like_the_driver_does():
    assert fringe_for_case(64, 100, 384, 16, 15, nc_override=4000) == pytest.approx(
        fringe_for_case(64, 100, 384, 16, 15)
    )


def test_an_override_names_one_label_exactly_and_only_for_the_same_pinning():
    # Issue #16: the driver matches the label exactly, so a future B30 (or a bare "B")
    # does not inherit B3's override.
    rows = [
        _row([(16, 15)], {"B": 64, "B3": 128}),
        _row([(32, 6)], {"B3": 999}),
    ]

    assert nc_override_for("B3", 16, 15, rows) == 128
    assert nc_override_for("B", 16, 15, rows) == 64
    assert nc_override_for("B30", 16, 15, rows) is None
    assert nc_override_for("B2", 16, 15, rows) is None
    assert nc_override_for("S1", 16, 15, rows) is None
    assert nc_override_for("B3", 32, 7, rows) is None


def test_our_fringe_can_ignore_the_overrides_like_the_driver_flag_does(monkeypatch):
    # With a B3 override of nc=128 in the config, --no-nc-overrides runs B3 at the
    # heuristic's nc=384: one padded block, not three.
    monkeypatch.setattr(fringe, "nc_override_for", lambda case, mr, nr, configs=None: 128)
    assert our_fringe("B3", 64, 384, 384, 16, 15) == pytest.approx(405 / 384 - 1)
    assert our_fringe("B3", 64, 384, 384, 16, 15, use_overrides=False) == pytest.approx(
        390 / 384 - 1
    )


def test_the_readme_table_quotes_the_computed_figures():
    """The README states these numbers; a change to blocking or cases.tsv must not
    leave them stale."""
    lines = [
        l.strip() for l in (HERE / "ref_baseline" / "README.md").read_text().splitlines()
    ]
    header = next(l for l in lines if l.startswith("| case | B1"))
    row = next(l for l in lines if l.startswith("| extra MACs"))
    cases = [c.strip() for c in header.strip("|").split("|")][1:]
    quoted = [c.strip() for c in row.strip("|").split("|")][1:]

    shapes = load_case_shapes()

    assert cases == list(shapes)
    assert quoted == [f"{100 * v:.2f}%" for v in shapes.values()]
