"""
Issue #48 step 1: `pack_dims_table.py` writes down what each vector packer does per MR and NR.

The facts it prints (vectors per k-step, the partial vector, row bytes, panel alignment, fringe panels and
tiles, live registers) are worked out by `pack_plan`. They are checked here against the real builders in
`pack_builder.py`, so the table cannot drift from the code it describes: the vector memory ops the builder
emits per k-step must be the ones `pack_plan` counts, and the vector length each chunk gets must be the one
the builder's `min`/`max` arithmetic gives for a full panel.
"""

import sys
from pathlib import Path

import pytest
from xdsl.dialects.builtin import Float32Type, IndexType, MemRefType
from xdsl.dialects.func import FuncOp
from xdsl.dialects.scf import ForOp
from xdsl.ir import Operation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pack_builder import build_pack_CB_v_if, build_pack_RB_v_if  # noqa: E402  # pyright: ignore[reportUnknownVariableType]
from pack_dims_table import (  # noqa: E402
    chunk_vls,
    format_table,
    pack_plan,
    sweep_tiles,
)
from xdsltemplate.dialects.rvv import (  # noqa: E402
    SetvlOp,
    vle32_v_f32m1Op,
    vlse32_v_f32m1Op,
    vse32_v_f32m1Op,
)

HERE = Path(__file__).resolve().parents[1]
IDX = IndexType()
MEM = MemRefType(Float32Type(), [-1])


def inner_loop_ops(fn: FuncOp) -> list[Operation]:
    """The ops of the innermost scf.for of a packer: what runs once per k-step."""
    loops = [op for op in fn.walk() if isinstance(op, ForOp)]
    return list(loops[-1].body.block.ops)


def count(ops: list[Operation], kind: type[Operation]) -> int:
    return sum(isinstance(op, kind) for op in ops)


@pytest.mark.parametrize("nr", [1, 6, 7, 8, 9, 12, 14, 15, 16, 24, 31])
def test_b_packer_vector_ops_per_k_step_match_the_builder(nr: int):
    ops = inner_loop_ops(build_pack_CB_v_if(IDX, MEM, nr, 4, "fp32"))
    plan = pack_plan(mr=16, nr=nr, mc=64, nc=384, kc=128)

    assert count(ops, vlse32_v_f32m1Op) == plan.b_vectors_per_k  # strided loads: one per chunk
    assert count(ops, vse32_v_f32m1Op) == plan.b_vectors_per_k
    assert count(ops, vle32_v_f32m1Op) == 0
    assert count(ops, SetvlOp) == plan.b_vectors_per_k


@pytest.mark.parametrize("mr", [1, 4, 8, 12, 16, 24, 32])
def test_a_packer_vector_ops_per_k_step_match_the_builder(mr: int):
    ops = inner_loop_ops(build_pack_RB_v_if(IDX, MEM, mr, 4, "fp32"))
    plan = pack_plan(mr=mr, nr=6, mc=64, nc=384, kc=128)

    assert count(ops, vle32_v_f32m1Op) == plan.a_vectors_per_k  # contiguous loads: one per chunk
    assert count(ops, vse32_v_f32m1Op) == plan.a_vectors_per_k
    assert count(ops, vlse32_v_f32m1Op) == 0
    assert count(ops, SetvlOp) == plan.a_vectors_per_k


@pytest.mark.parametrize(
    "width, expected",
    [(15, (8, 7)), (14, (8, 6)), (16, (8, 8)), (8, (8,)), (6, (6,)), (31, (8, 8, 8, 7)), (9, (8, 1)), (24, (8, 8, 8))],
)
def test_chunk_vector_lengths_of_a_panel_of_that_width(width: int, expected: tuple[int, ...]):
    assert chunk_vls(width, width) == expected


def test_a_fringe_panel_narrower_than_a_chunk_still_issues_the_chunk_with_length_zero():
    # the builder clamps with max(min(avl, end), start) - start, so the vector ops run with vl = 0, they are not skipped
    assert chunk_vls(5, 16, 8) == (5, 0)
    assert chunk_vls(1, 15, 8) == (1, 0)


def test_the_shipped_b3_tile_16x15():
    plan = pack_plan(mr=16, nr=15, mc=64, nc=384, kc=128)

    assert plan.b_chunks == (8, 7)
    assert plan.b_row_bytes == 60
    assert plan.b_panels == 26 and plan.b_full_panels == 25 and plan.b_fringe_cols == 9
    assert plan.b_fringe_chunks == (8, 1)
    assert plan.b_panel_bytes == 15 * 128 * 4 and plan.b_panel_line_aligned  # 7680 B = 120 lines
    assert plan.a_chunks == (8, 8) and plan.a_row_bytes == 64 and plan.a_panels == 4 and plan.a_fringe_rows == 0
    assert plan.live_vregs == 32 and not plan.over_register_budget
    assert plan.tiles == 4 * 26 and plan.fringe_tiles == 4  # the 4 tiles of the 9-column panel take the temp-buffer path


def test_vector_stores_that_cross_a_cache_line_depend_on_the_row_width():
    # Rows of Bc are NR*4 bytes apart and a vector store is up to 32 B: with a 64 B row every store sits in one line;
    # a 60 B row (NR 15) walks through the line, so 13 of every 32 stores (two per row, 16 rows per period) straddle two.
    assert pack_plan(mr=16, nr=16, mc=64, nc=384, kc=128).b_stores_crossing == 0.0
    assert pack_plan(mr=16, nr=15, mc=64, nc=384, kc=128).b_stores_crossing == 13 / 32
    assert pack_plan(mr=16, nr=14, mc=64, nc=384, kc=128).b_stores_crossing == 5 / 16
    assert pack_plan(mr=16, nr=15, mc=64, nc=384, kc=128).a_stores_crossing == 0.0  # Ac rows of 16 floats = 64 B


def test_16x16_is_over_the_register_budget_and_16x14_is_not():
    assert pack_plan(mr=16, nr=16, mc=64, nc=384, kc=128).live_vregs == 34
    assert pack_plan(mr=16, nr=16, mc=64, nc=384, kc=128).over_register_budget
    assert pack_plan(mr=16, nr=14, mc=64, nc=384, kc=128).live_vregs == 30
    assert pack_plan(mr=32, nr=7, mc=64, nc=384, kc=128).live_vregs == 32
    assert pack_plan(mr=8, nr=31, mc=64, nc=384, kc=132).live_vregs == 32


def test_a_tile_that_divides_n_has_no_fringe_panel_and_one_that_does_not_has():
    for nr in (6, 8, 12, 16):
        assert pack_plan(mr=16, nr=nr, mc=64, nc=384, kc=128).b_fringe_cols == 0, nr
    for nr, rest in ((7, 6), (14, 6), (15, 9)):
        assert pack_plan(mr=16, nr=nr, mc=64, nc=384, kc=128).b_fringe_cols == rest, nr


def test_mr_24_leaves_a_fringe_of_rows_on_64_row_blocks():
    plan = pack_plan(mr=24, nr=6, mc=64, nc=384, kc=170)

    assert plan.a_panels == 3 and plan.a_fringe_rows == 16
    assert plan.a_fringe_chunks == (8, 8, 0)


def test_a_panel_whose_size_is_not_a_whole_number_of_lines_is_reported_unaligned():
    # kc = 170 (the cache model's choice for MR 24): 6 * 170 * 4 = 4080 B = 63.75 lines
    assert not pack_plan(mr=24, nr=6, mc=64, nc=384, kc=170).b_panel_line_aligned
    assert pack_plan(mr=16, nr=15, mc=64, nc=384, kc=128).b_panel_line_aligned


def test_the_table_lists_every_tile_of_the_sweep_config_with_the_blocking_the_cache_model_picks():
    tiles = sweep_tiles(HERE / "experiments/issue48/config_issue48_sweep.json")
    text = format_table(tiles, m=64, n=384, k=384)

    assert (16, 15) in tiles and (16, 14) in tiles and (16, 16) in tiles
    for mr, nr in tiles:
        assert f"| {mr}x{nr} |" in text
    assert "| 16x15 | 64 | 384 | 128 |" in text  # B3's mc, nc, kc at the shipped tile (the reps logs carry the same)
    assert "over" in text  # 16x16 is flagged
