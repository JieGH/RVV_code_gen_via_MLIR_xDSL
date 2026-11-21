"""
Issue #53: `issue53_packer_compare.py` reads `logs/issue48_2026-09-30_report.md` directly
(no hand-transcribed numbers to drift), so this checks the parser's shape against the report
rather than against a frozen copy of the numbers.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(HERE))

from issue53_packer_compare import (  # noqa: E402
    BOARDS,
    SCALAR_TILES,
    SHAPES,
    parse_pack,
    parse_speed,
)

REPORT = HERE / "logs" / "issue48_2026-09-30_report.md"
TEXT = REPORT.read_text()


def test_every_board_and_shape_has_exactly_the_scalar_tiles_in_both_packers():
    speed = parse_speed(TEXT)
    assert set(speed) == {(b, s) for b in BOARDS for s in SHAPES}
    for key, rows in speed.items():
        tiles_packers = {(r.tile, r.packer) for r in rows}
        assert tiles_packers == {(t, p) for t in SCALAR_TILES for p in ("vector", "scalar")}, key


def test_pack_rows_match_the_speed_rows_one_for_one():
    speed = parse_speed(TEXT)
    pack = parse_pack(TEXT)
    assert set(pack) == set(speed)
    for key in speed:
        assert {(r.tile, r.packer) for r in pack[key]} == {(r.tile, r.packer) for r in speed[key]}, key


def test_b3_matches_the_48_finding_scalar_wins_from_nr_8_up_vector_ties_or_wins_below():
    # #48 finding 3: "At NR of 6 or less the vector packer is level with or ahead of the scalar
    # one." 32x6 and 4x4 are the only scalar-arm tiles at NR <= 6.
    speed = parse_speed(TEXT)
    for board in BOARDS:
        by_tile = {r.tile: r for r in speed[(board, "B3")] if r.packer in ("vector", "scalar")}
        rows = speed[(board, "B3")]
        by_tile_packer = {(r.tile, r.packer): r for r in rows}
        for tile in ("32x6", "4x4"):
            v, s = by_tile_packer[(tile, "vector")], by_tile_packer[(tile, "scalar")]
            assert v.mean_gops >= s.mean_gops, (board, tile)
        for tile in ("16x8", "16x14", "16x15", "8x15"):
            v, s = by_tile_packer[(tile, "vector")], by_tile_packer[(tile, "scalar")]
            assert s.mean_gops > v.mean_gops, (board, tile)


def test_scalar_never_beats_vector_on_b2():
    # The planned answer (#53 §4): "Scalar loses on B2" would mean the packer choice is per
    # shape, not per tile.
    speed = parse_speed(TEXT)
    for board in BOARDS:
        by_tile_packer = {(r.tile, r.packer): r for r in speed[(board, "B2")]}
        for tile in SCALAR_TILES:
            v, s = by_tile_packer[(tile, "vector")], by_tile_packer[(tile, "scalar")]
            assert v.mean_gops > s.mean_gops, (board, tile)
