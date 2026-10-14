#!/usr/bin/env python3
"""
Issue #53: scalar against vector packer on all four shapes of the #48 sweep, from the
committed logs (no board, no code change) -- #48 only compared the two packers on B3.

Reads `logs/issue48_2026-09-30_report.md` directly (no hand-transcription: this prints what
the report says, so it cannot drift from it) and prints, for the 7 tiles of the scalar arm
(16x8, 16x14, 16x15, 16x16, 8x15, 32x6, 4x4): one speed table per shape (vector and scalar,
both boards), a per-tile summary of which packer wins on how many shapes, and the instrumented
fast-rep pack times of both packers per tile and shape.

Usage: python3 issue53_packer_compare.py [--report logs/issue48_2026-09-30_report.md]
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent

SCALAR_TILES = ("16x8", "16x14", "16x15", "16x16", "8x15", "32x6", "4x4")
SHAPES = ("B3", "B3_K256", "B2", "R24")
BOARDS = ("jbpi2", "jbpi")  # report order


@dataclass
class SpeedRow:
    tile: str
    packer: str
    mean_gops: float
    median_gops: float
    slow_pct: int


@dataclass
class PackRow:
    tile: str
    packer: str
    fast_pack_ms: float
    fast_compute_ms: float


def _sections(text: str, heading_num: str) -> dict[tuple[str, str], str]:
    """`## <heading_num>.` section's `### <board>, <shape>` subsections, first occurrence of
    each key only (the two `## 1.`/`## 2.` top sections reuse the same `### board, shape`
    headings, so this must not cross between them)."""
    top = re.split(r"^## ", text, flags=re.M)
    body = next(s for s in top if s.startswith(f"{heading_num}."))
    out: dict[tuple[str, str], str] = {}
    for sec in re.split(r"^### ", body, flags=re.M)[1:]:
        head = sec.split("\n", 1)[0].strip()
        m = re.match(r"(jbpi2?), (\w+)$", head)
        if not m:
            continue
        key = (m.group(1), m.group(2))
        if key not in out:
            out[key] = sec
    return out


def _table_rows(section: str) -> list[list[str]]:
    rows = []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 7 or not re.match(r"^\d+x\d+$", parts[0]):
            continue
        rows.append(parts)
    return rows


def parse_speed(text: str) -> dict[tuple[str, str], list[SpeedRow]]:
    """(board, shape) -> speed rows for the 7 scalar-arm tiles, both packers."""
    out: dict[tuple[str, str], list[SpeedRow]] = {}
    for key, sec in _sections(text, "1").items():
        rows = []
        for parts in _table_rows(sec):
            tile, _kc, packer = parts[0], parts[1], parts[2]
            if tile not in SCALAR_TILES:
                continue
            mean_gops = float(parts[4].split()[0])
            median_gops = float(parts[5])
            slow_pct = int(parts[6])
            rows.append(SpeedRow(tile, packer, mean_gops, median_gops, slow_pct))
        out[key] = rows
    return out


def parse_pack(text: str) -> dict[tuple[str, str], list[PackRow]]:
    """(board, shape) -> instrumented fast-rep pack/compute rows for the 7 scalar-arm tiles,
    both packers. "(N reps)" or "no slow reps" rows still carry a fast-rep split; only the
    fast side is read here (plan step 3 asks for the fast-rep pack time, not the slow one)."""
    out: dict[tuple[str, str], list[PackRow]] = {}
    for key, sec in _sections(text, "2").items():
        rows = []
        for parts in _table_rows(sec):
            tile, _kc, packer = parts[0], parts[1], parts[2]
            if tile not in SCALAR_TILES:
                continue
            fast_pack, fast_compute = (float(x) for x in parts[6].split("/"))
            rows.append(PackRow(tile, packer, fast_pack, fast_compute))
        out[key] = rows
    return out


def format_speed_tables(speed: dict[tuple[str, str], list[SpeedRow]]) -> str:
    out = []
    for shape in SHAPES:
        out.append(f"### {shape}\n")
        out.append(
            "| tile | jbpi2 vector | jbpi2 scalar | jbpi vector | jbpi scalar |\n"
            "|---|---|---|---|---|"
        )
        by_board = {b: {(r.tile, r.packer): r for r in speed[(b, shape)]} for b in BOARDS}
        for tile in SCALAR_TILES:
            cells = []
            for board in BOARDS:
                for packer in ("vector", "scalar"):
                    r = by_board[board].get((tile, packer))
                    cells.append(f"{r.mean_gops:.2f} ({r.slow_pct}%)" if r else "-")
            out.append(f"| {tile} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def format_summary(speed: dict[tuple[str, str], list[SpeedRow]]) -> str:
    """Per tile, per board: how many of the 4 shapes each packer wins by mean GOPs."""
    out = ["| tile | jbpi2: vector wins / scalar wins | jbpi: vector wins / scalar wins |",
           "|---|---|---|"]
    for tile in SCALAR_TILES:
        cells = []
        for board in BOARDS:
            v_wins = s_wins = 0
            for shape in SHAPES:
                by_packer = {r.packer: r for r in speed[(board, shape)] if r.tile == tile}
                if "vector" in by_packer and "scalar" in by_packer:
                    if by_packer["vector"].mean_gops >= by_packer["scalar"].mean_gops:
                        v_wins += 1
                    else:
                        s_wins += 1
            cells.append(f"{v_wins} / {s_wins}")
        out.append(f"| {tile} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def format_pack_tables(pack: dict[tuple[str, str], list[PackRow]]) -> str:
    out = []
    for shape in SHAPES:
        out.append(f"### {shape}, fast-rep pack ms (vector / scalar)\n")
        out.append("| tile | jbpi2 | jbpi |\n|---|---|---|")
        by_board = {b: {(r.tile, r.packer): r for r in pack[(b, shape)]} for b in BOARDS}
        for tile in SCALAR_TILES:
            cells = []
            for board in BOARDS:
                v = by_board[board].get((tile, "vector"))
                s = by_board[board].get((tile, "scalar"))
                cells.append(f"{v.fast_pack_ms:.3f} / {s.fast_pack_ms:.3f}" if v and s else "-")
            out.append(f"| {tile} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", default=str(HERE / "logs" / "issue48_2026-09-30_report.md"))
    args = ap.parse_args(argv)

    text = Path(args.report).read_text()
    speed = parse_speed(text)
    pack = parse_pack(text)

    print("## Speed by shape (GOPs by mean time; slow-rep share in brackets)\n")
    print(format_speed_tables(speed))
    print("## Per-tile summary: shapes won (of 4), by mean GOPs\n")
    print(format_summary(speed))
    print()
    print("## Instrumented: fast-rep pack time (debug numbers, not speed)\n")
    print(format_pack_tables(pack))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
