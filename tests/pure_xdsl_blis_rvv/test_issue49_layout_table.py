"""
Issue #49 step 1: `issue49_layout_table.py`'s `SLOW_REP_PCT` is a frozen transcription of the
vector-packer slow-rep share from `logs/issue48_2026-09-30_report.md`'s "B3"/"B3_K256" tables
(jbpi and jbpi2). Frozen on purpose -- the desk table's finding (no single quantity) is about
that specific report, and must not silently drift if the report is ever re-run or corrected.
This re-parses the report independently and checks the transcription is still exact, so a
stale copy fails loudly instead of quietly misleading the next read of the desk table.
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from issue49_layout_table import SLOW_REP_PCT  # noqa: E402

REPORT = HERE / "logs" / "issue48_2026-09-30_report.md"


def parse_report_slow_pct(report: Path) -> dict[tuple[str, str], dict[tuple[int, int], int]]:
    """The vector-packer "slow reps %" column of every "### <board>, <shape>" table in the
    report, first occurrence of each only (the report's section 2 reuses the same headings
    for an unrelated instrumented table further down)."""
    text = report.read_text()
    want = {("jbpi", "B3"), ("jbpi", "B3_K256"), ("jbpi2", "B3"), ("jbpi2", "B3_K256")}
    out: dict[tuple[str, str], dict[tuple[int, int], int]] = {}
    for sec in re.split(r"^### ", text, flags=re.M):
        head = sec.split("\n", 1)[0].strip()
        m = re.match(r"(jbpi2?), (B3(?:_K256)?)$", head)
        if not m:
            continue
        key = (m.group(1), m.group(2))
        if key not in want or key in out:  # first occurrence only
            continue
        rows: dict[tuple[int, int], int] = {}
        for line in sec.splitlines():
            if not line.startswith("|"):
                continue
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) < 7:
                continue
            tile_m = re.match(r"^(\d+)x(\d+)$", parts[0])
            if not tile_m or parts[2] != "vector":
                continue
            try:
                slow = int(parts[6])
            except ValueError:
                continue
            rows[(int(tile_m.group(1)), int(tile_m.group(2)))] = slow
        out[key] = rows
    return out


def test_the_transcribed_slow_rep_shares_still_match_the_report():
    parsed = parse_report_slow_pct(REPORT)

    assert set(parsed) == set(SLOW_REP_PCT)
    for key in SLOW_REP_PCT:
        assert parsed[key] == SLOW_REP_PCT[key], key


def test_every_tile_has_all_25_sweep_tiles_and_no_extras():
    for key, rows in SLOW_REP_PCT.items():
        assert len(rows) == 25, key
