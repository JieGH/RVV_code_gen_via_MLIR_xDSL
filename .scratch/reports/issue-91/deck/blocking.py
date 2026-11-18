"""Issue #91: cache blocking (mc, nc, kc) of every run, read from the committed console logs.

The runs' CSVs do not record the blocking; the console logs print it per shape and tile
("+- Y00 (76800x16x27) 8x4 [MR..." or "[C_n4k16 (1200x4x16)] Testing variant 1/1 : MR24_NR8",
then "[cache blocking: mc=.. nc=.. kc=..]"). The board's C model (modelLevel/model_level.c, --no-nc-overrides) sets it
from shape and tile only: blk() asserts that every log agrees, across threads, packers and launches.
"""
import glob, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../../.."))
LOGS = sorted(glob.glob(os.path.join(REPO, "tests/pure_xdsl_blis_rvv/logs/yolo_sweep_*/*.log"))
              + glob.glob(os.path.join(REPO, "tests/pure_xdsl_blis_rvv/logs/issue8[35]*.out"))
              + glob.glob(os.path.join(REPO, ".scratch/reports/issue-85/driver_logs/*.out")))
HDR = [re.compile(r"\+- \S+ \((\d+)x(\d+)x(\d+)\) (\d+)x(\d+) \[MR"),
       re.compile(r"\[\S+ \((\d+)x(\d+)x(\d+)\)\] Testing variant \d+/\d+ : MR(\d+)_NR(\d+)")]
BLK = re.compile(r"cache blocking: mc=(\d+) .*?nc=(\d+) .*?kc=(\d+)")

_seen = {}
for _f in LOGS:
    _cur = None
    for _line in open(_f, errors="replace"):
        for _h in HDR:
            if _m := _h.search(_line):
                _cur = tuple(map(int, _m.groups()))
        if (_b := BLK.search(_line)) and _cur:
            _seen.setdefault(_cur, set()).add(tuple(map(int, _b.groups())))
            _cur = None


def blk(m, n, k, tile):
    """(mc, nc, kc) the logs print for shape m x n x k and tile 'MRxNR'."""
    mr, nr = map(int, tile.split("x"))
    got = _seen.get((m, n, k, mr, nr))
    assert got and len(got) == 1, ((m, n, k, tile), got)
    return next(iter(got))


def fmt(m, n, k, tile):
    mc, nc, kc = blk(m, n, k, tile)
    return f"mc {mc}, nc {nc}, kc {kc}"


if __name__ == "__main__":
    clash = {k: v for k, v in _seen.items() if len(v) > 1}
    assert not clash, clash
    print(len(_seen), "shape/tile pairs, one blocking each, from", len(LOGS), "logs")
