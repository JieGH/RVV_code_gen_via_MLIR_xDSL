"""Trial s/v heatmaps for §3.6 from the #85 one-parameter sweeps (the report's data/tab_*.tex).

Usage: python3 make_heatmap.py <report/data dir>
Writes one CSV per plane and thread count; cells #85 did not run get run=0 (drawn grey).
"""
import csv
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
NS = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
KS = [16, 32, 64, 144, 256, 576, 1152]
MS = [64, 128, 300, 600, 1200, 2400, 4800]
SIM = "$^{\\sim}$"
ROW = re.compile(r"^\w+=\d+ \((\d+)\$\\times\$(\d+)\$\\times\$(\d+)\)")


def rows(path: Path):
    """Yield (m, n, k, {1: (s/v, noise), 8: (s/v, noise)}) from one #85 GOPs table."""
    for line in path.read_text().splitlines():
        hit = ROW.match(line)
        if not hit:
            continue
        m, n, k = map(int, hit.groups())
        c = [x.strip() for x in line.rstrip("\\ ").split("&")]
        # label, v1, s1, s/v 1T, v8, s8, s/v 8T, ...; a ~ on v or s = gap inside the launch range
        yield m, n, k, {1: (float(c[3]), SIM in c[1] + c[2]), 8: (float(c[6]), SIM in c[4] + c[5])}


def plane(data: Path, tables, ykey, keep):
    return {(n, {"k": k, "m": m}[ykey]): sv
            for t in tables for m, n, k, sv in rows(data / t) if keep(m, k)}


def write(pts, ys, th, out):
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "sv", "run", "noise", "label"])
        for yi, y in enumerate(ys):
            for xi, n in enumerate(NS):
                if (n, y) in pts:
                    sv, noise = pts[(n, y)][th]
                    w.writerow([xi, yi, f"{sv:.3f}", 1, int(noise), f"{sv:.2f}" + ("$^{\\sim}$" if noise else "")])
                else:
                    w.writerow([xi, yi, "nan", 0, 0, "{}"])


def main():
    data = Path(sys.argv[1])
    nk = plane(data, ["tab_pn.tex", "tab_pk16.tex", "tab_pk256.tex"], "k", lambda m, k: m == 1200)
    nm = plane(data, ["tab_pn.tex", "tab_pm16.tex", "tab_pm256.tex"], "m", lambda m, k: k == 256)
    # a cross: 9 points along n, plus 7 along the other axis at n = 16 and n = 256, sharing the base row
    assert len(nk) == len(nm) == 9 + 6 + 6, (len(nk), len(nm))
    for th in (1, 8):
        write(nk, KS, th, HERE / f"nk_{th}t.csv")
        write(nm, MS, th, HERE / f"nm_{th}t.csv")
    print(f"n x k: {len(nk)} of {len(NS) * len(KS)} cells; n x m: {len(nm)} of {len(NS) * len(MS)} cells")


if __name__ == "__main__":
    main()
