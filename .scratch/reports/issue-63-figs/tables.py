"""Issue #63 report tables: per shape, ours at 1 thread (#61) and 8 threads beside Héctor's 8 threads.

Reads the search_report.py outputs (ours) and his gemm.x CSVs; prints Markdown.
Usage: tables.py <ours 1T report.md> <ours 8T report.md> <his L*_T8_*.csv ...>
"""
import re
import sys
from pathlib import Path


def best_rows(md: str) -> dict[str, list[str]]:
    """shape -> cells of the 'Best tile per shape' table"""
    sec = md.split("## Best tile per shape")[1].split("## ")[0]
    rows = [[c.strip() for c in l.strip("|").split("|")] for l in sec.splitlines() if l.startswith("| Y")]
    return {r[0]: r for r in rows}


def grid(md: str) -> tuple[list[str], dict[str, list[float]]]:
    sec = md.split("## GOPs grid")[1]
    lines = [l for l in sec.splitlines() if l.startswith("|")]
    tiles = [c.strip() for c in lines[0].strip("|").split("|")][4:]
    out = {}
    for l in lines[2:]:
        c = [x.strip().strip("*") for x in l.strip("|").split("|")]
        out[c[0]] = [float(x) for x in c[4:]]
    return tiles, out


def his(csvs: list[Path]) -> dict[tuple[int, int, int], list[float]]:
    """(m, n, k) -> seconds per call, one per launch"""
    out: dict[tuple[int, int, int], list[float]] = {}
    for p in csvs:
        for line in p.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            _, m, n, k, _g, s = line.split(";")
            out.setdefault((int(m), int(n), int(k)), []).append(float(s))
    return out


def gops(m: int, n: int, k: int, s: float) -> float:
    return 2 * m * n * k / s / 1e9


def main() -> None:
    one, eight = (Path(a).read_text() for a in sys.argv[1:3])
    b1, b8 = best_rows(one), best_rows(eight)
    tiles, g8 = grid(eight)
    h = his([Path(a) for a in sys.argv[3:]])
    print("| shape | m | n | k | ours 1T best (#61) | ours 8T best, by mean time | ours 8T best, by median rep "
          "| his 8T (16x8 / 16x4), by mean time |")
    print("|---|---|---|---|---|---|---|---|")
    t_ours = t_his = flops = 0.0
    for shape, r in b8.items():
        m, n, k = (int(x) for x in r[1:4])
        hs = h[(m, n, k)]
        hg = [gops(m, n, k, s) for s in hs]
        print(f"| {shape} | {m} | {n} | {k} | {b1[shape][5]} {b1[shape][6].split()[0]} | {r[5]} {r[6]} "
              f"| {r[7]} {r[8]} | {sum(hg) / len(hg):.2f} ({min(hg):.2f}-{max(hg):.2f}) |")
        f = 2 * m * n * k
        t_ours += f / (max(g8[shape]) * 1e9)
        t_his += sum(hs) / len(hs)
        flops += f
    print()
    print(f"One pass, best 8T tile per shape (ours): {t_ours * 1e3:.1f} ms, {flops / t_ours / 1e9:.2f} GOPs")
    print(f"One pass, his 8T (same 50 shapes, avg over launches): {t_his * 1e3:.1f} ms, {flops / t_his / 1e9:.2f} GOPs")
    y0 = next(s for s, r in b8.items() if r[1:4] == ["76800", "16", "27"])
    f0 = 2 * 76800 * 16 * 27
    t0_ours = f0 / (max(g8[y0]) * 1e9)
    t0_his = sum(h[(76800, 16, 27)]) / len(h[(76800, 16, 27)])
    print(f"Without {y0} (76800x16x27): ours {(t_ours - t0_ours) * 1e3:.1f} ms, "
          f"{(flops - f0) / (t_ours - t0_ours) / 1e9:.2f} GOPs; his {(t_his - t0_his) * 1e3:.1f} ms, "
          f"{(flops - f0) / (t_his - t0_his) / 1e9:.2f} GOPs")
    assert len(b8) == 50 and set(b1) == set(b8) and len(tiles) == 19  # self-check: same 50 shapes, 19 tiles


if __name__ == "__main__":
    main()
