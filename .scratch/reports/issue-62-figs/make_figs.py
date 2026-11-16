#!/usr/bin/env python3
# pyright: reportUnknownMemberType=false
"""Issue #62 report figure: 8-thread speedup by n, jbpi beside jbpi2, from the two committed series.

Run from the repo root: .venv/bin/python .scratch/reports/issue-62-figs/make_figs.py
Also prints the per-shape table of Appendix A.
"""

import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests/pure_xdsl_blis_rvv"))
from hector_gemm_report import gops, load  # noqa: E402

LOGS = ROOT / "tests/pure_xdsl_blis_rvv/logs"
SERIES = {"jbpi": LOGS / "hector_gemm_2026-11-16_111457", "jbpi2": LOGS / "hector_gemm_2026-11-16_113052"}
COLOR = {"jbpi": "#2a78d6", "jbpi2": "#eb6834"}  # reference palette slots 1-2, validated (dataviz validate_palette.js)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

data = {b: load(d) for b, d in SERIES.items()}
shape = {row: v[:3] for row, v in data["jbpi"][1][0].items()}


def mean_t(board: str, th: int, row: int) -> float:
    return statistics.mean(c[row][3] for c in data[board][th])


ns = sorted({n for _, n, _ in shape.values()})
speed: dict[str, list[float]] = {}
for b in SERIES:
    speed[b] = []
    for n in ns:
        rows = [r for r, (_, rn, _) in shape.items() if rn == n]
        speed[b].append(sum(mean_t(b, 1, r) for r in rows) / sum(mean_t(b, 8, r) for r in rows))

fig, ax = plt.subplots(figsize=(9, 4.2))
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
xs = list(range(len(ns)))
w = 0.38
for i, b in enumerate(SERIES):
    bars = ax.bar([x + (i - 0.5) * (w + 0.02) for x in xs], speed[b], width=w, color=COLOR[b], label=b)
    for bar, v in zip(bars, speed[b]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.08, f"{v:.1f}", ha="center", va="bottom", fontsize=7, color=MUTED)
ax.axhline(8, color=MUTED, lw=1, ls="--")
ax.text(len(ns) - 0.5, 8, "8x (ideal)", ha="right", va="bottom", fontsize=8, color=MUTED)
ax.set_xticks(xs)
ax.set_xticklabels([f"{n}\n(GEMV)" if n == 1 else str(n) for n in ns], color=MUTED)
ax.set_xlabel("output width n", color=INK)
ax.set_ylabel("8-thread speedup (one pass over the rows)", color=INK)
ax.set_ylim(0, 10)
ax.legend(frameon=False, loc="upper left", ncol=2)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(GRID)
ax.set_title("Héctor's gemm_B3A2C0: 8 threads (cores 0-7) against 1 (core 4), by n", color=INK, fontsize=11)
fig.tight_layout()
fig.savefig(Path(__file__).parent / "speedup_by_n.png", dpi=150, facecolor=SURFACE)

print("| row | m | n | k | jbpi 1T | jbpi 8T | jbpi speedup | jbpi2 1T | jbpi2 8T | jbpi2 speedup |")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in sorted(shape, key=lambda r: (shape[r][1], shape[r][0], shape[r][2])):
    m, n, k = shape[r]
    cells = []
    for b in SERIES:
        g1 = statistics.mean(gops(m, n, k, c[r][3]) for c in data[b][1])
        g8 = statistics.mean(gops(m, n, k, c[r][3]) for c in data[b][8])
        cells.append(f"{g1:.2f} | {g8:.2f} | {g8 / g1:.2f}x")
    print(f"| {r} | {m} | {n}{' (GEMV)' if n == 1 else ''} | {k} | " + " | ".join(cells) + " |")
