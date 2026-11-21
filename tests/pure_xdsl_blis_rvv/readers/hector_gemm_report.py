#!/usr/bin/env python3
# pyright: reportUnknownMemberType=false
# (matplotlib ships partial types; the figure is its only use here)
"""
Issue #62: report on a run_hector_gemm.sh series (Héctor's gemm.x, unedited, at 1 and 8 threads).

Per shape (one row of his dataset): his GOPs by mean time (his driver reports only the mean time per call, so there is
no "by median rep"), averaged over the launches of each thread count, with the lowest and highest launch; the 8-thread
speedup (avg 8T GOPs / avg 1T GOPs). Then one pass over the dataset per thread count (each shape once, mean time
averaged over launches). n = 1 rows run his GEMV, not gemm_B3A2C0. WARNING under 3 launches (house rule 2, #39).

--plot draws two panels over the rows of the dataset (sorted by n, m, k): GOPs at 1 and 8 threads (avg, with
the lowest-highest launch as a whisker), and the 8-thread speedup.

Usage: hector_gemm_report.py <series folder> [-o report.md] [--plot fig.png]
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import rawlog


def read_csv(path: Path) -> dict[int, tuple[int, int, int, float]]:
    """row -> (m, n, k, seconds per call)"""
    out: dict[int, tuple[int, int, int, float]] = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        row, m, n, k, _gflops, secs = line.split(";")
        out[int(row)] = (int(m), int(n), int(k), float(secs))
    return out


Launches = dict[int, list[dict[int, tuple[int, int, int, float]]]]  # threads -> one CSV per ok launch


def load(series: Path) -> Launches:
    by_th: Launches = {1: [], 8: []}
    for r in rawlog.read_tsv(series / "runs.tsv"):
        if r["exit"] == "0":
            by_th.setdefault(int(r["threads"]), []).append(read_csv(series / f"L{r['launch']}_T{r['threads']}_{r['board']}.csv"))
    return by_th


def gops(m: int, n: int, k: int, s: float) -> float:
    return 2.0 * m * n * k / s / 1e9


def render(series: Path) -> str:
    by_th = load(series)

    out = [f"# Héctor's GEMM on the board: `{series.name}`", "", "```", (series / "plan.txt").read_text().rstrip(), "```", ""]
    notes = (series / "side_notes.txt").read_text().strip() if (series / "side_notes.txt").exists() else ""
    out += ["**Side notes (busy boards):** " + (notes.replace("\n", "; ") if notes else "none"), ""]
    out += [f"Launches ok: {len(by_th[1])} at 1 thread (core 4), {len(by_th[8])} at 8 threads (cores 0-7, core 0 "
            "included: user's exception for the 8-thread run). Logs: runs.tsv, L<n>_T<threads>_<board>.{log,csv}.", ""]
    for th, ls in by_th.items():
        if len(ls) < 3:
            out += [f"WARNING: only {len(ls)} launch(es) at {th} thread(s); house rule 2 asks for 3.", ""]

    def cell(rows: list[float]) -> str:
        return f"{statistics.mean(rows):.2f} ({min(rows):.2f}-{max(rows):.2f})" if rows else "-"

    out += ["GOPs by mean time, avg over launches (lowest-highest launch).", "",
            "| row | m | n | k | 1 thread | 8 threads | speedup |", "|---|---|---|---|---|---|---|"]
    rows = sorted({row for ls in by_th.values() for r in ls for row in r})
    for row in rows:
        m, n, k, _ = next(r[row] for ls in by_th.values() for r in ls if row in r)
        g = {th: [gops(m, n, k, r[row][3]) for r in ls if row in r] for th, ls in by_th.items()}
        sp = f"{statistics.mean(g[8]) / statistics.mean(g[1]):.2f}" if g[1] and g[8] else "-"
        tag = f"{n} (GEMV)" if n == 1 else str(n)
        out.append(f"| {row} | {m} | {tag} | {k} | {cell(g[1])} | {cell(g[8])} | {sp} |")
    out += ["", "One pass over the dataset (each row once, mean time per call averaged over launches):", "",
            "| threads | time | GOPs |", "|---|---|---|"]
    for th, ls in by_th.items():
        if not ls:
            continue
        secs = sum(statistics.mean(r[row][3] for r in ls if row in r) for row in rows)
        flop = sum(2.0 * m * n * k for m, n, k, _ in (next(r[row] for r in ls if row in r) for row in rows))
        out.append(f"| {th} thread{'s' if th > 1 else ''} | {secs * 1e3:.3f} ms | {flop / secs / 1e9:.2f} |")
    return "\n".join(out) + "\n"


def plot(series: Path, png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_th = load(series)
    shapes = {row: r[row][:3] for ls in by_th.values() for r in ls for row in r}
    rows = sorted(shapes, key=lambda row: (shapes[row][1], shapes[row][0], shapes[row][2]))
    xs = list(range(len(rows)))
    g = {th: [[gops(*shapes[row], r[row][3]) for r in ls if row in r] for row in rows] for th, ls in by_th.items()}
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    color = {1: "#2a78d6", 8: "#eb6834"}  # reference palette slots 1-2, validated (dataviz validate_palette.js)

    fig, (top, bot) = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    fig.patch.set_facecolor("#fcfcfb")
    for th in (1, 8):
        pts = [(x, v) for x, v in zip(xs, g[th]) if v]
        if not pts:
            continue
        avg = [statistics.mean(v) for _, v in pts]
        lo = [a - min(v) for a, (_, v) in zip(avg, pts)]
        hi = [max(v) - a for a, (_, v) in zip(avg, pts)]
        top.errorbar([x for x, _ in pts], avg, yerr=[lo, hi], fmt="o", ms=5, color=color[th], elinewidth=1,
                     capsize=0, label=f"{th} thread{'s' if th > 1 else ''} ({'core 4' if th == 1 else 'cores 0-7'})",
                     markeredgecolor="#fcfcfb", markeredgewidth=1)
    top.set_ylabel("GOPs by mean time", color=ink)
    top.legend(frameon=False, loc="upper left")
    sp = [statistics.mean(b) / statistics.mean(a) if a and b else 0.0 for a, b in zip(g[1], g[8])]
    bot.bar(xs, sp, width=0.7, color=color[8])
    bot.axhline(8, color=muted, lw=1, ls="--")
    bot.text(len(xs) - 0.5, 8, "8x (ideal)", ha="right", va="bottom", color=muted, fontsize=8)
    bot.axhline(1, color=muted, lw=1)
    bot.set_ylabel("8-thread speedup", color=ink)
    bot.set_xticks(xs)
    bot.set_xticklabels([f"{m}x{n}x{k}" + (" GEMV" if n == 1 else "") for m, n, k in (shapes[r] for r in rows)],
                        rotation=90, fontsize=7, color=muted)
    bot.set_xlabel("shape m x n x k (dataset rows, sorted by n, m, k)", color=ink)
    for ax in (top, bot):
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
    n1, n8 = len(by_th[1]), len(by_th[8])
    fig.suptitle(f"Héctor's gemm_B3A2C0 (b292c92, unedited) on {series.name}: {n1} + {n8} launches, "
                 "avg with lowest-highest launch", color=ink, fontsize=11)
    fig.tight_layout()
    fig.savefig(png, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("series", type=Path)
    ap.add_argument("-o", "--out", type=Path)
    ap.add_argument("--plot", type=Path)
    a = ap.parse_args()
    if a.plot:
        plot(a.series, a.plot)
    md = render(a.series)
    if a.out:
        a.out.write_text(md)
    else:
        print(md, end="")


if __name__ == "__main__":
    main()
