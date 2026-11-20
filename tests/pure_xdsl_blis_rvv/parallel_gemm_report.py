#!/usr/bin/env python3
# pyright: reportUnknownMemberType=false
# (matplotlib ships partial types; the figure is its only use here)
"""
Issue #56: report on a run_parallel_gemm.sh series (our GEMM, tile 24x8, at 1 and 8 threads on the YOLO26 shapes).

Per shape: our GOPs by mean time (avg over the launches of each thread count, lowest-highest launch) and by median rep,
the 8-thread speedup (avg 8T / avg 1T, by mean time), and Héctor's GEMMLib at 1 and 8 threads from his #62 series
(--hector; his driver gives the mean only), matched by row (our label Y<l> = his row l). Then one pass over the shapes
per thread count (each shape once), and the same per n. Correctness: PASS/FAIL rows per launch (all_raw_*.txt).
WARNING under 3 launches (house rule 2, #39); RED NOTE for a launch made under another user's load.

--plot: GOPs by mean time per shape (ours at 1 and 8 threads, his at 8), and our 8-thread speedup.

Usage: parallel_gemm_report.py <series folder> [--logs DIR] [--hector <#62 series>] [-o report.md] [--plot fig.png]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import hector_gemm_report  # noqa: E402
import rawlog  # noqa: E402
from reps_report import MIN_LAUNCHES, CaseStats, case_stats, conditions_lines, load_notes, read_reps, reps_files  # noqa: E402

Shape = tuple[int, int, int]
THREADS = (1, 8)


def load(series: Path, logs: Path) -> tuple[dict[str, Shape], dict[int, dict[str, list[CaseStats]]], list[tuple[str, str, Path]]]:
    """Shapes by label; per thread count and label one CaseStats per ok launch; (launch, threads, run dir) per ok launch."""
    shapes: dict[str, Shape] = {}
    stats: dict[int, dict[str, list[CaseStats]]] = {th: {} for th in THREADS}
    runs: list[tuple[str, str, Path]] = []
    for row in rawlog.read_tsv(series / "runs.tsv"):
        if row["exit"] != "0":
            continue
        launch, th, run_dir = row["launch"], row["threads"], logs / row["result_dir"]
        runs.append((launch, th, run_dir))
        for f in reps_files(run_dir):
            for (label, _nc), case in read_reps(f).items():
                shapes[label] = (case.m, case.n, case.k)
                stats.setdefault(int(th), {}).setdefault(label, []).append(case_stats(case.ops, case.seconds))
    return shapes, stats, runs


def hector(series: Path | None) -> dict[int, dict[str, list[float]]]:
    """threads -> label Y<row> -> his GOPs per launch."""
    out: dict[int, dict[str, list[float]]] = {th: {} for th in THREADS}
    if series is None or not (series / "runs.tsv").exists():
        return out
    for th, launches in hector_gemm_report.load(series).items():
        for csv in launches:
            for row, (m, n, k, secs) in csv.items():
                out.setdefault(th, {}).setdefault(f"Y{row:02d}", []).append(hector_gemm_report.gops(m, n, k, secs))
    return out


def _avg(xs: list[float]) -> float:
    return statistics.fmean(xs)


def _spread(xs: list[float]) -> str:
    return f"{_avg(xs):.2f} ({min(xs):.2f}-{max(xs):.2f})" if xs else "-"


def _pass_ms(shapes: dict[str, Shape], labels: list[str], gops_of: dict[str, float]) -> float:
    return sum(2.0 * shapes[lb][0] * shapes[lb][1] * shapes[lb][2] / gops_of[lb] for lb in labels) / 1e6


def _correctness(run_dir: Path) -> str:
    status = [r["Status"] for f in rawlog.any_raw(run_dir) for r in rawlog.read_rows(f)]
    return f"{status.count('PASS')} PASS, {status.count('FAIL')} FAIL, {len(status) - status.count('PASS') - status.count('FAIL')} other"


def render(series: Path, logs: Path, hector_series: Path | None) -> str:
    shapes, stats, runs = load(series, logs)
    his = hector(hector_series)
    labels = sorted(shapes, key=lambda lb: (shapes[lb][1], shapes[lb][0], shapes[lb][2]))
    full = [lb for lb in labels if all(lb in stats[th] for th in THREADS)]
    by_mean = {th: {lb: _avg([s.gops_mean_time for s in stats[th][lb]]) for lb in full} for th in THREADS}
    by_med = {th: {lb: _avg([s.gops_median for s in stats[th][lb]]) for lb in full} for th in THREADS}
    his_full = [lb for lb in full if all(his[th].get(lb) for th in THREADS)]
    his_mean = {th: {lb: _avg(his[th][lb]) for lb in his_full} for th in THREADS}

    out = [f"# Our parallel GEMM on the board: `{series.name}`", "", "```", (series / "plan.txt").read_text().rstrip(), "```", ""]
    n_launch = {th: sum(1 for _, t, _ in runs if int(t) == th) for th in THREADS}
    out += [f"Launches ok: {n_launch[1]} at 1 thread (core 4), {n_launch[8]} at 8 threads (cores 0-7, core 0 included: "
            "user's exception, #56 §5). Héctor's numbers: " + (f"`{hector_series.name}` (#62)" if hector_series else "none") + ".", ""]
    for th in THREADS:
        if n_launch[th] < MIN_LAUNCHES:
            out += [f"WARNING: {n_launch[th]} launch(es) at {th} thread(s), fewer than {MIN_LAUNCHES} (house rule 2).", ""]
    out += ["## Run conditions", ""] + [f"- L{lc} {t}T: {_correctness(d)}" for lc, t, d in runs] + [""]
    out += conditions_lines([d for _, _, d in runs]) + load_notes([d for _, _, d in runs]) + [""]
    if not full:
        return "\n".join(out + ["No shape has launches at both thread counts yet.", ""])

    out += ["## One pass over the shapes", "",
            f"{len(full)} shapes, each once. ms = sum of 2mnk / GOPs; GOPs = total ops / total time.", "",
            "| | 1 thread (core 4) | 8 threads (cores 0-7) | speedup |", "|---|---|---|---|"]
    ops = sum(2.0 * shapes[lb][0] * shapes[lb][1] * shapes[lb][2] for lb in full)
    for name, src in (("ours, by mean time", by_mean), ("ours, by median rep", by_med)):
        t1, t8 = _pass_ms(shapes, full, src[1]), _pass_ms(shapes, full, src[8])
        out.append(f"| {name} | {t1:.1f} ms · {ops / t1 / 1e6:.2f} GOPs | {t8:.1f} ms · {ops / t8 / 1e6:.2f} GOPs | {t1 / t8:.2f}x |")
    if his_full:
        ops_h = sum(2.0 * shapes[lb][0] * shapes[lb][1] * shapes[lb][2] for lb in his_full)
        for name, src in (("ours, by mean time, same shapes", by_mean), ("Héctor, by mean time", his_mean)):
            t1, t8 = _pass_ms(shapes, his_full, src[1]), _pass_ms(shapes, his_full, src[8])
            out.append(f"| {name} ({len(his_full)} shapes) | {t1:.1f} ms · {ops_h / t1 / 1e6:.2f} GOPs | "
                       f"{t8:.1f} ms · {ops_h / t8 / 1e6:.2f} GOPs | {t1 / t8:.2f}x |")
    out.append("")

    out += ["## By n (one pass over the shapes with that n, by mean time)", "",
            "| n | shapes | ours 1T | ours 8T | ours speedup | Héctor 1T | Héctor 8T | Héctor speedup |", "|---|---|---|---|---|---|---|---|"]
    for n in sorted({shapes[lb][1] for lb in full}):
        ls = [lb for lb in full if shapes[lb][1] == n]
        o = sum(2.0 * shapes[lb][0] * shapes[lb][1] * shapes[lb][2] for lb in ls)
        g = {th: o / _pass_ms(shapes, ls, by_mean[th]) / 1e6 for th in THREADS}
        hs = [lb for lb in ls if lb in his_full]
        if hs:
            oh = sum(2.0 * shapes[lb][0] * shapes[lb][1] * shapes[lb][2] for lb in hs)
            h = {th: oh / _pass_ms(shapes, hs, his_mean[th]) / 1e6 for th in THREADS}
            hcol = f"{h[1]:.2f} | {h[8]:.2f} | {h[8] / h[1]:.2f}x"
        else:
            hcol = "- | - | -"
        out.append(f"| {n} | {len(ls)} | {g[1]:.2f} | {g[8]:.2f} | {g[8] / g[1]:.2f}x | {hcol} |")
    out.append("")

    out += ["## Per shape", "",
            "GOPs by mean time: avg over launches (lowest-highest launch). Median: GOPs by median rep, avg over launches.", "",
            "| shape | m x n x k | 1T mean | 1T median | 8T mean | 8T median | speedup | Héctor 1T | Héctor 8T | ours 8T / his 8T |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for lb in full:
        m, n, k = shapes[lb]
        m1 = [s.gops_mean_time for s in stats[1][lb]]
        m8 = [s.gops_mean_time for s in stats[8][lb]]
        h1, h8 = his[1].get(lb, []), his[8].get(lb, [])
        ratio = f"{_avg(m8) / _avg(h8):.2f}" if h8 else "-"
        out.append(f"| {lb} | {m}x{n}x{k} | {_spread(m1)} | {by_med[1][lb]:.2f} | {_spread(m8)} | {by_med[8][lb]:.2f} | "
                   f"{_avg(m8) / _avg(m1):.2f}x | {_spread(h1)} | {_spread(h8)} | {ratio} |")
    out.append("")
    return "\n".join(out)


def plot(series: Path, logs: Path, hector_series: Path | None, png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shapes, stats, _ = load(series, logs)
    his = hector(hector_series)
    labels = [lb for lb in sorted(shapes, key=lambda lb: (shapes[lb][1], shapes[lb][0], shapes[lb][2]))
              if all(lb in stats[th] for th in THREADS)]
    if not labels:
        return
    xs = list(range(len(labels)))
    ink, muted, grid, bg = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    color = {"1": "#2a78d6", "8": "#eb6834", "h8": "#52514e"}  # palette slots 1-2 as in hector_gemm_report.py, plus ink-muted
    fig, (top, bot) = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    fig.patch.set_facecolor(bg)
    series_pts = [("1", "ours, 1 thread (core 4)", [[s.gops_mean_time for s in stats[1][lb]] for lb in labels]),
                  ("8", "ours, 8 threads (cores 0-7)", [[s.gops_mean_time for s in stats[8][lb]] for lb in labels]),
                  ("h8", "Héctor, 8 threads (cores 0-7)", [his[8].get(lb, []) for lb in labels])]
    for key, name, vals in series_pts:
        pts = [(x, v) for x, v in zip(xs, vals) if v]
        if not pts:
            continue
        avg = [_avg(v) for _, v in pts]
        top.errorbar([x for x, _ in pts], avg, yerr=[[a - min(v) for a, (_, v) in zip(avg, pts)],
                                                     [max(v) - a for a, (_, v) in zip(avg, pts)]],
                     fmt="o" if key != "h8" else "_", ms=5 if key != "h8" else 9, color=color[key], elinewidth=1,
                     capsize=0, label=name, markeredgewidth=1.5 if key == "h8" else 1,
                     markeredgecolor=bg if key != "h8" else color[key])
    top.set_ylabel("GOPs by mean time", color=ink)
    top.legend(frameon=False, loc="upper left")
    sp = [_avg([s.gops_mean_time for s in stats[8][lb]]) / _avg([s.gops_mean_time for s in stats[1][lb]]) for lb in labels]
    bot.bar(xs, sp, width=0.7, color=color["8"])
    bot.axhline(8, color=muted, lw=1, ls="--")
    bot.text(len(xs) - 0.5, 8, "8x (ideal)", ha="right", va="bottom", color=muted, fontsize=8)
    bot.axhline(1, color=muted, lw=1)
    bot.set_ylabel("our 8-thread speedup", color=ink)
    bot.set_xticks(xs)
    bot.set_xticklabels(["{}x{}x{}".format(*shapes[lb]) for lb in labels], rotation=90, fontsize=7, color=muted)
    bot.set_xlabel("shape m x n x k (sorted by n, m, k)", color=ink)
    for ax in (top, bot):
        ax.set_facecolor(bg)
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
    fig.suptitle(f"Our GEMM (24x8) at 1 and 8 threads, {series.name}: avg with lowest-highest launch", color=ink, fontsize=11)
    fig.tight_layout()
    fig.savefig(png, dpi=150, facecolor=bg)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("series", type=Path)
    ap.add_argument("--logs", type=Path, default=HERE / "logs", help="where the run_* folders of runs.tsv are")
    ap.add_argument("--hector", type=Path, default=None, help="Héctor's #62 series folder (hector_gemm_*)")
    ap.add_argument("-o", type=Path, default=None)
    ap.add_argument("--plot", type=Path, default=None)
    a = ap.parse_args()
    text = render(a.series, a.logs, a.hector)
    if a.o:
        a.o.write_text(text)
    else:
        print(text)
    if a.plot:
        plot(a.series, a.logs, a.hector, a.plot)


if __name__ == "__main__":
    main()
