#!/usr/bin/env python3
"""Issue #94 report data: V (ours, vector packer), S (ours, scalar packer), H (Héctor's gemm.x) at 1 and 8 threads.

Reads the run_issue94.sh series folder (steps.tsv: one row per launch, the series folder each runner wrote) and the
committed run folders under tests/pure_xdsl_blis_rvv/logs/. Writes CSVs (pgfplots) and LaTeX table rows into data/.

Metrics, per shape and launch: mean time per call (GOPs by mean time = 2mnk / mean time) and median rep time
(GOPs by median rep). Across launches the times are averaged. One pass = the sum of those times over the 50 shapes,
each shape once; its GOPs = total ops / pass time. Héctor's driver gives the mean time only.

Usage: make_data.py <issue94 series folder> [--logs DIR] [--out DIR]
       make_data.py --check   (reproduces the #56/#62 numbers quoted in #91 §1.2 from their committed logs)
"""
from __future__ import annotations

import re
import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TESTS = ROOT / "tests" / "pure_xdsl_blis_rvv"
sys.path.insert(0, str(TESTS))
import hector_gemm_report  # noqa: E402
from reps_report import case_stats, read_reps, reps_files  # noqa: E402

SETUPS = ("V", "S", "H")
NAMES = {"V": "ours, vector packer", "S": "ours, scalar packer", "H": "H\\'ector's gemm.bin"}
THREADS = (1, 8)
Shape = tuple[int, int, int]
# times[setup][threads][label] = list over launches of (mean s, median s or None)
Times = dict[str, dict[int, dict[str, list[tuple[float, float | None]]]]]


def ours_launch(run_dir: Path, shapes: dict[str, Shape]) -> dict[str, tuple[float, float | None]]:
    out: dict[str, tuple[float, float | None]] = {}
    for f in reps_files(run_dir):
        for (label, _nc), case in read_reps(f).items():
            shapes[label] = (case.m, case.n, case.k)
            st = case_stats(case.ops, case.seconds)
            out[label] = (case.ops / (st.gops_mean_time * 1e9), case.ops / (st.gops_median * 1e9))
    return out


def load(launches: list[tuple[str, int, Path]], logs: Path) -> tuple[Times, dict[str, Shape], list[str]]:
    """launches: (setup, threads, series folder of one runner). Returns times, our shapes, correctness lines."""
    times: Times = {s: {t: {} for t in THREADS} for s in SETUPS}
    shapes: dict[str, Shape] = {}
    checks: list[str] = []
    his_rows: list[tuple[str, int, dict[int, tuple[int, int, int, float]]]] = []
    for setup, th, series in launches:
        if setup == "H":
            for t, csvs in hector_gemm_report.load(series).items():
                for csv in csvs:
                    his_rows.append((setup, t, csv))
            continue
        for row in (series / "runs.tsv").read_text().splitlines()[1:]:
            launch, t, _c, _b, run, rc = row.split("\t")
            if rc != "0":
                continue
            run_dir = logs / run
            for label, v in ours_launch(run_dir, shapes).items():
                times[setup][int(t)].setdefault(label, []).append(v)
            st = [ln.split(",")[6] for f in run_dir.glob("all_raw_*.txt") for ln in f.read_text().splitlines()[1:] if ln]
            checks.append(f"{setup} {t}T {run}: {st.count('PASS')} of {len(st)} PASS")
    for setup, t, csv in his_rows:  # his row l = our label Y<l>; n = 1 rows have no label of ours
        for row, (m, n, k, secs) in csv.items():
            lb = f"Y{row:02d}"
            if lb in shapes and shapes[lb] == (m, n, k):
                times[setup][t].setdefault(lb, []).append((secs, None))
    return times, shapes, checks


def run_of(series: Path) -> str:
    """Our runner's run folder (runs.tsv), or '-' for H\'ector's (his CSV lives in the series folder)."""
    tsv = series / "runs.tsv"
    if not tsv.exists():
        return "-"
    r = tsv.read_text().splitlines()[1].split("\t")
    return r[4] if len(r) == 6 else "-"


def hector_ok(logs: Path, rows: list[list[str]]) -> list[str]:
    """'H 1T <series>: 53 of 53 OK' per H step; his log marks each row OK or ERROR in the last column."""
    out = []
    for _i, sv, t, d, _rc in rows:
        if sv != "H":
            continue
        for log in sorted((logs / d).glob("L*_T*_*.log")):
            cells = [ln.split("|") for ln in re.sub(r"\x1b\[[0-9;]*m", "", log.read_text()).splitlines()]
            marks = [c[3].strip() for c in cells if len(c) > 4 and re.search(r"\d", c[1])]
            out.append(f"H {t}T {d}: {marks.count('OK')} of {len(marks)} OK")
    return out


def ops(shapes: dict[str, Shape], labels: list[str]) -> float:
    return sum(2.0 * shapes[lb][0] * shapes[lb][1] * shapes[lb][2] for lb in labels)


def pass_s(times: dict[str, list[tuple[float, float | None]]], labels: list[str], which: int) -> float:
    return sum(statistics.fmean([x[which] for x in times[lb]]) for lb in labels)  # type: ignore[misc]


def launch_range(times: dict[str, list[tuple[float, float | None]]], shapes: dict[str, Shape], labels: list[str]) -> tuple[float, float]:
    """Lowest and highest one-pass GOPs by mean time over the launches (each launch is one full pass)."""
    n = min(len(times[lb]) for lb in labels)
    g = [ops(shapes, labels) / sum(times[lb][i][0] for lb in labels) / 1e9 for i in range(n)]
    return min(g), max(g)


COLOR = {"V": "vec", "S": "scal", "H": "his"}
LEGEND = {"V": "V: ours, vector packer", "S": "S: ours, scalar packer", "H": "H: H\\'ector's"}


def bar_fig(ylabels: list[str], series: dict[str, dict[int, float]], xlabel: str, note: str, one_row: bool = False) -> str:
    """One horizontal bar chart (one figure): bars per setup and y row, the value printed on each bar.
    one_row: each setup has its own y row (a single bar per row, no legend)."""
    xmax = max(v for vs in series.values() for v in vs.values()) * 1.18
    rows = len(ylabels) * (1 if one_row else len(series))
    order = list(reversed(series.items()))  # drawn H S V: under y dir=reverse, V then sits on top of each group
    plots = "\n".join(
        f"\\addplot[fill={COLOR[s]}, draw=none] coordinates {{" + " ".join(f"({v:.2f},{i})" for i, v in vs.items()) + "};"
        for s, vs in order)
    legend = "" if one_row else "\\legend{" + ",".join("{" + LEGEND[s] + "}" for s, _ in order) + "}"
    return (f"\\begin{{figure}}[htbp]\\centering\n\\begin{{tikzpicture}}\n"
            f"\\begin{{axis}}[house, xbar, bar width=7pt, width=12cm, height={1.2 + (0.7 if one_row else 0.42) * rows:.1f}cm, y dir=reverse, "
            f"ytick={{{','.join(str(i) for i in range(len(ylabels)))}}}, "
            f"yticklabels={{{','.join('{' + y + '}' for y in ylabels)}}}, "
            f"enlarge y limits={{abs=0.7}}, xmin=0, xmax={xmax:.1f}, xlabel={{{xlabel}}}, nodes near coords, "
            "nodes near coords style={font=\\scriptsize, anchor=west, /pgf/number format/fixed, /pgf/number format/precision=2, /pgf/number format/zerofill}, "
            "legend style={at={(0.5,1.02)}, anchor=south, legend columns=-1}"
            + (", bar shift=0pt" if one_row else ", reverse legend") + "]\n"
            f"{plots}\n{legend}\n\\end{{axis}}\n\\end{{tikzpicture}}\n\\note{{{note}}}\n\\end{{figure}}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("series", nargs="?", type=Path)
    ap.add_argument("--logs", type=Path, default=TESTS / "logs")
    ap.add_argument("--out", type=Path, default=None, help="default: data/, or check/ with --check (keeps data/ intact)")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    a.out = a.out or Path(__file__).resolve().parent / ("check" if a.check else "data")
    if a.check:
        rep = ROOT / ".scratch" / "reports"
        launches = [("V", 0, rep / "issue-56" / "parallel_gemm_2026-11-16_193132"),
                    ("H", 0, rep / "issue-62" / "hector_gemm_2026-11-16_111457")]
    else:
        steps = [r.split("\t") for r in (a.series / "steps.tsv").read_text().splitlines()[1:] if r.strip()]
        launches = [(s, int(t), a.logs / d) for _i, s, t, d, rc in steps if rc == "0"]
    times, shapes, checks = load(launches, a.logs)
    have = [s for s in SETUPS if all(times[s][t] for t in THREADS)]
    labels = sorted((lb for lb in shapes if all(lb in times[s][t] for s in have for t in THREADS)),
                    key=lambda lb: (shapes[lb][1], shapes[lb][0], shapes[lb][2], lb))
    total = ops(shapes, labels)
    nl = {s: {t: min(len(times[s][t][lb]) for lb in labels) for t in THREADS} for s in have}

    lines = [f"shapes: {len(labels)}; launches per setup and thread count: {nl}"]
    pass_rows, pass_csv = [], ["idx,label,g1,g8,l1,l8"]
    for i, s in enumerate(have):
        for which, metric in ((0, "by mean time"), (1, "by median rep")):
            if s == "H" and which == 1:
                continue
            t = {th: pass_s(times[s][th], labels, which) for th in THREADS}
            g = {th: total / t[th] / 1e9 for th in THREADS}
            rng = {th: launch_range(times[s][th], shapes, labels) for th in THREADS} if which == 0 else None
            r = (lambda th: f"{rng[th][0]:.2f}--{rng[th][1]:.2f}") if rng else (lambda th: "")
            pass_rows.append(f"{NAMES[s]}, {metric} & {t[1] * 1e3:.1f} & {g[1]:.2f} & {r(1)} & {t[8] * 1e3:.1f} & "
                             f"{g[8]:.2f} & {r(8)} & {t[1] / t[8]:.2f}$\\times$ \\\\")
            lines.append(f"{s} {metric}: 1T {t[1] * 1e3:.1f} ms {g[1]:.2f} GOPs | 8T {t[8] * 1e3:.1f} ms {g[8]:.2f} GOPs | {t[1] / t[8]:.2f}x")
            if which == 0:
                pass_csv.append(f"{i},{NAMES[s]},{g[1]:.2f},{g[8]:.2f},{g[1]:.2f},{g[8]:.2f}")

    # #91 quotes ours against Héctor's without Y00 too (his Y00 time swings between launches, #82 §2)
    nz = [lb for lb in labels if lb != "Y00"]
    wo = {s: {t: pass_s(times[s][t], nz, 0) for t in THREADS} for s in have}
    noy00_rows = [f"{NAMES[s]} & {wo[s][1] * 1e3:.1f} & {ops(shapes, nz) / wo[s][1] / 1e9:.2f} & {wo[s][8] * 1e3:.1f} & "
                  f"{ops(shapes, nz) / wo[s][8] / 1e9:.2f} & {wo[s][1] / wo[s][8]:.2f}$\\times$ \\\\" for s in have]
    y00 = {s: {t: [x[0] * 1e3 for x in times[s][t]["Y00"]] for t in THREADS} for s in have}
    lines += [f"{s} without Y00: 1T {wo[s][1] * 1e3:.1f} ms | 8T {wo[s][8] * 1e3:.1f} ms; Y00 ms per launch 1T "
              f"{' '.join(f'{v:.1f}' for v in y00[s][1])} 8T {' '.join(f'{v:.1f}' for v in y00[s][8])}" for s in have]

    ns = sorted({shapes[lb][1] for lb in labels})
    by_n_mean, by_n_med, by_n_csv = [], [], ["idx,label," + ",".join(f"{s}{t}" for s in have for t in THREADS)]
    for i, n in enumerate(ns):
        ls = [lb for lb in labels if shapes[lb][1] == n]
        o = ops(shapes, ls)
        gm = {(s, t): o / pass_s(times[s][t], ls, 0) / 1e9 for s in have for t in THREADS}
        cells = " & ".join(f"{gm[s, 1]:.2f} & {gm[s, 8]:.2f} & {gm[s, 8] / gm[s, 1]:.2f}$\\times$" for s in have)
        by_n_mean.append(f"{n} & {len(ls)} & {cells} \\\\")
        ours = [s for s in have if s != "H"]
        gd = {(s, t): o / pass_s(times[s][t], ls, 1) / 1e9 for s in ours for t in THREADS}
        by_n_med.append(f"{n} & {len(ls)} & " + " & ".join(f"{gd[s, 1]:.2f} & {gd[s, 8]:.2f} & {gd[s, 8] / gd[s, 1]:.2f}$\\times$" for s in ours) + " \\\\")
        by_n_csv.append(f"{i},n = {n} ({len(ls)})," + ",".join(f"{gm[s, t]:.2f}" for s in have for t in THREADS))
        lines.append(f"n={n}: " + " ".join(f"{s}{t}={gm[s, t]:.2f}" for s in have for t in THREADS))

    shape_rows = []
    for lb in labels:
        m, n, k = shapes[lb]
        g = {(s, t): 2.0 * m * n * k / statistics.fmean([x[0] for x in times[s][t][lb]]) / 1e9 for s in have for t in THREADS}
        shape_rows.append(f"{lb} & {m}$\\times${n}$\\times${k} & " + " & ".join(f"{g[s, t]:.2f}" for t in THREADS for s in have) + " \\\\")

    gpass = {s: {t: total / pass_s(times[s][t], labels, 0) / 1e9 for t in THREADS} for s in have}
    pass_figs = [bar_fig([LEGEND[s] for s in have], {s: {i: gpass[s][t]} for i, s in enumerate(have)},
                         f"GOPs by mean time, {where}, one pass over {len(labels)} shapes", "\\runs", one_row=True)
                 for t, where in ((1, "1T (core 4)"), (8, "8T (cores 0--7)"))]
    nfigs = [bar_fig([f"n = {n} ({sum(1 for lb in labels if shapes[lb][1] == n)})" for n in ns],
                     {s: {i: ops(shapes, [lb for lb in labels if shapes[lb][1] == n]) /
                             pass_s(times[s][t], [lb for lb in labels if shapes[lb][1] == n], 0) / 1e9 for i, n in enumerate(ns)} for s in have},
                     f"GOPs by mean time, {where}, one pass over the shapes with that n (shape count in brackets)", "\\runs")
             for t, where in ((8, "8T (cores 0--7)"), (1, "1T (core 4)"))]

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "pass_figs.tex").write_text("\n".join(pass_figs))
    (a.out / "by_n_figs.tex").write_text("\n".join(nfigs))
    if a.series:
        plan = (a.series / "plan.txt").read_text()
        cl = [ln for ln in plan.splitlines() if ln.startswith("[") and "PASS" in ln]
        rows = [r.split("\t") for r in (a.series / "steps.tsv").read_text().splitlines()[1:] if r.strip()]
        (a.out / "correctness.tex").write_text(
            "Before timing, tile 24x8 with the scalar packer and the scalar oracle on, on the small YOLO-like set "
            "(\\texttt{datasets/yolo\\_correctness\\_small.dat}, as \\iss{83} C1/C2), \\texttt{--tmin 0.2}:\n\\begin{itemize}\\setlength\\itemsep{0pt}\n"
            + "".join(f"\\item {ln.split('] ', 1)[1].replace('_', '\\_')}\n" for ln in cl)
            + "\\item Every timed launch of ours also checks all 50 shapes against the oracle:\n"
            "\\begin{itemize}\\setlength\\itemsep{0pt}\\footnotesize\n"
            + "".join(f"\\item {c.replace('_', '\\_')}\n" for c in checks) + "\\end{itemize}\n"
            "\\item H\\'ector's launches, his own check (\\texttt{TEST=T}), rows marked OK in his log:\n"
            "\\begin{itemize}\\setlength\\itemsep{0pt}\\footnotesize\n"
            + "".join(f"\\item {c.replace('_', '\\_')}\n" for c in hector_ok(a.logs, rows)) + "\\end{itemize}\n"
            "\\end{itemize}\n")
        (a.out / "series.tex").write_text("\\texttt{" + a.series.name.replace("_", "\\_") + "}")
        (a.out / "logs.tex").write_text(
            "Series folder \\texttt{tests/pure\\_xdsl\\_blis\\_rvv/logs/" + a.series.name.replace("_", "\\_")
            + "/} (plan, order, \\texttt{steps.tsv}); copies in \\texttt{.scratch/reports/issue-94/}. One row per launch:\n"
            "{\\footnotesize\\begin{longtable}{@{}rllll@{}}\\toprule step & setup & threads & runner's series folder & run folder \\\\ \\midrule\n"
            + "".join(f"{i} & {sv} & {t}T & \\texttt{{{d.replace('_', '\\_')}}} & \\texttt{{{run_of(a.logs / d).replace('_', '\\_')}}} \\\\\n" for i, sv, t, d, _rc in rows)
            + "\\bottomrule\\end{longtable}}\n")
        import datetime
        (a.out / "date.tex").write_text(datetime.date.today().strftime("%-d %b %Y"))
    for name, body in (("pass_rows.tex", pass_rows), ("by_n_mean_rows.tex", by_n_mean), ("by_n_median_rows.tex", by_n_med),
                       ("shape_rows.tex", shape_rows), ("noy00_rows.tex", noy00_rows), ("pass.csv", pass_csv), ("by_n.csv", by_n_csv),
                       ("checks.txt", checks), ("summary.txt", lines)):
        (a.out / name).write_text("\n".join(body) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
