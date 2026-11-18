#!/usr/bin/env python3
"""Issue #95 report data: why our GEMM trails Héctor's on YOLO at 1 thread (the A pack read order).

Usage (from the repo root or anywhere): python3 make_data.py [--out DIR]
Reads only committed logs:
  tests/pure_xdsl_blis_rvv/logs/issue95_probe_*        one-shape runs, ours then his, 1T core 4 (probe_issue95.sh)
  tests/pure_xdsl_blis_rvv/logs/issue95_pack_order_50_* pack test, both read orders, all 50 shapes (issue95/pack_order.c)
  .scratch/reports/issue-94/report/data/shape_rows.tex  #94 per-shape 1T GOPs (ours vector packer, his)
Writes LaTeX rows, figures and summary.txt to data/.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TESTS = ROOT / "tests" / "pure_xdsl_blis_rvv"
LOGS = TESTS / "logs"
sys.path.insert(0, str(TESTS))
from reps_report import case_stats, read_reps, reps_files  # noqa: E402

COLOR = {"gap": "his", "order": "vec", "ours": "vec", "his": "his", "po": "scal", "ph": "vec"}
LEGEND = {"gap": "1T gap: ours $-$ H\\'ector's (\\#94)", "order": "A pack: our order $-$ his order",
          "po": "A pack, our order", "ph": "A pack, his order"}


def tex(s: str) -> str:
    return s.replace("_", "\\_")


def bar_fig(ylabels: list[str], series: dict[str, dict[int, float]], xlabel: str, note: str, xmin: float = 0) -> str:
    """One horizontal bar chart (one figure), the value printed on each bar (as #94's make_data.bar_fig)."""
    xmax = max(v for vs in series.values() for v in vs.values()) * 1.2
    rows = len(ylabels) * len(series)
    order = list(reversed(series.items()))
    plots = "\n".join(f"\\addplot[fill={COLOR[s]}, draw=none] coordinates {{"
                      + " ".join(f"({v:.2f},{i})" for i, v in vs.items()) + "};" for s, vs in order)
    legend = "\\legend{" + ",".join("{" + LEGEND[s] + "}" for s, _ in order) + "}"
    return (f"\\begin{{figure}}[htbp]\\centering\n\\begin{{tikzpicture}}\n"
            f"\\begin{{axis}}[house, xbar, bar width=6pt, width=12cm, height={1.4 + 0.36 * rows:.1f}cm, y dir=reverse, "
            f"ytick={{{','.join(str(i) for i in range(len(ylabels)))}}}, yticklabels={{{','.join('{' + y + '}' for y in ylabels)}}}, "
            f"enlarge y limits={{abs=0.7}}, xmin={xmin:.1f}, xmax={xmax:.1f}, xlabel={{{xlabel}}}, nodes near coords, "
            "nodes near coords style={font=\\scriptsize, anchor=west, /pgf/number format/fixed, /pgf/number format/precision=1, /pgf/number format/zerofill}, "
            "legend style={at={(0.5,1.02)}, anchor=south, legend columns=-1}, reverse legend]\n"
            f"{plots}\n{legend}\n\\end{{axis}}\n\\end{{tikzpicture}}\n\\note{{{note}}}\n\\end{{figure}}\n")


def probes() -> list[dict]:
    out = []
    for d in sorted(LOGS.glob("issue95_probe_*")):
        res = (d / "result.txt").read_text().split()
        log = (d / "ours.log").read_text()
        blk = re.search(r"cache blocking mc:(\d+) nc:(\d+) kc:(\d+)", log)
        run = LOGS / (d / "ours_run.txt").read_text().strip()
        med = None
        for f in reps_files(run):
            for _key, c in read_reps(f).items():
                med = case_stats(c.ops, c.seconds).gops_median
        split = {}
        for name in ("vector pack", "scalar pack", "compute"):
            m = re.search(rf"\|\s+{name}\s+: ([\d.]+) ms", log)
            if m:
                split[name] = float(m.group(1))
            m = re.search(rf"\|\s+{name}\s+: IPC ([\d.]+) \| L1D miss ([\d.]+)% read", log)
            if m:
                split[name + " ipc"], split[name + " l1d"] = float(m.group(1)), float(m.group(2))
        out.append(dict(name=d.name, label=d.name.split("_", 4)[-1], shape=res[0], ours=float(res[4]), his=float(res[6]),
                        median=med, mc=blk.group(1), kc=blk.group(3), instr="--instr" in log and "INSTRUMENTED" in log
                        and bool(split), forced="RVV_FORCE_MC=" in log, split=split, run=run.name))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    # one-shape runs
    pr = probes()
    rows = []
    for p in pr:
        if p["split"]:
            continue
        rows.append(f"{tex(p['label'])} & {p['shape'].replace('x', '$\\times$')} & {p['mc']} & {p['kc']} & "
                    f"{'forced' if p['forced'] else 'model'} & {p['ours']:.2f} & {p['his']:.2f} \\\\")
        lines.append(f"probe {p['label']}: {p['shape']} mc {p['mc']} kc {p['kc']} ours {p['ours']:.2f} his {p['his']:.2f}")
    (a.out / "probe_rows.tex").write_text("\n".join(rows) + "\n")

    irows = []
    for p in pr:
        s = p["split"]
        if not s:
            continue
        share = s["vector pack"] / (s["vector pack"] + s["compute"]) * 100
        m, n, k = map(int, p["shape"].split("x"))
        irows.append(f"{p['shape'].replace('x', '$\\times$')} & {s['vector pack']:.2f} & {s['scalar pack']:.2f} & {s['compute']:.2f} & "
                     f"{share:.0f}\\% & {2 * m * n * k / s['compute'] / 1e6:.1f} & {s['vector pack ipc']:.2f} & {s['vector pack l1d']:.1f}\\% \\\\")
        lines.append(f"instr {p['shape']}: vector pack {s['vector pack']} scalar pack {s['scalar pack']} compute {s['compute']} share {share:.1f}%")
    (a.out / "instr_rows.tex").write_text("\n".join(irows) + "\n")

    # #94 per-shape 1T times beside the pack test
    sh = {}
    for ln in (ROOT / ".scratch/reports/issue-94/report/data/shape_rows.tex").read_text().splitlines():
        if "&" in ln:
            c = [x.strip() for x in ln.replace("\\\\", "").split("&")]
            m, n, k = map(int, c[1].split("$\\times$"))
            ops = 2 * m * n * k
            sh[c[0]] = dict(m=m, n=n, k=k, ours=ops / float(c[2]) / 1e6, his=ops / float(c[4]) / 1e6)
    pack_file = sorted(LOGS.glob("issue95_pack_order_50_*.txt"))[-1]
    for ln in pack_file.read_text().splitlines():
        lab = ln.split()[0]
        ms = re.findall(r"([\d.]+) ms", ln)
        blk = re.search(r"MC (\d+) KC (\d+)", ln)
        sh[lab].update(po=float(ms[0]), ph=float(ms[1]), mc=int(blk.group(1)), kc=int(blk.group(2)))
    labs = sorted(sh, key=lambda lb: -(sh[lb]["po"] - sh[lb]["ph"]))

    def grp(sel):
        ls = [lb for lb in sh if sel(sh[lb])]
        t = {x: sum(sh[lb][x] for lb in ls) for x in ("ours", "his", "po", "ph")}
        return len(ls), t
    grows = []
    for name, sel in (("all 50", lambda s: True), ("m $>$ 300", lambda s: s["m"] > 300),
                      ("\\quad k $\\le$ kc (A packed in one block of k)", lambda s: s["m"] > 300 and s["k"] <= s["kc"]),
                      ("\\quad k $>$ kc", lambda s: s["m"] > 300 and s["k"] > s["kc"]), ("m = 300", lambda s: s["m"] == 300)):
        cnt, t = grp(sel)
        grows.append(f"{name} & {cnt} & {t['ours']:.1f} & {t['his']:.1f} & {t['ours'] - t['his']:.1f} & {t['po']:.1f} & {t['ph']:.1f} & {t['po'] - t['ph']:.1f} \\\\")
        lines.append(f"group {name}: n {cnt} ours {t['ours']:.1f} his {t['his']:.1f} gap {t['ours'] - t['his']:.1f} po {t['po']:.1f} ph {t['ph']:.1f} cost {t['po'] - t['ph']:.1f}")
    (a.out / "group_rows.tex").write_text("\n".join(grows) + "\n")
    over = [lb for lb in sh if sh[lb]["po"] > sh[lb]["ours"]]
    lines.append(f"pack test above our whole call: {len(over)} {sorted(over)}")
    (a.out / "over.tex").write_text(", ".join(sorted(over)))
    (a.out / "shape_rows.tex").write_text("\n".join(
        f"{lb} & {s['m']}$\\times${s['n']}$\\times${s['k']} & {s['mc']} & {s['kc']} & {s['ours']:.2f} & {s['his']:.2f} & "
        f"{s['ours'] - s['his']:.2f} & {s['po']:.2f} & {s['ph']:.2f} & {s['po'] - s['ph']:.2f} \\\\"
        for lb in labs for s in [sh[lb]]) + "\n")

    big = [lb for lb in labs if sh[lb]["m"] > 300]
    figs = [
        bar_fig([f"{lb} {sh[lb]['m']}$\\times${sh[lb]['n']}$\\times${sh[lb]['k']}" for lb in big],
                {"gap": {i: sh[lb]["ours"] - sh[lb]["his"] for i, lb in enumerate(big)},
                 "order": {i: sh[lb]["po"] - sh[lb]["ph"] for i, lb in enumerate(big)}},
                "ms per call, 1 thread (core 4)",
                "The 33 shapes with m $>$ 300, sorted by the order cost. Gap: \\#94 times (GOPs by mean time, mean of 3 launches; "
                "ours uninstrumented, reference protocol, \\texttt{--tmin 2}, tile 24x8; his \\texttt{gemm.bin}, \\texttt{TIMIN=2.0}). "
                f"Order cost: \\texttt{{{tex(pack_file.name)}}}, our model's mc and kc, MR 24.",
                xmin=min(0.0, min(sh[lb]["ours"] - sh[lb]["his"] for lb in big) * 1.2)),
    ]
    sel = ["Y00", "Y05", "Y01", "Y10"]
    figs.append(bar_fig([f"{lb} A {sh[lb]['m']}$\\times${sh[lb]['k']}" for lb in sel],
                        {"po": {i: sh[lb]["po"] for i, lb in enumerate(sel)}, "ph": {i: sh[lb]["ph"] for i, lb in enumerate(sel)}},
                        "ms to pack all of A, 1 thread (core 4)",
                        f"Same bytes, same packed layout, same RVV copy; mc $\\times$ kc blocks as the GEMM (\\texttt{{{tex(pack_file.name)}}})."))
    (a.out / "figs.tex").write_text("\n".join(figs))
    (a.out / "pack_file.tex").write_text("\\texttt{" + tex(pack_file.name) + "}")
    (a.out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
