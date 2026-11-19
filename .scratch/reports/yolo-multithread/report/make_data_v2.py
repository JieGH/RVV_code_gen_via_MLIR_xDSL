#!/usr/bin/env python3
"""Combined YOLO report v2 data (#109): the best-single-tile re-search and the head-to-head series. Writes data_v2/.

Usage (after `issue-94/report/make_data.py <#109 series> --out ../../issue-109/report/data`):
  python3 make_data_v2.py --search-1t <yolo_sweep folder> --search-8t <yolo_sweep folder> --tiles 24x8,<8T tile>

Metric: per layer and launch, mean time per call (GOPs by mean time); per tile, the mean over launches of each
layer's time, summed over the layers: ms per pass (each layer once). The same as the #109 series tables.
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent.parent
TESTS = R.parent.parent / "tests" / "pure_xdsl_blis_rvv"
sys.path.insert(0, str(TESTS))
from search_report import collect  # noqa: E402
from make_data import cells, lid  # noqa: E402

OUT = HERE / "data_v2"
Shape = tuple[int, int, int]


def runs(sweep: Path) -> list[Path]:
    rows = [r.split("\t") for r in (sweep / "runs.tsv").read_text().splitlines()[1:] if r.strip()]
    assert rows and all(r[4] == "0" for r in rows), f"{sweep}: a launch failed"
    return [TESTS / "logs" / r[3] for r in rows]


def launch_times(sweep: Path) -> tuple[dict[str, Shape], dict[str, dict[str, list[float]]], list[str]]:
    """Shapes; tile -> layer -> ms per launch (in launch order); one PASS line per launch."""
    shapes, stats = collect(runs(sweep))
    ms: dict[str, dict[str, list[float]]] = {}
    for lb, by_tile in stats.items():
        ops = 2.0 * shapes[lb][0] * shapes[lb][1] * shapes[lb][2]
        for tile, sts in by_tile.items():
            ms.setdefault(tile, {})[lb] = [ops / s.gops_mean_time / 1e6 for s in sts]
    checks = []
    for rd in runs(sweep):
        st = [ln.split(",")[6] for f in rd.glob("all_raw_*.txt") for ln in f.read_text().splitlines()[1:] if ln]
        checks.append(f"{rd.name}: {st.count('PASS')} of {len(st)} PASS")
    return shapes, ms, checks


def ranking(shapes: dict[str, Shape], ms: dict[str, dict[str, list[float]]], labels: list[str]) -> list[dict]:
    """One row per tile, fastest pass first: mean pass, per-launch passes, launches won."""
    nl = min(len(v) for t in ms.values() for v in t.values())
    rows = [{"tile": t, "ms": sum(statistics.fmean(v[lb]) for lb in labels),
             "per": [sum(v[lb][i] for lb in labels) for i in range(nl)]} for t, v in ms.items()]
    ops = sum(2.0 * shapes[lb][0] * shapes[lb][1] * shapes[lb][2] for lb in labels)
    for r in rows:
        r["gops"] = ops / r["ms"] / 1e6
        r["won"] = sum(min(rows, key=lambda x: x["per"][i])["tile"] == r["tile"] for i in range(nl))
    return sorted(rows, key=lambda r: r["ms"])


def search(th: int, sweep: Path, old: Path, lines: list[str]) -> str:
    shapes, ms, checks = launch_times(sweep)
    all_l = sorted(shapes, key=lid)
    gemm = [lb for lb in all_l if shapes[lb][1] > 1]
    assert len(all_l) == 53 and len(gemm) == 50, (len(all_l), len(gemm))
    r53, r50 = ranking(shapes, ms, all_l), ranking(shapes, ms, gemm)
    p50 = {r["tile"]: r for r in r50}
    sec = old.read_text().split("## Best single tile over the dataset")[1].split("##")[0]  # the 19-tile ranking (#61 / #63)
    oldms = {m.group(1): float(m.group(2)) for m in re.finditer(r"^\| (\d+x\d+) \| ([\d.]+) \|", sec, re.M)}
    assert len(oldms) == 19, old
    oldrank = {t: i + 1 for i, t in enumerate(sorted(oldms, key=oldms.get))}
    # n = 1 layers (GEMV) left out: our design has no GEMV (user, 2026-11-19)
    best, nxt = r50[0], r50[1]
    tie = max(best["per"]) >= min(nxt["per"])  # launch ranges overlap
    nl = len(best["per"])
    lines += [f"{th}T re-search {sweep.name}: {nl} launches; best {best['tile']} {best['ms']:.1f} ms "
              f"(range {min(best['per']):.1f}-{max(best['per']):.1f}, won {best['won']}); next {nxt['tile']} {nxt['ms']:.1f} "
              f"(range {min(nxt['per']):.1f}-{max(nxt['per']):.1f}, won {nxt['won']}); gap {100 * (nxt['ms'] / best['ms'] - 1):.2f}%; "
              f"ranges overlap: {tie}"]
    lines += [f"  {i + 1}. {r['tile']} 50: {r['ms']:.1f} ms {r['gops']:.2f} GOPs [{' '.join(f'{x:.1f}' for x in r['per'])}] won {r['won']}"
              f" | old #{'61' if th == 1 else '63'} rank {oldrank[r['tile']]} {oldms[r['tile']]:.1f}"
              for i, r in enumerate(r50)]
    lines += [f"  check {c}" for c in checks]
    (OUT / f"search{th}_rows.tex").write_text("".join(
        f"{i + 1} & {r['tile']} & {r['ms']:.1f} & {r['gops']:.2f} & {min(r['per']):.1f}--{max(r['per']):.1f} & {r['won']} of {nl} \\\\\n"
        f"" for i, r in enumerate(r50)))
    (OUT / f"search{th}_checks.tex").write_text(", ".join(c.split(": ")[1] for c in checks))
    # chart: ms per pass over 50 layers with n $>$ 1, whisker = launch range, best tile in its own colour
    y = len(r50)
    coords = lambda keep: " ".join(f"({r['ms']:.1f},{i}) += ({max(r['per']) - r['ms']:.1f},0) -= ({r['ms'] - min(r['per']):.1f},0) [{r['ms']:.1f} ms, {r['won']} of {nl} launches]"
                                   for i, r in enumerate(r50) if keep(r["tile"]))
    xmin = min(min(r["per"]) for r in r50) * 0.985
    xmax = max(max(r["per"]) for r in r50) + (max(max(r["per"]) for r in r50) - xmin) * 0.9
    plot = lambda col, keep, lg: (f"\\addplot[only marks, mark=*, mark size=2.4pt, color={col}, nodes near coords, point meta=explicit symbolic, "
                                  f"error bars/.cd, x dir=both, x explicit, error bar style={{color={col}, line width=0.8pt}}] coordinates {{{coords(keep)}}};"
                                  f"\\addlegendentry{{{lg}}}")
    plots = [plot("muted", lambda t: t not in (best["tile"], "16x8", "24x8"), "other candidates"),
             plot("his", lambda t: t == "16x8", "16x8 (H\\'ector's tile size, in our GEMM)"),
             plot("sq", lambda t: t == best["tile"], f"{best['tile']} (shortest pass)")]
    if best["tile"] != "24x8":
        plots.insert(2, plot("vec", lambda t: t == "24x8", "24x8"))
    plots = [p for p in plots if "coordinates {}" not in p]
    where = "1T (core 4)" if th == 1 else "8T (cores 0--7)"
    return f"""\\begin{{figure}}[htbp]\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[house, width=11cm, height={1.6 + 0.75 * y:.1f}cm, y dir=reverse,
  ytick={{0,...,{y - 1}}}, yticklabels={{{','.join(r['tile'] for r in r50)}}}, enlarge y limits={{abs=0.6}},
  xmin={xmin:.0f}, xmax={xmax:.0f}, xlabel={{ms per pass over the 50 layers with n $>$ 1, {where}; whisker: fastest to slowest launch}},
  every node near coord/.append style={{font=\\scriptsize, anchor=west, xshift=4pt}},
  legend style={{at={{(0.5,1.02)}}, anchor=south, legend columns=-1}}]
{chr(10).join(plots)}
\\end{{axis}}
\\end{{tikzpicture}}
\\note{{\\iss{{109}} re-search, {where}: jbpi, ours only, vector packer, uninstrumented, reference protocol, \\texttt{{--tmin 2}}, \\texttt{{--no-nc-overrides}}, {nl} launches with the tile order rotated, every row checked by the oracle. Dot: mean over launches of each layer's mean time, summed over the 50 layers with n $>$ 1 (each layer once). Label: that pass, then how many launches the tile was fastest in.}}
\\caption{{Best single tile, {th}T: the {y} re-searched tiles, shortest pass at the top.}}\\label{{fig:search{th}}}
\\end{{figure}}
"""


def blk_of(run: Path) -> dict[str, str]:
    """layer -> 'mc/kc' from one run's console summary."""
    txt = "".join(f.read_text() for f in run.glob("all_summary_*.txt"))
    return {m.group(1): f"{m.group(2)}/{m.group(3)}" for m in re.finditer(r"▶ (Y\d+) .*?mc:(\d+) nc:\d+ kc:(\d+)", txt)}


def shape_figs(data: Path, series: Path, tiles: dict[int, str]) -> str:
    """Per-layer V/S/H bars at each thread count, 53 layers in Héctor's order, our tile per thread count."""
    rows = sorted(cells(data / "shape_rows.tex"), key=lambda r: lid(r[0]))
    assert len(rows) == 53, len(rows)
    steps = [r.split("\t") for r in (series / "steps.tsv").read_text().splitlines()[1:] if r.strip()]
    first = {t: next(TESTS / "logs" / (series.parent / d / "runs.tsv").read_text().splitlines()[1].split("\t")[4]
                     for _i, s, th, d, _rc in steps if s == "V" and int(th) == t) for t in (1, 8)}
    out = []
    for th, cols, where in ((8, (5, 6, 7), "8T (cores 0--7)"), (1, (2, 3, 4), "1T (core 4)")):
        blk = blk_of(first[th])
        xmax = max(float(r[c]) for r in rows for c in cols) * 1.3
        for chunk in (rows[:27], rows[27:]):
            labels, bars = [], {c: [] for c in cols}
            for i, r in enumerate(chunk):
                n = int(r[1].split("$\\times$")[1])
                if n == 1:  # GEMV layer: left out, our design has no GEMV (user, 2026-11-19)
                    labels.append(f"{{{r[0]} {r[1]}\\\\[-1pt]\\tiny n = 1 (GEMV), left out}}")
                    continue
                mc, kc = blk[r[0]].split("/")
                labels.append(f"{{{r[0]} {r[1]}\\\\[-1pt]\\tiny mc {mc} nc {n} kc {kc}}}")
                for c, t in zip(cols, (tiles[th], tiles[th], {1: "GEMV", 4: "16x4"}.get(n, "16x8"))):
                    bars[c].append(f"({r[c]},{i}) [{t} {r[c]}]")
            plots = "\n".join(f"\\addplot[fill={col}, draw=none] coordinates {{{' '.join(bars[c])}}};"
                              for c, col in zip(reversed(cols), ("his", "scal", "vec")))
            out.append(f"""\\begin{{figure}}[p]\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[house, xbar, bar width=3.2pt, width=11.5cm, height={0.84 * len(chunk):.1f}cm, y dir=reverse,
  ytick={{0,...,{len(chunk) - 1}}}, yticklabels={{{','.join(labels)}}}, yticklabel style={{align=right, font=\\scriptsize}},
  enlarge y limits={{abs=0.6}}, xmin=0, xmax={xmax:.0f},
  xlabel={{GOPs by mean time, {where}; layers {chunk[0][0]}--{chunk[-1][0]}, in H\\'ector's layer order}},
  nodes near coords, point meta=explicit symbolic, every node near coord/.append style={{font=\\tiny, anchor=west}},
  legend style={{at={{(0.5,1.01)}}, anchor=south, legend columns=-1}}, reverse legend]
{plots}
\\legend{{{{H: H\\'ector's}},{{S: ours, scalar packer}},{{V: ours, vector packer}}}}
\\end{{axis}}
\\end{{tikzpicture}}
\\note{{Row label: H\\'ector's layer ID, m$\\times$n$\\times$k, then our cache blocking for {tiles[th]} at {th}T (V and S share it; from the run log). Bar label: tile, then GOPs. H\\'ector's blocking is fixed, MC = KC = 256, NC = n; n = 1 layers (GEMV) left out. Run settings: first table of \\S\\ref{{sec:109}}.}}
\\end{{figure}}
""")
    return "\n".join(out)


SWEEPS = {1: ["run_2026-11-15_220501", "run_2026-11-15_231549", "run_2026-11-16_002638"],  # #61, 1T
          8: ["run_2026-11-16_222551", "run_2026-11-16_234245", "run_2026-11-17_005337"]}  # #63, 8T
STOPS = ("Y09", "Y19", "Y29", "Y39", "Y52")


def cum_rows(th: int, single: str, lines: list[str]) -> str:
    """Accumulated work and time over the layers in Héctor's order: best tile per layer, our best single tile,
    Héctor's tile (16x8; 16x4 at n = 4) in our GEMM. Per layer and tile: mean over the 3 launches of RVV_Time_ms."""
    import csv
    ms: dict[str, dict[str, list[float]]] = {}
    shp: dict[str, Shape] = {}
    for r in SWEEPS[th]:
        f = TESTS / "logs" / r / "all_raw_fp32_AXPY_FP32_Pack_noinstr_refproto_no_nc_override.txt"
        for row in csv.DictReader(f.open()):
            assert row["Status"] in ("PASS", "SKIP"), (r, row["Layer"])  # #61/#63 ran without the oracle (§1.8)
            t = row["Variant"].replace("MR", "").replace("_NR", "x")
            ms.setdefault(row["Layer"], {}).setdefault(t, []).append(float(row["RVV_Time_ms"]))
            shp[row["Layer"]] = (int(row["M"]), int(row["N"]), int(row["K"]))
    assert len(ms) == 50 and all(len(v) == 3 for d in ms.values() for v in d.values()), th
    gop = best = one = his = 0.0
    wins = n = 0
    out = []
    for y in sorted(ms, key=lambda s: int(s[1:])):
        mean = {t: statistics.mean(v) for t, v in ms[y].items()}
        m, nn, k = shp[y]
        h = mean["16x4" if nn == 4 else "16x8"]
        gop += 2e-9 * m * nn * k
        best += min(mean.values()); one += mean[single]; his += h
        wins += mean[single] < h
        n += 1
        if int(y[1:]) >= int(STOPS[len(out)][1:]):
            g = lambda t: gop / t * 1e3  # noqa: E731
            last = len(out) == len(STOPS) - 1
            row = (f"Y00--{y} & {n} & {gop:.2f} & {g(his):.2f} & {g(one):.2f} & {g(best):.2f} & "
                   f"{his / one:.2f}$\\times$ & {his / best:.2f}$\\times$ & {wins} of {n} \\\\")
            out.append(("\\midrule\n" if last else "") + (row.replace(f"Y00--{y}", "\\textbf{all 50}") if last else row))
    lines.append(f"{th}T accumulated: work {gop:.3f} GOP; best per layer {best:.1f} ms; {single} {one:.1f} ms; "
                 f"Hector's tile {his:.1f} ms; {single} faster on {wins} of {n}")
    return "\n".join(out) + "\n"


def tile_rows(fixed: tuple[str, str] = ("24x8", "32x7")) -> str:
    """Appendix: the table behind the §1.5 / §1.6 per-layer charts (same source as make_data.tile_figs):
    per layer, our best of 19 tiles at 1T and 8T (GOPs, launch range) and Héctor's tile in our GEMM."""
    from make_data import UNTIMED
    srcs = [TESTS / "logs/yolo_sweep_2026-11-15_205445/tile_search_report.md",  # #61, 1T
            R / "issue-63/report/tile_search_8t_3launches.md"]  # #63, 8T
    rows: list[dict[str, list[str]]] = []
    grids: list[dict[str, dict[str, str]]] = []  # layer -> tile -> GOPs, from the "GOPs grid" section
    for src in srcs:
        txt = src.read_text().split("## GOPs grid")[1]
        head = [x.strip() for x in txt.split("| shape |")[1].split("\n")[0].strip().strip("|").split("|")][3:]
        g = {}
        for ln in txt.splitlines():
            c = [x.strip().strip("*") for x in ln.strip().strip("|").split("|")]
            if c and re.fullmatch(r"Y\d\d", c[0]):
                g[c[0]] = dict(zip(head, c[4:]))
        assert len(g) == 50 and all(len(v) == 19 for v in g.values()), src
        grids.append(g)
        r: dict[str, list[str]] = {}
        for ln in src.read_text().splitlines():
            c = [x.strip() for x in ln.strip().strip("|").split("|")]
            if re.fullmatch(r"Y\d\d", c[0]) and c[0] not in r and len(c) == 11:
                r[c[0]] = c
        assert len(r) == 50, src
        rows.append(r)
    out = []
    for y in sorted(set(rows[0]) | set(UNTIMED), key=lambda s: int(s[1:])):
        if y in UNTIMED:
            out.append(f"{y} & {UNTIMED[y]} & \\multicolumn{{10}}{{l}}{{\\textit{{n = 1 (GEMV), left out}}}} \\\\")
            continue
        _, m, n, k, *_ = rows[0][y]
        cells = []
        for r, grid, ft in zip(rows, grids, fixed):
            bt, g, ct, cg = r[y][5], r[y][6], r[y][9], r[y][10]
            gv = g.split()[0]
            assert grid[y][bt] == gv and grid[y][ct] == cg, (y, bt, gv, grid[y][bt])  # the grid agrees with the summary rows
            cells.append(f"{bt} & {gv} & {grid[y][ft]} & {ct} & {cg}")
        out.append(f"{y} & {m}$\\times${n}$\\times${k} & " + " & ".join(cells) + " \\\\")
    tot = []
    for f in ("cum1_rows.tex", "cum8_rows.tex"):  # last row: all 50; columns: his tile, single tile, best per layer
        c = [x.strip() for x in (OUT / f).read_text().strip().splitlines()[-1].split("&")]
        his, one, best = float(c[3]), float(c[4]), float(c[5])
        tot.append(f"per layer & {best:.2f} & {one:.2f} & & {his:.2f}")
    out.append("\\midrule\n\\textbf{all 50} & accumulated (\\S\\ref{sec:1t}, \\S\\ref{sec:8t}) & " + " & ".join(tot) + " \\\\")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-1t", type=Path, required=True)
    ap.add_argument("--search-8t", type=Path, required=True)
    ap.add_argument("--series", type=Path, help="the #109 issue94_* series folder (per-layer charts)")
    ap.add_argument("--tiles", default="24x8,16x12", help="our tile at 1T,8T in the series")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    lines: list[str] = []
    t1, t8 = a.tiles.split(",")
    (OUT / "cum1_rows.tex").write_text(cum_rows(1, t1, lines))
    (OUT / "cum8_rows.tex").write_text(cum_rows(8, t8, lines))
    (OUT / "tile_rows.tex").write_text(tile_rows())
    figs =[search(1, a.search_1t, R / "issue-61/sweep_final_2026-11-15/tile_search_report.md", lines),
            search(8, a.search_8t, R / "issue-63/report/tile_search_8t_3launches.md", lines)]
    (OUT / "search_figs.tex").write_text("\n".join(figs))
    if a.series:
        t1, t8 = a.tiles.split(",")
        (OUT / "by_n_rows.tex").write_text("".join(l for l in (R / "issue-109/report/data/by_n_mean_rows.tex").read_text().splitlines(True) if not l.startswith("1 &")))
        (OUT / "shape_figs.tex").write_text(shape_figs(R / "issue-109/report/data", a.series, {1: t1, 8: t8}))
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
