#!/usr/bin/env python3
"""Combined YOLO report data: GOPs by mean time only (the user's choice, 2026-11-18).

Usage: python3 make_data.py   (run after the issue-91 make_data.py)
Reads the issue folders' data and writes mean-time-only copies to data/:
  ../../issue-91/deck/data/{paper_sv,yolo_sv_rows}.tex, ../../issue-91/deck/p85/data/tab_*.tex
                                                   "mean / median" cells -> mean (the ~ mark stays)
"""
import csv
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent.parent
OUT = HERE / "data"
PAIR = re.compile(r"(\d+\.\d+) / \d+\.\d+(\$\^\{\\sim\}\$)?")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    srcs = [R / "issue-91/deck/data/paper_sv.tex", R / "issue-91/deck/data/yolo_sv_rows.tex",
            *sorted((R / "issue-91/deck/p85/data").glob("tab_*.tex"))]
    for src in srcs:
        s = PAIR.sub(lambda m: m.group(1) + (m.group(2) or ""), src.read_text())
        assert " / " not in s, src
        (OUT / src.name).write_text(s)
    print(f"wrote {len(srcs)} files to {OUT}")
    paper_tiles()
    shape_figs()
    layer_order()
    tile_figs()
    t4_fig()


# Héctor's n = 1 layers: run on his GEMV, left out of our results; shown as empty rows (user, 2026-11-18).
UNTIMED = {"Y06": "4800$\\times$1$\\times$9", "Y21": "1200$\\times$1$\\times$9", "Y35": "300$\\times$1$\\times$9"}


def lid(label: str) -> int:
    return int(label.split()[0][1:])


def layer_rows(src: Path, dst: Path, full: bool) -> None:
    """Table rows in Héctor's layer order; full tables get empty rows for the untimed layers."""
    rows = cells(src)
    ncol = len(rows[0])
    if full:
        assert len(rows) == 50 and not set(UNTIMED) & {r[0] for r in rows}, src
        rows += [[k, v] + [""] * (ncol - 2) for k, v in UNTIMED.items()]
    rows.sort(key=lambda r: lid(r[0]))
    dst.write_text("".join(" & ".join(r) + " \\\\\n" for r in rows))


def reorder_fig(fig: str) -> str:
    """Permute one pgfplots bar chart (yticklabels + coordinates) into Héctor's layer order."""
    m = re.search(r"yticklabels=\{(\{.*?\})\}, ", fig)
    labels = re.findall(r"\{([^{}]*)\}", m.group(1))
    order = sorted(range(len(labels)), key=lambda i: lid(labels[i]))
    new = {old: i for i, old in enumerate(order)}
    fig = fig.replace(m.group(1), ",".join("{" + labels[i] + "}" for i in order))
    return re.sub(r"\(([-\d.]+),(\d+)\)", lambda c: f"({c.group(1)},{new[int(c.group(2))]})", fig)


def layer_order() -> None:
    """Every per-layer YOLO table and chart in Héctor's layer order (Y00..Y52)."""
    layer_rows(OUT / "yolo_sv_rows.tex", OUT / "yolo_sv_rows.tex", True)
    layer_rows(R / "issue-91/deck/data/yolo_blk_rows.tex", OUT / "yolo_blk_rows.tex", True)
    layer_rows(R / "issue-95/report/data/shape_rows.tex", OUT / "shape95_rows.tex", True)
    figs = (R / "issue-95/report/data/figs.tex").read_text().split("\\begin{figure}")
    (OUT / "figs95.tex").write_text("\\begin{figure}".join([figs[0]] + [reorder_fig(f) for f in figs[1:]])
                                    .replace("sorted by the order cost", "in H\\'ector's layer order"))
    lines = (R / "issue-91/deck/data/t4_time.csv").read_text().splitlines()
    body = [ln.split(",", 1) for ln in lines[2:]]
    yolo = sorted((b for b in body if b[1].startswith("Y")), key=lambda b: lid(b[1]))
    rest = [b for b in body if not b[1].startswith("Y")]
    (OUT / "t4_time.csv").write_text("\n".join(lines[:2] + [f"{i},{b[1]}" for i, b in enumerate(yolo + rest)]) + "\n")
    print("wrote layer-ordered yolo_sv_rows, yolo_blk_rows, shape95_rows, figs95, t4_time")


def paper_tiles() -> None:
    """Part 2 chart data with Square shapes written in full (1000^3 -> 1000x1000x1000)."""
    s = (R / "issue-91/deck/data/paper_tiles.csv").read_text()
    s, k = re.subn(r"(\d+)\\textsuperscript\{3\}", r"\1$\\times$\1$\\times$\1", s)
    assert k == 5, k
    (OUT / "paper_tiles.csv").write_text(s)


def cells(path: Path) -> list[list[str]]:
    return [[c.strip() for c in line.rstrip().removesuffix("\\\\").split("&")]
            for line in path.read_text().splitlines() if "&" in line]


def shape_figs() -> None:
    """§1.2 per shape: one row per YOLO shape (m x n x k + our blocking), bars V/S/H with tile and GOPs.

    GOPs from #94 shape_rows.tex (cols V1 S1 H1 V8 S8 H8); our 24x8 mc/kc from issue-91 yolo_blk_rows.tex.
    """
    rows = sorted(cells(R / "issue-94/report/data/shape_rows.tex"), key=lambda r: int(r[0][1:]))  # Héctor's layer ID
    blk = {r[0]: r[2] for r in cells(R / "issue-91/deck/data/yolo_blk_rows.tex")}
    assert len(rows) == 50 and set(blk) == {r[0] for r in rows}
    rows = sorted(rows + [[k, v] for k, v in UNTIMED.items()], key=lambda r: lid(r[0]))
    out = []
    for th, cols, xmax in (("8T (cores 0--7)", (5, 6, 7), 62), ("1T (core 4)", (2, 3, 4), 13)):
        for chunk in (rows[:27], rows[27:]):
            labels, bars = [], {c: [] for c in cols}
            for i, r in enumerate(chunk):
                if r[0] in UNTIMED:  # empty row, data later
                    labels.append(f"{{{r[0]} {r[1]}\\\\[-1pt]\\tiny n = 1 (GEMV), left out}}")
                    continue
                n = int(r[1].split("$\\times$")[1])
                mc, kc = blk[r[0]].split("/")
                labels.append(f"{{{r[0]} {r[1]}\\\\[-1pt]\\tiny mc {mc} nc {n} kc {kc}}}")
                tiles = ("24x8", "24x8", "16x4" if n == 4 else "16x8")
                for c, t in zip(cols, tiles):
                    bars[c].append(f"({r[c]},{i}) [{t} {r[c]}]")
            plots = "\n".join(
                f"\\addplot[fill={col}, draw=none] coordinates {{{' '.join(bars[c])}}};"
                for c, col in zip(reversed(cols), ("his", "scal", "vec")))
            out.append(f"""\\begin{{figure}}[p]\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[house, xbar, bar width=3.2pt, width=11.5cm, height={0.84 * len(chunk):.1f}cm, y dir=reverse,
  ytick={{0,...,{len(chunk) - 1}}}, yticklabels={{{','.join(labels)}}}, yticklabel style={{align=right, font=\\scriptsize}},
  enlarge y limits={{abs=0.6}}, xmin=0, xmax={xmax},
  xlabel={{GOPs by mean time, {th}; layers {chunk[0][0]}--{chunk[-1][0]}, in H\\'ector's layer order}},
  nodes near coords, point meta=explicit symbolic, every node near coord/.append style={{font=\\tiny, anchor=west}},
  legend style={{at={{(0.5,1.01)}}, anchor=south, legend columns=-1}}, reverse legend]
{plots}
\\legend{{{{H: H\\'ector's}},{{S: ours, scalar packer}},{{V: ours, vector packer}}}}
\\end{{axis}}
\\end{{tikzpicture}}
\\note{{Row label: H\\'ector's layer ID (Y06, Y21, Y35: n = 1 (GEMV), left out), m$\\times$n$\\times$k, then our cache blocking for 24x8 (V and S share it; from the run logs). Bar label: tile, then GOPs. H\\'ector's blocking is fixed, MC = KC = 256, NC = n. \\runs}}
\\end{{figure}}
""")
    (OUT / "shape_figs.tex").write_text("\n".join(out))
    print("wrote paper_tiles.csv, shape_figs.tex")


def log_blk(y: str, tile: str) -> str:
    """mc/kc of one layer and tile from the #61 launch-1 console log (for tiles Appendix C does not list)."""
    log = (R.parent.parent / "tests/pure_xdsl_blis_rvv/logs/yolo_sweep_2026-11-15_205445/jbpi_L1.log").read_text()
    mr, nr = tile.split("x")
    m = re.search(rf"\[{y} \([^)]*\)\] Testing variant \d+/\d+ : MR{mr}_NR{nr}\b.*?mc=(\d+) rows.*?kc=(\d+) depth", log, re.S)
    assert m, (y, tile)
    return f"{m.group(1)}/{m.group(2)}"


def tile_figs() -> None:
    """§1.3 / §1.4 per layer: best of our 19 tiles (launch-range whisker) and Héctor's tile in our GEMM.

    GOPs from the #61 (1T) and #63 (8T) tile-search reports; blocking from data/yolo_blk_rows.tex (Appendix C).
    """
    repo = R.parent.parent
    srcs = {"1t": ("1T (core 4)", "b1", repo / "tests/pure_xdsl_blis_rvv/logs/yolo_sweep_2026-11-15_205445/tile_search_report.md",
                   "\\iss{61}: jbpi, core 4"),
            "8t": ("8T (cores 0--7)", "b8", R / "issue-63/report/tile_search_8t_3launches.md", "\\iss{63}: jbpi, cores 0--7")}
    blk = {}
    for c in cells(OUT / "yolo_blk_rows.tex"):
        if c[2]:
            blk[c[0]] = dict(t168=c[3], b1=c[4].split(), b8=c[5].split(), known=dict(x.split() for x in c[4:8]))
    for key, (th, bk, src, who) in srcs.items():
        rows = {}
        for ln in src.read_text().splitlines():
            c = [x.strip() for x in ln.strip().strip("|").split("|")]
            if re.fullmatch(r"Y\d\d", c[0]) and c[0] not in rows and len(c) == 11:
                rows[c[0]] = c
        assert set(rows) == set(blk), src
        ids = sorted(list(rows) + list(UNTIMED), key=lambda y: int(y[1:]))
        xmax = max(max(float(r[6].split()[0]), float(r[10])) for r in rows.values()) * 1.55
        out = []
        for chunk in (ids[:27], ids[27:]):
            labels, best, his = [], [], []
            for i, y in enumerate(chunk):
                if y in UNTIMED:
                    labels.append(f"{{{y} {UNTIMED[y]}\\\\[-1pt]\\tiny n = 1 (GEMV), left out}}")
                    continue
                _, m, n, k, _, bt, g, *_r, ct, cg = rows[y]
                tb = blk[y][bk]
                assert tb[0] == bt, (y, bt, tb)
                mc, kc = tb[1].split("/")
                cb = blk[y]["t168"] if ct == "16x8" else blk[y]["known"].get(ct) or log_blk(y, ct)
                cmc, ckc = cb.split("/")
                gv, lo, hi = re.fullmatch(r"([\d.]+) \(([\d.]+)-([\d.]+)\)", g).groups()
                labels.append(f"{{{y} {m}$\\times${n}$\\times${k}}}")
                best.append(f"({gv},{i}) +- ({float(hi) - float(gv):.2f},0) -= ({float(gv) - float(lo):.2f},0) "
                            f"[{bt} mc={mc} nc={n} kc={kc}: {gv}]")
                his.append(f"({cg},{i}) [{ct} mc={cmc} nc={n} kc={ckc}: {cg}]")
            out.append(f"""\\begin{{figure}}[p]\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[house, xbar, bar width=4pt, width=11.5cm, height={0.8 * len(chunk):.1f}cm, y dir=reverse,
  ytick={{0,...,{len(chunk) - 1}}}, yticklabels={{{','.join(labels)}}}, yticklabel style={{align=right, font=\\scriptsize}},
  enlarge y limits={{abs=0.6}}, xmin=0, xmax={xmax:.1f},
  xlabel={{GOPs by mean time, {th}; layers {chunk[0]}--{chunk[-1]}, in H\\'ector's layer order}},
  nodes near coords, point meta=explicit symbolic, every node near coord/.append style={{font=\\tiny, anchor=west, xshift=2pt}},
  legend style={{at={{(0.5,1.01)}}, anchor=south, legend columns=-1}}]
\\addplot[fill=vec, draw=none, error bars/.cd, x dir=both, x explicit, error bar style={{muted}}] coordinates {{{' '.join(best)}}};
\\addplot[fill=his!60, draw=none] coordinates {{{' '.join(his)}}};
\\legend{{{{best of our 19 tiles (whisker: launch range)}},{{H\\'ector's tile (16x8; 16x4 at n = 4) in our GEMM}}}}
\\end{{axis}}
\\end{{tikzpicture}}
\\note{{Row: H\\'ector's layer ID, m$\\times$n$\\times$k (Y06, Y21, Y35: n = 1 (GEMV), left out). Bar label: tile, its cache blocking (from the run logs), GOPs by mean time, mean of 3 launches. {who}, uninstrumented, reference protocol, \\texttt{{--tmin 2}}, \\texttt{{--no-nc-overrides}}, 19 tiles, vector packer.}}
\\end{{figure}}
""")
        (OUT / f"tile_figs_{key}.tex").write_text("\n".join(out))
    print("wrote tile_figs_1t.tex, tile_figs_8t.tex")


def t4_fig() -> None:
    """§3.4 T4 chart: one GEMM call per packer, pack (blue/orange) + compute (grey), vector call = 100%; table row order."""
    rows = [r for r in csv.DictReader(l for l in open(OUT / 't4_time.csv') if not l.startswith('#'))]
    order = ['Y14', 'Y30', 'Y47', 'Y10', 'Y40', 'Y29', 'Y19', 'Y42', 'B1', 'B2', 'S1']  # = table order
    rows.sort(key=lambda r: order.index(r['label'].split()[0]))
    lab = ",".join("{" + r['label'].split('\\\\')[0].replace('x', '$\\times$') + "\\\\[-1pt]\\tiny "
                   + r['label'].split('\\\\')[1] + "}" for r in rows)
    tot = lambda r, p: float(r[p + 'pack']) + float(r[p + 'comp'])
    def bars(p, shift, name, col):
        full = " ".join(f"({tot(r, p):.2f},{i}) [{name}: {tot(r, p):.0f}\\%, pack {r[p + 'share']}\\%]" for i, r in enumerate(rows))
        pack = " ".join(f"({r[p + 'pack']},{i})" for i, r in enumerate(rows))
        ys = '-3.6pt' if shift.startswith('-') else '3.6pt'
        return (f"\\addplot[xbar, fill=comp, draw=none, bar shift={shift}, forget plot, nodes near coords, point meta=explicit symbolic,"
                f" every node near coord/.append style={{font=\\tiny, anchor=west, yshift={ys}}}] coordinates {{{full}}};\n"
                f"\\addplot[xbar, fill={col}, draw=none, bar shift={shift}, forget plot] coordinates {{{pack}}};")
    fig = rf"""\begin{{tikzpicture}}
    \begin{{axis}}[house, width=12.5cm, height=12.5cm, y dir=reverse, bar width=6.5pt,
      ytick={{0,...,{len(rows) - 1}}}, yticklabels={{{lab}}}, yticklabel style={{align=right, font=\scriptsize}},
      enlarge y limits={{abs=0.6}}, xmin=0, xmax=228, xtick={{0,25,...,150}}, xticklabel={{\pgfmathprintnumber{{\tick}}\%}},
      xlabel={{time of one GEMM call of that shape, as \% of the same call with the vector packer (top bar of each pair = 100\%)}},
      legend style={{at={{(0.5,1.01)}}, anchor=south, legend columns=-1, font=\scriptsize}}]
    \addlegendimage{{area legend, fill=vec, draw=none}}\addlegendentry{{packing A and B, vector packer}}
    \addlegendimage{{area legend, fill=scal, draw=none}}\addlegendentry{{packing A and B, scalar packer}}
    \addlegendimage{{area legend, fill=comp, draw=none}}\addlegendentry{{microkernel compute}}
    {bars('v', '3.6pt', 'vector packer', 'vec')}
    {bars('s', '-3.6pt', 'scalar packer', 'scal')}
    \draw[muted, dashed] (axis cs:100,-0.6) -- (axis cs:100,{len(rows) - 0.4});
    \end{{axis}}
    \end{{tikzpicture}}"""
    note = (r"\note{\iss{83} T4: jbpi2, instrumented build (timers cost 11--17\%, so only shares and ratios are used), warm-up protocol, "
            r"200 reps, 1T core 4, 2 launches per packer (V S S V); each shape on its 1T best tile. Row: shape, then the tile and cache blocking "
            r"that run used (from its log). Two bars per shape: top with the vector packer, bottom with the scalar packer. Each bar is one GEMM "
            r"call split into packing (A and B) and microkernel compute. Bar label: packer, call time as \% of the vector call, then the share "
            r"of that call spent packing. Rows in the order of the table below (by n).}")
    (OUT / "t4_fig.tex").write_text("\\definecolor{comp}{HTML}{B4B2A9}\n\\begin{center}\n" + fig + "\n" + note + "\n\\end{center}\n")
    print("wrote t4_fig.tex")


if __name__ == "__main__":
    main()
