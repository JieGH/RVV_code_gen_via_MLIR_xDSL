"""Issue #91 deck: packing numbers recomputed with GOPs by mean time = 2mnk / mean time per call (KerGOPs_Harmonic).

#83 and #85 read KerGOPs_Mean (the average of each rep's GOPs) and called it "by mean time". This script redoes their
numbers from the same committed runs with the true mean time, plus KerGOPs_Median (by median rep).
Usage (from the repo root): python3 .scratch/reports/issue-91/deck/make_data.py > .scratch/reports/issue-91/deck/data/packing_tables.md
Writes .scratch/reports/issue-91/deck/data/{yolo_ratio_1t,yolo_ratio_8t,paper_1t,paper_8t}.csv and #85's chart CSVs (p85_*.csv).
"""
import csv, glob, io, os, statistics as st, sys, contextlib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../../.."))
LOGS = os.path.join(REPO, "tests/pure_xdsl_blis_rvv/logs")
DATA = os.path.join(HERE, "data")
MEAN, MED = "KerGOPs_Harmonic", "KerGOPs_Median"
# Same runs as .scratch/reports/issue-83/README.md "Runs": (vector launches 1-3, scalar launches 1-3)
RUNS83 = {
    "yolo_1t": (["run_2026-11-15_220501", "run_2026-11-15_231549", "run_2026-11-16_002638"],
                ["run_2026-11-17_171901", "run_2026-11-17_182951", "run_2026-11-17_194103"]),
    "yolo_8t": (["run_2026-11-16_222551", "run_2026-11-16_234245", "run_2026-11-17_005337"],
                ["run_2026-11-17_133645", "run_2026-11-17_145239", "run_2026-11-17_160305"]),
    "paper_1t": (["run_2026-11-17_020326", "run_2026-11-17_043612", "run_2026-11-17_053929"],
                 ["run_2026-11-17_133114", "run_2026-11-17_160334", "run_2026-11-17_170643"]),
    "paper_8t": (["run_2026-11-17_024808", "run_2026-11-17_033254", "run_2026-11-17_062411"],
                 ["run_2026-11-17_141548", "run_2026-11-17_150024", "run_2026-11-17_175117"]),
}


def load(dirs):
    """(layer, tile) -> [(mean-time GOPs, median-rep GOPs) per launch]; layer -> (m, n, k)."""
    acc, dims = {}, {}
    for d in dirs:
        (f,) = glob.glob(os.path.join(LOGS, d, "all_raw_*.txt"))
        for r in csv.DictReader(open(f)):
            dims[r["Layer"]] = (int(r["M"]), int(r["N"]), int(r["K"]))
            acc.setdefault((r["Layer"], r["Variant"].replace("MR", "").replace("_NR", "x")), []).append(
                (float(r[MEAN]), float(r[MED])))
    return acc, dims


def best(acc):
    """layer -> (tile, GOPs by mean time, GOPs by median rep, launch min, launch max): best tile by mean time, avg over launches."""
    b = {}
    for (layer, tile), xs in acc.items():
        assert len(xs) == 3, (layer, tile, len(xs))
        m = st.mean(x[0] for x in xs)
        if layer not in b or m > b[layer][1]:
            b[layer] = (tile, m, st.mean(x[1] for x in xs), min(x[0] for x in xs), max(x[0] for x in xs))
    return b


ALL = {}
print("# Packing numbers, true GOPs by mean time (KerGOPs_Harmonic) and by median rep\n")
for test, (vd, sd) in RUNS83.items():
    (va, dims), (sa, _) = load(vd), load(sd)
    vb, sb = best(va), best(sa)
    rows = sorted(vb, key=lambda l: (dims[l][1], dims[l][0], dims[l][2], l))
    r_mean = [sb[l][1] / vb[l][1] for l in rows]
    r_med = [sb[l][2] / vb[l][2] for l in rows]
    changed = sum(vb[l][0] != sb[l][0] for l in rows)
    ALL[test] = (vb, sb, dims, rows)
    print(f"## {test}: {len(rows)} shapes; scalar/vector by mean time median {st.median(r_mean):.3f} "
          f"({min(r_mean):.3f}-{max(r_mean):.3f}); by median rep median {st.median(r_med):.3f} "
          f"({min(r_med):.3f}-{max(r_med):.3f}); best tile changed on {changed}\n")
    print("| shape | m x n x k | vector tile | vector mean / median | scalar tile | scalar mean / median | s/v mean | s/v median rep |")
    print("|---|---|---|---|---|---|---|---|")
    with open(os.path.join(DATA, f"{test}.csv"), "w") as f:
        f.write("# Made by make_data.py from the #83 runs; GOPs by mean time = KerGOPs_Harmonic, mean of 3 launches\n")
        f.write("idx,label,n,vec,vmed,scal,smed,ratio,ratio_med\n")
        for i, l in enumerate(rows):
            (m, n, k), v, s = dims[l], vb[l], sb[l]
            print(f"| {l} | {m}x{n}x{k} | {v[0]} | {v[1]:.2f} / {v[2]:.2f} | {s[0]} | {s[1]:.2f} / {s[2]:.2f} | "
                  f"{s[1] / v[1]:.3f} | {s[2] / v[2]:.3f} |")
            f.write(f"{i},{l},{n},{v[1]:.2f},{v[2]:.2f},{s[1]:.2f},{s[2]:.2f},{s[1] / v[1]:.4f},{s[2] / v[2]:.4f}\n")
    if test.startswith("yolo"):
        print("\n| n | shapes | s/v by mean time: median (min-max) | by median rep: median (min-max) |\n|---|---|---|---|")
        byn = []
        for n in sorted({dims[l][1] for l in rows}):
            rs = [sb[l][1] / vb[l][1] for l in rows if dims[l][1] == n]
            rd = [sb[l][2] / vb[l][2] for l in rows if dims[l][1] == n]
            print(f"| {n} | {len(rs)} | {st.median(rs):.3f} ({min(rs):.3f}-{max(rs):.3f}) | "
                  f"{st.median(rd):.3f} ({min(rd):.3f}-{max(rd):.3f}) |")
            byn.append((n, len(rs), st.median(rs), min(rs), max(rs)))
        with open(os.path.join(DATA, f"{test}_by_n.csv"), "w") as f:
            f.write("# Made by make_data.py: YOLO scalar/vector GOPs by mean time per n group, median with min-max whisker\n")
            f.write("idx,label,ratio,rlo,rhi\n")
            for i, (n, c, md, lo, hi) in enumerate(byn):
                f.write(f"{i},n={n} ({c}),{md:.4f},{md - lo:.4f},{hi - md:.4f}\n")
    print()

# #85: rerun its own script on its committed runs, with the mean-time column swapped in and output sent here.
src85 = os.path.join(REPO, ".scratch/reports/issue-85/report/make_tables_and_figures.py")
code = open(src85).read()
assert code.count('"KerGOPs_Mean"') == 1 and 'OUT = os.path.join(ROOT, "report")' in code
code = code.replace('"KerGOPs_Mean"', f'"{MEAN}"').replace('OUT = os.path.join(ROOT, "report")', "OUT = OUT85")
out85 = os.path.join(HERE, "p85")
sys.argv = [src85, os.path.join(REPO, ".scratch/reports/issue-85")]
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(code, src85, "exec"), g85 := {"__name__": "__main__", "OUT85": out85})
open(os.path.join(DATA, "p85_tables.md"), "w").write(
    "# #85 tables recomputed with GOPs by mean time = KerGOPs_Harmonic (make_data.py)\n" + buf.getvalue())
print("## #85 model check (true mean time)\n")
print("\n".join(l for l in buf.getvalue().splitlines() if l.startswith("model check")))

# Decision-table regions over the #85 points, scalar/vector GOPs by mean time (real = gap outside the launch range).
REGIONS = [
    ("n <= 16 and k <= 64 (any m, tile)", lambda m, n, k, t: n <= 16 and k <= 64),
    ("n <= 16, m <= 128", lambda m, n, k, t: n <= 16 and m <= 128),
    ("n <= 32, k >= 144, m >= 300", lambda m, n, k, t: n <= 32 and k >= 144 and m >= 300),
    ("n = 64", lambda m, n, k, t: n == 64),
    ("n = 256, m <= 300", lambda m, n, k, t: n == 256 and m <= 300),
    ("n >= 128, k = 144-256, m = 1200-2400, 24x8", lambda m, n, k, t: n >= 128 and 144 <= k <= 256 and 1200 <= m <= 2400 and t == "24x8"),
    ("n >= 128, k >= 576", lambda m, n, k, t: n >= 128 and k >= 576),
    ("n >= 128, m >= 4800", lambda m, n, k, t: n >= 128 and m >= 4800),
    ("n >= 128, all", lambda m, n, k, t: n >= 128),
]
print("\n## #85 decision-table regions (true mean time)\n")
print("| region | threads | points | s/v min-max | real | median-rep s/v min-max |\n|---|---|---|---|---|---|")
for name, f in REGIONS:
    for th in ("1t", "8t"):
        rs = [(k, g85["row"](th, k)) for k in g85["data"][th][0] if f(*k)]
        if not rs:
            continue
        r = [x["ratio"] for _, x in rs]
        rm = [x["ratio_med"] for _, x in rs]
        print(f"| {name} | {th} | {len(rs)} | {min(r):.3f}-{max(r):.3f} | {sum(x['real'] for _, x in rs)} | "
              f"{min(rm):.3f}-{max(rm):.3f} |")


# BERT/Square scalar/vector rows for the deck table: vector and scalar GOPs (mean time / median rep), s/v, 1T then 8T.
P = {th: list(csv.DictReader(l for l in open(os.path.join(DATA, f"paper_{th}.csv")) if not l.startswith("#")))
     for th in ("1t", "8t")}
with open(os.path.join(DATA, "paper_sv.tex"), "w") as f:
    for a, b in zip(P["1t"], P["8t"]):
        cells = [a["label"]]
        for r, th in ((a, "paper_1t"), (b, "paper_8t")):
            vt, st_ = ALL[th][0][r["label"]][0], ALL[th][1][r["label"]][0]
            cells += [f"{vt} {r['vec']} / {r['vmed']}", f"{st_} {r['scal']} / {r['smed']}", f"{float(r['ratio']):.3f}"]
        f.write(" & ".join(cells) + " \\\\\n")

# Per-layer YOLO rows for the report appendix: tile and GOPs (mean time / median rep) per packer, s/v, 1T then 8T.
with open(os.path.join(DATA, "yolo_sv_rows.tex"), "w") as f:
    v1, s1, dims, rows = ALL["yolo_1t"]
    v8, s8, _, _ = ALL["yolo_8t"]
    for l in rows:
        m, n, k = dims[l]
        cells = [l, f"{m}$\\times${n}$\\times${k}"]
        for vb, sb in ((v1, s1), (v8, s8)):
            cells += [f"{vb[l][0]} {vb[l][1]:.2f} / {vb[l][2]:.2f}", f"{sb[l][0]} {sb[l][1]:.2f} / {sb[l][2]:.2f}",
                      f"{sb[l][1] / vb[l][1]:.3f}"]
        f.write(" & ".join(cells) + " \\\\\n")

# Cache blocking (mc, nc, kc) for every chart label and table: read from the console logs (blocking.py).
from blocking import blk  # noqa: E402


def span(xs):
    lo, hi = min(xs), max(xs)
    return f"{lo}" if lo == hi else f"{lo}--{hi}"


def blk_range(layers, dims, tile):
    """'mc=a--b kc=c--d' over the layers at one tile; YOLO has nc = n on every layer (asserted)."""
    bs = [blk(*dims[l], tile) for l in layers]
    assert all(b[1] == dims[l][1] for b, l in zip(bs, layers)), tile
    return f"mc={span([b[0] for b in bs])} kc={span([b[2] for b in bs])}"


def readcsv(path):
    lines = open(path).read().splitlines()
    head = [l for l in lines if l.startswith("#")]
    return head, list(csv.DictReader(l for l in lines if not l.startswith("#")))


def writecsv(path, head, rows):
    with open(path, "w") as f:
        f.write("".join(h + "\n" for h in head))
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


v1, s1, ydims, yrows = ALL["yolo_1t"]
v8, s8, _, _ = ALL["yolo_8t"]
by_n = {}
for l in yrows:
    by_n.setdefault(ydims[l][1], []).append(l)
_, t1 = readcsv(os.path.join(DATA, "tile1t_by_n.csv"))
_, t8 = readcsv(os.path.join(DATA, "tile8t_by_n.csv"))
with open(os.path.join(DATA, "yolo_blk_by_n.csv"), "w") as f:
    f.write("# Made by make_data.py: cache blocking per n group and tile (logs via blocking.py); nc = n\n")
    f.write("idx,n,b2408,b168,b164,best1,b1,best8,b8\n")
    for i, (n, a, b) in enumerate(zip(sorted(by_n), t1, t8)):
        L = by_n[n]
        cells = [str(i), str(n)] + [blk_range(L, ydims, t) for t in ("24x8", "16x8", "16x4")]
        cells += [a["tile"], blk_range(L, ydims, a["tile"]), b["t8"], blk_range(L, ydims, b["t8"])]
        f.write(",".join(cells) + "\n")

# YOLO per-layer blocking table (appendix): vector best tile each thread count, plus 24x8 and 16x8 of Part 1; mc/kc (nc = n).
with open(os.path.join(DATA, "yolo_blk_rows.tex"), "w") as f:
    for l in yrows:
        m, n, k = ydims[l]
        cell = lambda t: "{}/{}".format(*blk(m, n, k, t)[::2])
        f.write(" & ".join([l, f"{m}$\\times${n}$\\times${k}", cell("24x8"), cell("16x8"),
                            f"{v1[l][0]} {cell(v1[l][0])}", f"{v8[l][0]} {cell(v8[l][0])}",
                            f"{s1[l][0]} {cell(s1[l][0])}", f"{s8[l][0]} {cell(s8[l][0])}"]) + " \\\\\n")

# Paper shapes: chart labels per tile, and a blocking table for every tile the Part 2 tables name.
head, prow = readcsv(os.path.join(DATA, "paper_tiles.csv"))
PDIMS = [ALL["paper_1t"][2][r["label"].split()[0]] for r in prow]
for r, d in zip(prow, PDIMS):
    for col, t in (("l168", "16x8"), ("l327", "32x7"), ("l1615", "16x15")):
        r[col] = "{} mc={} nc={} kc={}".format(t, *blk(*d, t))
writecsv(os.path.join(DATA, "paper_tiles.csv"), head, prow)
with open(os.path.join(DATA, "paper_blk_rows.tex"), "w") as f:
    for r, d in zip(prow, PDIMS):
        f.write(" & ".join([r["label"].split()[0], "$\\times$".join(map(str, d))]
                           + ["{}/{}/{}".format(*blk(*d, t)) for t in ("32x7", "16x15", "16x8", "16x12", "8x24")]) + " \\\\\n")

# #85 charts: right-margin label = tile and blocking, then pack share; dims from the matching table rows.
import re  # noqa: E402
DIMRE = re.compile(r"(\d+)\$\\times\$(\d+)\$\\times\$(\d+)(?:, (\d+x\d+))?")
for sweep in ("pn", "pk16", "pk256", "pm16", "pm256", "ptile"):
    pts = [DIMRE.search(l).groups() for l in open(os.path.join(out85, "data", f"tab_{sweep}.tex"))]
    for th in ("1t", "8t"):
        path = os.path.join(out85, "data", f"{sweep}_{th}.csv")
        head, rows = readcsv(path)
        assert len(rows) == len(pts), (sweep, th)
        for r, (m, n, k, t) in zip(rows, pts):
            t = t or "24x8"
            r["shares"] = r["shares"].replace("%", r"\%")  # % starts a TeX comment once the column is not the last
            r["blk"] = "{} mc={} nc={} kc={} \\textbar\\ {}".format(t, *blk(int(m), int(n), int(k), t), r["shares"])
        writecsv(path, head, rows)

# #83 T4 chart: shape, its T4 tile and blocking on the y label.
T4TILE = {"Y47": "16x4", "Y14": "24x4", "Y30": "32x4", "Y40": "24x8", "Y10": "24x8", "Y29": "32x4", "Y19": "32x7",
          "Y42": "24x8", "B2": "16x15", "B1": "32x7", "S1": "32x7"}
head, rows = readcsv(os.path.join(REPO, ".scratch/reports/issue-83/report/data/t4_time.csv"))
for r in rows:
    name, dims = r["label"].split("\\\\")[0].split()
    m, n, k = map(int, dims.split("x"))
    r["label"] = "{} {}\\\\{} mc={} nc={} kc={}".format(name, dims, T4TILE[name], *blk(m, n, k, T4TILE[name]))
writecsv(os.path.join(DATA, "t4_time.csv"), ["# Copy of issue-83/report/data/t4_time.csv; label carries the T4 tile and its blocking (make_data.py)"], rows)

# Part 1 chart labels: tile and blocking range per n group (ours); his GEMM uses fixed MC = KC = 256, NC = n.
for name, cols in (("engine_by_n.csv", (("lours", "24x8"),)),
                   ("tile1t_by_n.csv", (("lbest", None), ("l168", "16x8"), ("l164", "16x4")))):
    head, rows = readcsv(os.path.join(DATA, name))
    for r, n in zip(rows, sorted(by_n)):
        assert r["label"].startswith(f"n={n} "), (name, r["label"])
        for col, t in cols:
            t = t or r["tile"]
            r[col] = f"{t} {blk_range(by_n[n], ydims, t)}"
        if name == "engine_by_n.csv":
            r["lhis"] = "16x4" if n == 4 else "16x8"
    writecsv(os.path.join(DATA, name), head, rows)

# #85 tables: drop the bold on the faster packer; keep the ~ on gaps inside the launch range.
BOLD = re.compile(r"\\textbf\{([^}]*)\}(\$\^\{\\sim\}\$)?")
for sweep in ("pn", "pk16", "pk256", "pm16", "pm256", "ptile"):
    path = os.path.join(out85, "data", f"tab_{sweep}.tex")
    rows = []
    for line in open(path):
        c = line.rstrip().removesuffix("\\\\").rstrip().split(" & ")
        for a in (1, 4):
            for i, j in ((a, a + 1), (a + 1, a)):
                if mt := BOLD.fullmatch(c[i]):
                    c[i] = mt.group(1) + (mt.group(2) or "")
                    break
        rows.append(" & ".join(c) + " \\\\\n")
    assert not any("textbf" in r for r in rows), sweep
    open(path, "w").writelines(rows)
