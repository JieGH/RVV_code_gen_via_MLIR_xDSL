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


print("# Packing numbers, true GOPs by mean time (KerGOPs_Harmonic) and by median rep\n")
for test, (vd, sd) in RUNS83.items():
    (va, dims), (sa, _) = load(vd), load(sd)
    vb, sb = best(va), best(sa)
    rows = sorted(vb, key=lambda l: (dims[l][1], dims[l][0], dims[l][2], l))
    r_mean = [sb[l][1] / vb[l][1] for l in rows]
    r_med = [sb[l][2] / vb[l][2] for l in rows]
    changed = sum(vb[l][0] != sb[l][0] for l in rows)
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
        for r in (a, b):
            cells += [f"{r['vec']} / {r['vmed']}", f"{r['scal']} / {r['smed']}", f"{float(r['ratio']):.3f}"]
        f.write(" & ".join(cells) + " \\\\\n")
