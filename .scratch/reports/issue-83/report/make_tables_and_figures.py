"""Issue #83 final tables + figures: vector vs scalar packing. Run from tests/pure_xdsl_blis_rvv/logs.
argv[1] = output dir. Prints markdown tables to stdout."""
import csv, glob, re, statistics as st, sys

OUT = sys.argv[1]
RUNS = {  # test: (vector run dirs (launch 1-3), scalar run dirs (launch 1-3))
    "yolo1": (["run_2026-11-15_220501", "run_2026-11-15_231549", "run_2026-11-16_002638"], sys.argv[2].split(",")),
    "yolo8": (["run_2026-11-16_222551", "run_2026-11-16_234245", "run_2026-11-17_005337"],
              ["run_2026-11-17_133645", "run_2026-11-17_145239", "run_2026-11-17_160305"]),
    "paper1": (["run_2026-11-17_020326", "run_2026-11-17_043612", "run_2026-11-17_053929"],
               ["run_2026-11-17_133114", "run_2026-11-17_160334", "run_2026-11-17_170643"]),
    "paper8": (["run_2026-11-17_024808", "run_2026-11-17_033254", "run_2026-11-17_062411"],
               ["run_2026-11-17_141548", "run_2026-11-17_150024", "run_2026-11-17_175117"]),
}
TITLE = {"yolo1": "YOLO, 1 thread (core 4)", "yolo8": "YOLO, 8 threads (cores 0-7)",
         "paper1": "BERT and Square, 1 thread (core 4)", "paper8": "BERT and Square, 8 threads (cores 0-7)"}
HECTOR = (["hector_gemm_2026-11-16_222943", "hector_gemm_2026-11-16_223301", "hector_gemm_2026-11-17_005723"],
          ["hector_gemm_2026-11-17_133951", "hector_gemm_2026-11-17_134308", "hector_gemm_2026-11-17_160610"])
T4 = sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] else []  # instr run dirs, order vector, scalar, scalar, vector
dims = {}


def load(dirs):
    """(layer, tile) -> list over launches of (mean GOPs, median GOPs)."""
    acc = {}
    for d in dirs:
        fs = glob.glob(d + "/all_raw_*.txt")
        assert len(fs) == 1, d
        for r in csv.DictReader(open(fs[0])):
            dims[r["Layer"]] = (int(r["M"]), int(r["N"]), int(r["K"]))
            acc.setdefault((r["Layer"], r["Variant"].replace("MR", "").replace("_NR", "x")), []).append(
                (float(r["KerGOPs_Mean"]), float(r["KerGOPs_Median"])))
    return acc


def best(acc, nl):
    """layer -> (tile, mean GOPs, median GOPs, launch spread, per-launch means), best tile by average mean GOPs."""
    b = {}
    for (l, t), xs in acc.items():
        assert len(xs) == nl, (l, t, len(xs))
        m = st.mean(x[0] for x in xs)
        if l not in b or m > b[l][1]:
            ms = [x[0] for x in xs]
            b[l] = (t, m, st.mean(x[1] for x in xs), (max(ms) - min(ms)) / m, ms)
    return b


def order(l):
    return (l[0], int(re.sub(r"\D", "", l)))


data = {k: (best(load(v), len(v)), best(load(s), len(s))) for k, (v, s) in RUNS.items()}

# ---------- tables ----------
for k in RUNS:
    v, s = data[k]
    print(f"\n### {TITLE[k]}\n")
    print("| shape | m x n x k | vector tile | vector GOPs mean / median | scalar tile | scalar GOPs mean / median | scalar/vector mean | spread vector / scalar |")
    print("|---|---|---|---|---|---|---|---|")
    rm, rd, ch = [], [], 0
    for l in sorted(v, key=order):
        a, b = v[l], s[l]
        rm.append(b[1] / a[1]); rd.append(b[2] / a[2])
        same = a[0] == b[0]; ch += not same
        print(f"| {l} | {'x'.join(map(str, dims[l]))} | {a[0]} | {a[1]:.2f} / {a[2]:.2f} | {b[0]}{'' if same else ' *'} | "
              f"{b[1]:.2f} / {b[2]:.2f} | {rm[-1]:.3f} | {100*a[3]:.1f}% / {100*b[3]:.1f}% |")
    print(f"\nShapes {len(rm)}. scalar/vector by mean time: median {st.median(rm):.3f} (min {min(rm):.3f}, max {max(rm):.3f}); "
          f"by median rep: median {st.median(rd):.3f} (min {min(rd):.3f}, max {max(rd):.3f}). Best tile changed (*): {ch}. "
          f"Median launch spread: vector {100*st.median(x[3] for x in v.values()):.1f}%, scalar {100*st.median(x[3] for x in s.values()):.1f}%.")

# ---------- by reuse n ----------
print("\n### scalar/vector (mean) by reuse n\n")
print("| test | " + " | ".join(f"n={n}" for n in sorted({d[1] for d in dims.values()})) + " |")
ns_all = sorted({d[1] for d in dims.values()})
print("|---|" + "---|" * len(ns_all))
for k in RUNS:
    v, s = data[k]
    groups = {}
    for l in v:
        groups.setdefault(dims[l][1], []).append(s[l][1] / v[l][1])
    print(f"| {TITLE[k]} | " + " | ".join(f"{st.median(groups[n]):.3f} ({len(groups[n])})" if n in groups else "" for n in ns_all) + " |")

# ---------- Héctor day drift ----------
def hector(dirs):
    acc = {}
    for d in dirs:
        for row in open(glob.glob(d + "/*.csv")[0]):
            if row.startswith("#"):
                continue
            f = row.strip().split(";")
            acc.setdefault(int(f[0]), []).append(float(f[4]))
    return {l: st.mean(x) for l, x in acc.items()}


h16, h17 = hector(HECTOR[0]), hector(HECTOR[1])
hr = [h17[l] / h16[l] for l in h16]
print(f"\nHéctor gemm.x (unchanged), 17 Nov / 16 Nov per layer, mean of 3 launches: median {st.median(hr):.3f} "
      f"(min {min(hr):.3f}, max {max(hr):.3f}); sum GFLOPS {sum(h16.values()):.1f} -> {sum(h17.values()):.1f}")

# ---------- T4 pack split ----------
t4 = {}
if T4:
    print("\n### T4: pack/compute split (instrumented build, warm-up protocol, 200 reps, 1T core 4)\n")
    for d, packer in zip(T4, ["vector", "scalar", "scalar", "vector"]):
        for r in csv.DictReader(open(glob.glob(d + "/all_raw_*.txt")[0])):
            l = r["Layer"]  # each shape runs on all 6 tiles; keep its 1T vector best tile (#61 / #71)
            if r["Variant"].replace("MR", "").replace("_NR", "x") != (data["yolo1"] if l[0] == "Y" else data["paper1"])[0][l][0]:
                continue
            p, c = float(r["Vector_Pack_Time_ms"]), float(r["Compute_Time_ms"])
            t4.setdefault((r["Layer"], packer), []).append((p, c))
            dims.setdefault(r["Layer"], (int(r["M"]), int(r["N"]), int(r["K"])))
    print("| shape | m x n x k | tile | vector pack ms / compute ms | vector pack share | scalar pack ms / compute ms | scalar pack share | scalar/vector pack time |")
    print("|---|---|---|---|---|---|---|---|")
    for l in sorted({l for l, _ in t4}, key=order):
        row = []
        for p in ("vector", "scalar"):
            xs = t4[(l, p)]
            assert len(xs) == 2, (l, p, len(xs))
            pk, cp = st.mean(x[0] for x in xs), st.mean(x[1] for x in xs)
            row.append((pk, cp, pk / (pk + cp)))
        (vp, vc, vs), (sp, sc, ss) = row
        t4[l] = row
        tl = (data["yolo1"] if l[0] == "Y" else data["paper1"])[0][l][0]
        print(f"| {l} | {'x'.join(map(str, dims[l]))} | {tl} | {vp:.3f} / {vc:.3f} | {100*vs:.1f}% | {sp:.3f} / {sc:.3f} | {100*ss:.1f}% | {sp/vp:.2f} |")

# ---------- chart data: the brief draws every chart in LaTeX (pgfplots, vector), user 2026-11-17 ----------
import os
os.makedirs(f"{OUT}/data", exist_ok=True)
for f in glob.glob(f"{OUT}/data/*"):
    os.remove(f)


def label(k):
    return f"{k} {'x'.join(map(str, dims[k]))} (n={dims[k][1]})"


# 1. Per-launch dots, one strip per shape with its own GOPs window (so a 2% jitter is visible on a 1-60 GOPs set).
ROWS_PER_PAGE = 17
for key in RUNS:
    v, s = data[key]
    ks = sorted(v, key=order)
    with open(f"{OUT}/data/launches_{key}.csv", "w") as f:
        f.write("shape,m,n,k,vec_tile,vec_l1,vec_l2,vec_l3,vec_mean,scal_tile,scal_l1,scal_l2,scal_l3,scal_mean\n")
        for k in ks:
            f.write(f"{k},{','.join(map(str, dims[k]))},{v[k][0]},{','.join(f'{x:.3f}' for x in v[k][4])},{v[k][1]:.3f},"
                    f"{s[k][0]},{','.join(f'{x:.3f}' for x in s[k][4])},{s[k][1]:.3f}\n")
    pages = [ks[i:i + ROWS_PER_PAGE] for i in range(0, len(ks), ROWS_PER_PAGE)]
    for pi, part in enumerate(pages):
        out = ["\\begin{tikzpicture}"]
        for r, k in enumerate(part):
            dots = v[k][4] + s[k][4]
            lo, hi = min(dots), max(dots)
            mid, width = (lo + hi) / 2, max(hi - lo, 0.08 * v[k][1])  # window at least 8% of the vector mean
            xmin, xmax = mid - 0.62 * width, mid + 0.62 * width
            out.append(
                f"\\begin{{axis}}[dotrow, at={{(0,{-r * 1.25:.2f}cm)}}, xmin={xmin:.4f}, xmax={xmax:.4f}]\n"
                f"\\draw[muted, dashed, line width=0.4pt] (axis cs:{v[k][1]:.4f},-0.6) -- (axis cs:{v[k][1]:.4f},1.6);\n"
                f"\\addplot[only marks, mark=*, mark size=1.7pt, color=vec, fill opacity=0.75] coordinates "
                f"{{{''.join(f'({x:.4f},0)' for x in v[k][4])}}};\n"
                f"\\addplot[only marks, mark=*, mark size=1.7pt, color=scal, fill opacity=0.75] coordinates "
                f"{{{''.join(f'({x:.4f},1)' for x in s[k][4])}}};\n"
                f"\\addplot[only marks, mark=|, mark size=4.5pt, line width=1.3pt, color=vec!60!black] coordinates {{({v[k][1]:.4f},0)}};\n"
                f"\\addplot[only marks, mark=|, mark size=4.5pt, line width=1.3pt, color=scal!60!black] coordinates {{({s[k][1]:.4f},1)}};\n"
                f"\\node[anchor=east, font=\\scriptsize] at (rel axis cs:0,0.5) {{{label(k)}}};\n"
                f"\\node[anchor=west, font=\\scriptsize, align=left] at (rel axis cs:1,0.5) "
                f"{{\\textcolor{{vec!70!black}}{{{v[k][1]:.2f}}}\\\\\\textcolor{{scal!70!black}}{{{s[k][1]:.2f}}}}};\n"
                f"\\end{{axis}}")
        out.append("\\end{tikzpicture}")
        open(f"{OUT}/data/dots_{key}_{pi + 1}.tex", "w").write("\n".join(out) + "\n")

# 2. Gap against noise: x = |vector - scalar| mean GOPs, y = the wider launch range of the two sides (GOPs), log-log.
#    Separate files for shapes where scalar is faster (drawn hollow).
for th in ("1", "8"):
    for name, key, pick in [("yolo", f"yolo{th}", lambda l: True), ("bert", f"paper{th}", lambda l: l[0] == "B"),
                            ("square", f"paper{th}", lambda l: l[0] == "S")]:
        v, s = data[key]
        rows = {"pos": [], "neg": []}
        for k in sorted((k for k in v if pick(k)), key=order):
            gap = v[k][1] - s[k][1]
            noise = max(max(v[k][4]) - min(v[k][4]), max(s[k][4]) - min(s[k][4]))
            rows["pos" if gap >= 0 else "neg"].append(f"{abs(gap):.4f},{max(noise, 0.002):.4f},{k}\n")
        for sign, rs in rows.items():
            with open(f"{OUT}/data/gapnoise_{th}t_{name}_{sign}.csv", "w") as f:
                f.write("gap,noise,shape\n" + "".join(rs or ["nan,nan,none\n"]))

# 3. scalar/vector against reuse n (kept for Part 2).
for th in ("1", "8"):
    for name, key, pick in [("yolo", f"yolo{th}", lambda l: True), ("bert", f"paper{th}", lambda l: l[0] == "B"),
                            ("square", f"paper{th}", lambda l: l[0] == "S")]:
        v, s = data[key]
        with open(f"{OUT}/data/ratio_{th}t_{name}.csv", "w") as f:
            f.write("n,ratio,shape\n")
            for k in sorted((k for k in v if pick(k)), key=order):
                f.write(f"{dims[k][1]},{s[k][1] / v[k][1]:.4f},{k}\n")

# 4. T4: pack + compute time, as % of the vector packer's total (B), and predicted against measured (C).
if T4:
    ls = sorted({l for l in t4 if isinstance(l, str)}, key=lambda l: dims[l][1])
    with open(f"{OUT}/data/t4_time.csv", "w") as f:
        f.write("idx,label,vpack,vcomp,spack,scomp,vshare,sshare\n")
        for i, l in enumerate(ls):
            (vp, vc, _), (sp, sc, _) = t4[l]
            tot = vp + vc
            f.write(f"{i},{label(l)},{100*vp/tot:.2f},{100*vc/tot:.2f},{100*sp/tot:.2f},{100*sc/tot:.2f},"
                    f"{100*vp/tot:.0f},{100*sp/(sp+sc):.0f}\n")
    with open(f"{OUT}/data/t4_predicted.csv", "w") as f:
        f.write("pred,meas,shape\n")
        for l in ls:
            (vp, vc, _), (sp, sc, _) = t4[l]
            v, s = data["yolo1" if l[0] == "Y" else "paper1"]
            f.write(f"{(vp + vc) / (sp + sc):.4f},{s[l][1] / v[l][1]:.4f},{l}\n")
