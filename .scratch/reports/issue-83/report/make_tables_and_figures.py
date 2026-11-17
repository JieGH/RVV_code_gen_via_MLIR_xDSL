"""Issue #83 final tables + figures: vector vs scalar packing. Run from tests/pure_xdsl_blis_rvv/logs.
argv[1] = output dir. Prints markdown tables to stdout."""
import csv, glob, re, statistics as st, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})
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

# ---------- figures ----------
def bars(ax, v, s, shapes, rot=0):
    x = range(len(shapes))
    for off, src, col, lab in [(-0.2, v, BLUE, "vector packer"), (0.2, s, ORANGE, "scalar packer")]:
        ys = [src[k][1] for k in shapes]
        lo = [src[k][1] - min(src[k][4]) for k in shapes]
        hi = [max(src[k][4]) - src[k][1] for k in shapes]
        ax.bar([i + off for i in x], ys, 0.38, color=col, label=lab)
        ax.errorbar([i + off for i in x], ys, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.5)
    ax.set_xticks(list(x), shapes, rotation=rot, fontsize=7 if rot else 9)
    ax.set_xlim(-0.7, len(shapes) - 0.3)
    ax.set_ylabel("GOPs by mean time (best tile)")
    ax.grid(axis="x", visible=False)


NOTE = "Uninstrumented build, reference protocol, 3 launches per side; bar = mean of launches, whisker = launch min-max."
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, k in [(axes[0], "paper1"), (axes[1], "paper8")]:
    v, s = data[k]
    bars(ax, v, s, sorted(v, key=order))
    ax.set_title(TITLE[k].split(", ")[1], color=INK, loc="left")
axes[1].legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 1.16), ncol=2)
fig.suptitle("BERT (B1-B5) and Square (S1-S5): vector vs scalar packing", x=0.01, ha="left", color=INK)
fig.text(0.01, 0.005, NOTE + " Vector on jbpi (#71), scalar on jbpi2 (#83).", fontsize=7, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
fig.savefig(f"{OUT}/fig1_paper_vector_vs_scalar.png", dpi=150)

fig, axes = plt.subplots(2, 1, figsize=(13, 7.2))
for ax, k in [(axes[0], "yolo1"), (axes[1], "yolo8")]:
    v, s = data[k]
    bars(ax, v, s, sorted(v, key=order), rot=90)
    ax.set_title(TITLE[k], color=INK, loc="left")
axes[0].legend(frameon=False, loc="upper left")
fig.text(0.01, 0.005, NOTE + " jbpi. Vector from #61 (1T) and #63 (8T).", fontsize=7, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(f"{OUT}/fig2_yolo_per_layer.png", dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, th in [(axes[0], "1"), (axes[1], "8")]:
    for key, pick, col, mk, lab in [(f"yolo{th}", lambda l: True, BLUE, "o", "YOLO"),
                                    (f"paper{th}", lambda l: l.startswith("B"), ORANGE, "s", "BERT"),
                                    (f"paper{th}", lambda l: l.startswith("S"), AQUA, "^", "Square")]:
        v, s = data[key]
        ks = [k for k in v if pick(k)]
        ax.scatter([dims[k][1] for k in ks], [s[k][1] / v[k][1] for k in ks], s=34, color=col, marker=mk, label=lab,
                   edgecolor="white", linewidth=1, zorder=3)
    ax.axhline(1.0, color=MUTED, linewidth=1)
    ax.axhspan(0.97, 1.03, color=GRID, alpha=0.6, zorder=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 16, 64, 256, 1000, 5000], ["4", "16", "64", "256", "1000", "5000"])
    ax.set_xlabel("n = uses of each packed A element (reuse), log scale")
    ax.set_title(f"{th} thread{'s' if th == '8' else ''}", color=INK, loc="left")
axes[0].set_ylabel("scalar / vector GOPs (best tile, by mean time)")
axes[1].legend(frameon=False, loc="lower right")
fig.suptitle("The packer matters only when reuse n is small", x=0.01, ha="left", color=INK)
fig.text(0.01, 0.005, "Grey band: ±3%, inside the day/board gap, not called a difference. One dot per shape; "
         "uninstrumented build, reference protocol, 3 launches per side. 1T BERT outlier at 1.19 is B3 (vector launches 5.3-8.4 GOPs).", fontsize=7, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
fig.savefig(f"{OUT}/fig3_ratio_vs_reuse_n.png", dpi=150)

if T4:
    ls = sorted({l for l in t4 if isinstance(l, str)}, key=lambda l: dims[l][1])
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = range(len(ls))
    ax.bar([i - 0.2 for i in x], [100 * t4[l][0][2] for l in ls], 0.38, color=BLUE, label="vector packer")
    ax.bar([i + 0.2 for i in x], [100 * t4[l][1][2] for l in ls], 0.38, color=ORANGE, label="scalar packer")
    ax.set_xticks(list(x), [f"{l}\nn={dims[l][1]}" for l in ls], fontsize=8)
    ax.set_ylabel("pack time / (pack + compute), %")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Share of time spent packing, by shape (sorted by reuse n)", color=INK, loc="left")
    fig.text(0.01, 0.005, "T4: instrumented build (timers cost 11-17%), warm-up protocol, 200 reps, 1T core 4, jbpi2; "
             "4 launches vector, scalar, scalar, vector; each shape on its 1T best tile.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(f"{OUT}/fig4_pack_share_t4.png", dpi=150)
