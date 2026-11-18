"""Issue #85 tables + chart data: when does vector packing pay off? Vector against scalar packer, 1T and 8T.

Reads <issue-85>/<series>/runs.tsv and the run folders copied next to it (<series>/runs/<run>/all_raw_*.txt), for the
series jbpi_1t (1 thread, core 4) and jbpi2_8t (8 threads, cores 0-7). Writes tables.md (stdout) and data/*.csv for the
brief's pgfplots charts. Usage: python make_tables_and_figures.py <issue-85 dir>
"""
import csv, glob, os, statistics as st, sys

ROOT = sys.argv[1]
OUT = os.path.join(ROOT, "report")
SERIES = {"1t": "jbpi_1t", "8t": "jbpi2_8t"}
BASE_TILE = "24x8"
# Sweep -> points as (label, m, n, k), in sweep order. Shared points are measured once (datasets/issue85_sweep.dat).
SWEEPS = {
    "P-n": [(f"n={n}", 1200, n, 256) for n in [4, 8, 16, 32, 64, 128, 256, 512, 1024]],
    "P-m n=16": [(f"m={m}", m, 16, 256) for m in [64, 128, 300, 600, 1200, 2400, 4800]],
    "P-m n=256": [(f"m={m}", m, 256, 256) for m in [64, 128, 300, 600, 1200, 2400, 4800]],
    "P-k n=16": [(f"k={k}", 1200, 16, k) for k in [16, 32, 64, 144, 256, 576, 1152]],
    "P-k n=256": [(f"k={k}", 1200, 256, k) for k in [16, 32, 64, 144, 256, 576, 1152]],
}
TILES = ["16x4", "24x8", "32x4", "16x15", "32x7"]
TILE_SHAPES = [(1200, 4, 16), (1200, 16, 256), (1200, 256, 256)]


def tile_of(variant):  # "MR24_NR8" -> "24x8"
    return variant.replace("MR", "").replace("_NR", "x")


def rows_of(series_dir, step_prefix):
    """packer -> list over launches of {(m, n, k, tile): raw row}, for the runs.tsv rows whose step starts with step_prefix."""
    out = {"vector": [], "scalar": []}
    for step, _tag, packer, _board, run in csv.reader(open(os.path.join(series_dir, "runs.tsv")), delimiter="\t"):
        if not step.startswith(step_prefix):
            continue
        (raw,) = glob.glob(os.path.join(series_dir, "runs", run, "all_raw_*.txt"))
        launch = {}
        for r in csv.DictReader(open(raw)):
            launch[(int(r["M"]), int(r["N"]), int(r["K"]), tile_of(r["Variant"]))] = r
        out[packer].append(launch)
    return out


def m1(series_dir):
    """(m, n, k, tile) -> per packer (GOPs by mean time avg, by median rep avg, launch min, launch max), 3 launches each."""
    runs = rows_of(series_dir, "M1-")
    res = {}
    for key in runs["vector"][0] if runs["vector"] else []:
        res[key] = {}
        for p, launches in runs.items():
            assert len(launches) == 3, (series_dir, p, len(launches))
            means = [float(L[key]["KerGOPs_Mean"]) for L in launches]
            meds = [float(L[key]["KerGOPs_Median"]) for L in launches]
            res[key][p] = (st.mean(means), st.mean(meds), min(means), max(means))
    return res


def split(series_dir, prefix):
    """(m, n, k, tile) -> per packer (pack ms, compute ms, wall ms), means of 2 instrumented launches."""
    runs = rows_of(series_dir, prefix)
    res = {}
    for key in runs["vector"][0] if runs["vector"] else []:
        res[key] = {}
        for p, launches in runs.items():
            assert len(launches) == 2, (series_dir, p, len(launches))
            res[key][p] = tuple(st.mean(float(L[key][c]) for L in launches)
                                for c in ("Vector_Pack_Time_ms", "Compute_Time_ms", "RVV_Time_ms"))
    return res


data = {th: (m1(os.path.join(ROOT, d)), split(os.path.join(ROOT, d), "M2-" if th == "1t" else "M3-"))
        for th, d in SERIES.items() if os.path.exists(os.path.join(ROOT, d, "runs.tsv"))}


def row(th, key):
    """One point's numbers, or None when not measured yet."""
    g, sp = data.get(th, ({}, {}))
    if key not in g:
        return None
    v, s = g[key]["vector"], g[key]["scalar"]
    gap, noise = v[0] - s[0], max(v[3] - v[2], s[3] - s[2])
    out = {"v": v, "s": s, "ratio": s[0] / v[0], "ratio_med": s[1] / v[1], "real": abs(gap) > noise}
    if key in sp:
        (vp, vc, vw), (spk, sc, sw) = sp[key]["vector"], sp[key]["scalar"]
        out.update(vshare=vp / (vp + vc), sshare=spk / (spk + sc), packx=spk / vp, pred=(vp + vc) / (spk + sc),
                   vwait=max(0.0, 1 - (vp + vc) / vw), swait=max(0.0, 1 - (spk + sc) / sw))
    return out


def fmt_row(label, th, key):
    r = row(th, key)
    if r is None:
        return f"| {label} | {th} | not measured | | | | | |"
    split_txt = (f"{100*r['vshare']:.0f}% / {100*r['sshare']:.0f}% | {r['packx']:.2f} | {r['pred']:.3f}"
                 if "vshare" in r else "| |")
    return (f"| {label} | {th} | {r['v'][0]:.2f} / {r['v'][1]:.2f} | {r['s'][0]:.2f} / {r['s'][1]:.2f} | "
            f"{r['ratio']:.3f}{'' if r['real'] else ' ~'} | {split_txt} |")


HDR = ("| point | threads | vector GOPs mean / median | scalar GOPs mean / median | scalar/vector (mean) | "
       "pack share vec / scal | scal/vec pack time | predicted |\n|---|---|---|---|---|---|---|---|")
for name, pts in SWEEPS.items():
    print(f"\n### {name} (tile {BASE_TILE})\n\n{HDR}")
    for th in SERIES:
        for label, m, n, k in pts:
            print(fmt_row(f"{label} ({m}x{n}x{k})", th, (m, n, k, BASE_TILE)))
print(f"\n### P-tile\n\n{HDR}")
for m, n, k in TILE_SHAPES:
    for th in SERIES:
        for t in TILES:
            print(fmt_row(f"{m}x{n}x{k} {t}", th, (m, n, k, t)))
print("\n~ = the vector-scalar gap is inside the wider launch range (min-max of 3 launches): not a real difference.")
print("predicted = (vector pack + compute) / (scalar pack + compute), from M2 (1T) / M3 (8T), against the measured ratio.")

# ---------- chart data for the brief (pgfplots, horizontal bars, one chart per sweep and thread count) ----------
os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
for f in glob.glob(os.path.join(OUT, "data", "*.csv")):
    os.remove(f)


def write(fname, pts):
    with open(os.path.join(OUT, "data", fname), "w") as f:
        f.write("idx,label,vec,vlo,vhi,scal,slo,shi,ratio,rlo,rhi,real,vshare,sshare,pred,vwait,swait,shares\n")
        for i, (label, th, key) in enumerate(pts):
            r = row(th, key)
            if r is None:
                continue
            v, s = r["v"], r["s"]
            f.write(f"{i},{label},{v[0]:.2f},{v[0]-v[2]:.3f},{v[3]-v[0]:.3f},{s[0]:.2f},{s[0]-s[2]:.3f},{s[3]-s[0]:.3f},"
                    f"{r['ratio']:.4f},{r['ratio']-s[2]/v[3]:.4f},{s[3]/v[2]-r['ratio']:.4f},{int(r['real'])},{100*r.get('vshare', float('nan')):.1f},"
                    f"{100*r.get('sshare', float('nan')):.1f},{r.get('pred', float('nan')):.4f},"
                    f"{100*r.get('vwait', float('nan')):.1f},{100*r.get('swait', float('nan')):.1f},"
                    f"{'pack %.0f/%.0f%%' % (100*r['vshare'], 100*r['sshare']) if 'vshare' in r else ''}\n")


slug = {"P-n": "pn", "P-m n=16": "pm16", "P-m n=256": "pm256", "P-k n=16": "pk16", "P-k n=256": "pk256"}
for name, pts in SWEEPS.items():
    for th in SERIES:
        write(f"{slug[name]}_{th}.csv", [(label, th, (m, n, k, BASE_TILE)) for label, m, n, k in pts])
for th in SERIES:
    write(f"ptile_{th}.csv", [(f"n={n} {t}", th, (m, n, k, t)) for m, n, k in TILE_SHAPES for t in TILES])
# Model check over every point: predicted against measured scalar/vector, both thread counts.
misses = {}
for th in SERIES:
    with open(os.path.join(OUT, "data", f"model_{th}.csv"), "w") as f:
        f.write("pred,meas\n")
        for key in data.get(th, ({}, {}))[1]:
            r = row(th, key)
            if r and "pred" in r:
                f.write(f"{r['pred']:.4f},{r['ratio']:.4f}\n")
                misses.setdefault(th, []).append((abs(r["pred"] - r["ratio"]), key))
for th, ms in misses.items():
    d = sorted(m for m, _ in ms)
    print(f"model check {th}: {len(d)} points, |pred - meas| median {st.median(d):.3f}, "
          f"within 0.02: {sum(x <= 0.02 for x in d)}, within 0.05: {sum(x <= 0.05 for x in d)}; worst: "
          + ", ".join(f"{k[0]}x{k[1]}x{k[2]} {k[3]} {m:.3f}" for m, k in sorted(ms, reverse=True)[:4]))


# ---------- LaTeX table rows: GOPs by mean / by median rep, 1T and 8T, better packer in bold ----------
def gops_cells(r):
    if r is None:
        return ["--", "--", "--"]
    v, s = r["v"], r["s"]
    tl = "" if r["real"] else "$^{\\sim}$"
    vb, sb = v[0] >= s[0], s[0] > v[0]
    cell = lambda g, b: (f"\\textbf{{{g[0]:.2f} / {g[1]:.2f}}}{tl}" if b else f"{g[0]:.2f} / {g[1]:.2f}")
    return [cell(v, vb), cell(s, sb), f"{r['ratio']:.3f}"]


def tex_rows(fname, pts):
    with open(os.path.join(OUT, "data", fname), "w") as f:
        for label, key in pts:
            r1, r8 = row("1t", key), row("8t", key)
            sp = (f"{r8['v'][0]/r1['v'][0]:.2f} & {r8['s'][0]/r1['s'][0]:.2f}" if r1 and r8 else "-- & --")
            f.write(" & ".join([label] + gops_cells(r1) + gops_cells(r8)) + f" & {sp} \\\\\n")


for name, pts in SWEEPS.items():
    tex_rows(f"tab_{slug[name]}.tex", [(f"{label} ({m}$\\times${n}$\\times${k})", (m, n, k, BASE_TILE)) for label, m, n, k in pts])
tex_rows("tab_ptile.tex", [(f"{m}$\\times${n}$\\times${k}, {t}", (m, n, k, t)) for m, n, k in TILE_SHAPES for t in TILES])
