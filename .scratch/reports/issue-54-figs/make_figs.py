"""Charts for the issue #54 brief, read straight from the committed series logs.

Run from the repo root with the project venv:
    .venv/bin/python .scratch/reports/issue-54-figs/make_figs.py

Writes fig_headline.pdf, fig_reps.pdf, fig_heatmap.pdf and fig_pack_cost.pdf next to this file.
Every number comes from logs/issue55_* (series reports, pack-only files) or the run folders they index;
nothing is typed in by hand. All runs: uninstrumented, reference protocol, --tmin 10, core 4,
--no-nc-overrides, board jbpi2.
"""
from __future__ import annotations

import csv
import gzip
import re
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
TESTS = ROOT / "tests" / "pure_xdsl_blis_rvv"
LOGS = TESTS / "logs"
S1 = LOGS / "issue55_2026-10-22_194043"
S2 = LOGS / "issue55_2026-10-23_100122"
OUT = Path(__file__).resolve().parent

sys.path.insert(0, str(TESTS))
from reps_report import slow_reps  # noqa: E402  (the same fast/slow split as the series reports)

# Reference palette, light mode (validated: first three slots, all pairs).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
OTHER = "#b5b3ab"
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
        "#256abf", "#1c5cab", "#184f95", "#104281"]

COLOR = {"row": ORANGE, "row-intr": ORANGE, "scalar": AQUA, "col-g1-m4": BLUE}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True, "text.color": INK,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
TEXTW = 6.85  # inches: a4 with 1.8 cm margins

TILES = ["16x6", "16x8", "16x15"]
GEMM_OPS = {"B3": 2 * 64 * 384 * 384, "B3_K256": 2 * 64 * 384 * 256}


def tile_tex(t: str) -> str:
    return t.replace("x", "×")


# ---------------------------------------------------------------- English and Spanish copies of every figure
LANGS = ("en", "es")


def _figure_texts(fig):
    from matplotlib.text import Text
    return [t for t in fig.findobj(Text) if t.get_text() and "Mono" not in " ".join(t.get_fontfamily())]


def save(fig, name, **kw):
    """Save the English figure to OUT/name and a Spanish copy to OUT/es/name (labels translated by es_labels.tr;
    code text, drawn in a monospace font, is left as it is)."""
    import os
    fig.savefig(OUT / name, **kw)
    if os.environ.get("DUMP_STRINGS"):
        with open(os.environ["DUMP_STRINGS"], "a") as f:
            fig.canvas.draw()
            for t in _figure_texts(fig):
                f.write(repr(t.get_text()) + "\n")
    import es_labels
    # Freeze every tick label as drawn (only the ticks inside the view), then translate it in place.
    fig.canvas.draw()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            pairs = [(loc, t.get_text()) for loc, t in zip(axis.get_majorticklocs(), axis.get_majorticklabels())
                     if lo - 1e-9 <= loc <= hi + 1e-9]
            if pairs and any(txt for _, txt in pairs):
                axis.set_ticks([loc for loc, _ in pairs])
                axis.set_ticklabels([es_labels.tr(txt) for _, txt in pairs])
    fig.canvas.draw()
    for t in _figure_texts(fig):
        t.set_text(es_labels.tr(t.get_text()))
    (OUT / "es").mkdir(exist_ok=True)
    fig.savefig(OUT / "es" / name, **kw)


# ---------------------------------------------------------------- series reports (GEMM tables)
def read_report(path: Path) -> dict[tuple[str, str, str], tuple[float, float, float]]:
    """(tile, shape, variant) -> (GOPs by mean time, GOPs by median rep, slow %), board jbpi2 only."""
    out, cur = {}, None
    for line in path.read_text().splitlines():
        m = re.match(r"### (\S+), (\S+), (\S+)$", line)
        if m:
            cur = (m.group(2), m.group(3)) if m.group(1) == "jbpi2" else None
            continue
        if cur and line.startswith("| ") and not line.startswith("| variant") and "---" not in line:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) >= 6 and re.match(r"[\d.]+ \(", cells[3]):
                out[(cur[0], cur[1], cells[0])] = (float(cells[3].split()[0]), float(cells[4]), float(cells[5]))
    return out


R1 = read_report(S1 / "report_series1_jbpi2.md")
R2 = read_report(S2 / "report_series2_jbpi2.md")


def cell(series: dict, tile: str, shape: str, variant: str):
    return series[(tile, shape, variant)]


# ---------------------------------------------------------------- per-rep files
def run_dir(series_dir: Path, launch: int, arm: str) -> Path:
    with open(series_dir / "runs.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["board"] == "jbpi2" and int(r["launch"]) == launch and r["arm"] == arm and r["status"] == "ok":
                return LOGS / r["result_dir"]
    raise LookupError((series_dir, launch, arm))


def gemm_reps(series_dir: Path, launch: int, arm: str, tile: str, shape: str) -> list[float]:
    nr = tile.split("x")[1]
    path = next((run_dir(series_dir, launch, arm) / f"fp32_AXPY_FP32_Pack_vlen256_MR16NR{nr}").glob("reps_*.gz"))
    with gzip.open(path, "rt") as f:
        return [float(r["Seconds"]) for r in csv.DictReader(f) if r["Layer"] == shape]


def pack_reps(series_dir: Path, launch: int, variant: str, shape: str, tile: str) -> list[float]:
    name = f"packonly_jbpi2_L{launch}_{variant}_{shape}_{tile}.txt"
    lines = (series_dir / name).read_text().splitlines()
    return [float(l.split(",")[1]) for l in lines if l and l[0].isdigit()]


# ---------------------------------------------------------------- figure 1: headline bars
def bars(ax, groups, variants, values, slow, ymax, label_slow=True, fs=5.6):
    n = len(variants)
    w = 0.8 / n
    for vi, v in enumerate(variants):
        for gi, _ in enumerate(groups):
            x = gi + (vi - (n - 1) / 2) * w
            y = values[v][gi]
            ax.bar(x, y, width=w - 0.03, color=COLOR[v], edgecolor="white", linewidth=0.6, zorder=2)
            ax.text(x, y + ymax * 0.012, f"{y:.2f}", ha="center", va="bottom", fontsize=fs, color=INK)
            s = slow[v][gi]
            if label_slow and s > 0:
                ax.text(x, y * 0.5, f"{s:.0f}% slow", ha="center", va="center", rotation=90, fontsize=5.4,
                        color="white", fontweight="bold")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylim(0, ymax)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("GOPs (by mean time)")


def fig_headline():
    variants = ["row", "scalar", "col-g1-m4"]
    names = {"row": "row (current xDSL packer)", "scalar": "scalar", "col-g1-m4": "col-g1-m4 (one column at a time)"}
    fig, (a, b) = plt.subplots(2, 1, figsize=(TEXTW, 4.1), gridspec_kw={"hspace": 0.55})

    groups, keys = [], []
    for shape in ["B3", "B3_K256"]:
        for t in TILES:
            groups.append(f"{shape}\n{tile_tex(t)}" + ("\n(default tile)" if t == "16x15" else ""))
            keys.append((t, shape))
    vals = {v: [cell(R1, t, s, v)[0] for t, s in keys] for v in variants}
    slow = {v: [cell(R1, t, s, v)[2] for t, s in keys] for v in variants}
    bars(a, groups, variants, vals, slow, 9.0)
    a.set_title("(a) Shape under test: B3 (64×384×384) and B3_K256 (64×384×256)", loc="left", color=INK)
    a.axvline(2.5, color=AXIS, linewidth=0.6)

    groups, keys = [], []
    for shape, label in [("B2", "B2"), ("R24", "Reuse24")]:
        for t in TILES:
            groups.append(f"{label}\n{tile_tex(t)}")
            keys.append((t, shape))
    vals = {v: [cell(R1, t, s, v)[0] for t, s in keys] for v in variants}
    slow = {v: [cell(R1, t, s, v)[2] for t, s in keys] for v in variants}
    bars(b, groups, variants, vals, slow, 14.0, fs=4.7)
    b.set_title("(b) Controls that must not get worse: B2 (384×384×64) and Reuse24 (384³)", loc="left", color=INK)
    b.axvline(2.5, color=AXIS, linewidth=0.6)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[v]) for v in variants]
    fig.legend(handles, [names[v] for v in variants], loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig_headline.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2: per-rep traces
def fig_reps():
    fig, axs = plt.subplots(2, 2, figsize=(TEXTW, 3.9), sharey="row",
                            gridspec_kw={"hspace": 0.75, "wspace": 0.06})
    ops = GEMM_OPS["B3"]
    panels = [
        (axs[0][0], "row", gemm_reps(S2, 1, "row", "16x15", "B3"), "GEMM, current row packer"),
        (axs[0][1], "col-g1-m4", gemm_reps(S2, 1, "col-g1-m4", "16x15", "B3"), "GEMM, col-g1-m4"),
    ]
    for ax, v, secs, title in panels:
        gops = [ops / (s * 1e9) for s in secs]
        mask = slow_reps(ops, secs)
        share = 100 * sum(mask[1:]) / max(1, len(mask) - 1)
        t, acc = [], 0.0
        for s in secs:
            acc += s
            t.append(acc)
        ax.scatter(t, gops, s=0.6, color=COLOR[v], linewidths=0, rasterized=True)
        ax.set_title(f"{title}\n{len(secs)} reps, {share:.0f}% slow, "
                     f"{len(secs) * ops / (sum(secs) * 1e9):.2f} GOPs by mean time", loc="left", color=INK)
        ax.set_xlabel("time into the run (s)")
        ax.set_ylim(0, 10)
    axs[0][0].set_ylabel("GOPs of one rep")

    pk = [
        (axs[1][0], "row-intr", pack_reps(S1, 1, "row-intr", "B3", "16x15"), "B packer alone, row-intr"),
        (axs[1][1], "col-g1-m4", pack_reps(S1, 1, "col-g1-m4", "B3", "16x15"), "B packer alone, col-g1-m4"),
    ]
    for ax, v, secs, title in pk:
        ms = [1e3 * s for s in secs]
        mask = slow_reps(1e9, secs)
        share = 100 * sum(mask[1:]) / max(1, len(mask) - 1)
        ax.scatter(range(len(ms)), ms, s=1.2, color=COLOR[v], linewidths=0, rasterized=True)
        ax.set_title(f"{title}\n{len(ms)} reps, {share:.0f}% slow, mean {statistics.fmean(ms):.3f} ms",
                     loc="left", color=INK)
        ax.set_xlabel("rep")
        ax.set_ylim(0, 1.6)
    axs[1][0].set_ylabel("pack time of one rep (ms)")
    for row in axs:
        for ax in row:
            ax.grid(True, axis="y")
    save(fig, "fig_reps.pdf", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- figure 3: heatmap of every GEMM cell
def fig_heatmap():
    rows = [("row (S1)", R1, "row"), ("row (S2)", R2, "row"), ("row-intr", R1, "row-intr"), ("scalar", R1, "scalar"),
            ("col-g1", R1, "col-g1"), ("col-g1-m4 (S1)", R1, "col-g1-m4"), ("col-g1-m4 (S2)", R2, "col-g1-m4"),
            ("col-g2", R2, "col-g2"), ("col-g4", R2, "col-g4"), ("col-gall (S1)", R1, "col-gall"),
            ("col-gall (S2)", R2, "col-gall")]
    cols = [(t, s) for t in TILES for s in ["B3", "B3_K256"]]
    cmap = LinearSegmentedColormap.from_list("blue", RAMP)
    lo, hi = 3.0, 8.5
    fig, ax = plt.subplots(figsize=(TEXTW, 3.5))
    ax.grid(False)
    for ri, (label, rep, v) in enumerate(rows):
        for ci, (t, s) in enumerate(cols):
            g, _, sl = cell(rep, t, s, v)
            frac = min(1, max(0, (g - lo) / (hi - lo)))
            ax.add_patch(plt.Rectangle((ci, ri), 1, 1, facecolor=cmap(frac), edgecolor="white", linewidth=1.5))
            ink = "white" if frac > 0.55 else INK
            ax.text(ci + 0.5, ri + 0.38, f"{g:.2f}", ha="center", va="center", fontsize=7, color=ink,
                    fontweight="bold")
            ax.text(ci + 0.5, ri + 0.72, f"{sl:.0f}% slow", ha="center", va="center",
                    fontsize=5.5, color=ink)
    ax.set_xlim(0, len(cols))
    ax.set_ylim(len(rows), 0)
    ax.set_xticks([i + 0.5 for i in range(len(cols))])
    ax.set_xticklabels([f"{tile_tex(t)}\n{s}" for t, s in cols])
    ax.xaxis.tick_top()
    ax.set_yticks([i + 0.5 for i in range(len(rows))])
    ax.set_yticklabels([r[0] for r in rows])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axhline(4, color=INK, linewidth=1.0)
    ax.axvline(2, color="white", linewidth=4)
    ax.axvline(4, color="white", linewidth=4)
    ax.text(len(cols) + 0.1, 2, "take 4 B\nper column", va="center", fontsize=6.5, color=INK2)
    ax.text(len(cols) + 0.1, 7.5, "take whole\ncolumns\n(32 or 128 B)", va="center", fontsize=6.5, color=INK2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo, hi))
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.03, aspect=50)
    cb.set_label("GOPs by mean time (darker is faster)")
    cb.outline.set_visible(False)
    save(fig, "fig_heatmap.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- figure 4: what one pack costs
def pack_stats(series_dir: Path, variant: str, shape: str, tile: str):
    alls, fasts, shares = [], [], []
    for launch in (1, 2, 3):
        secs = pack_reps(series_dir, launch, variant, shape, tile)
        mask = slow_reps(1e9, secs)
        warm = list(zip(secs[1:], mask[1:]))
        fast = [s for s, m in warm if not m]
        alls.append(1e3 * statistics.fmean(secs))
        fasts.append(1e3 * statistics.fmean(fast))
        shares.append(100 * sum(m for _, m in warm) / len(warm))
    return statistics.fmean(alls), statistics.fmean(fasts), statistics.fmean(shares)


def fig_pack_cost():
    order = [("row-intr", S1), ("scalar", S1), ("col-g1", S1), ("col-g1-m4", S1), ("col-g2", S2), ("col-g4", S2),
             ("col-gall", S1)]
    fig, axs = plt.subplots(1, 2, figsize=(TEXTW, 2.3), sharey=True, gridspec_kw={"wspace": 0.1})
    for ax, (shape, tile) in zip(axs, [("B3", "16x15"), ("B3", "16x8")]):
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        for i, (v, sd) in enumerate(order):
            mean_all, mean_fast, share = pack_stats(sd, v, shape, tile)
            ax.barh(i, mean_all, height=0.62, color=COLOR.get(v, OTHER), zorder=2)
            ax.plot([mean_fast, mean_fast], [i - 0.36, i + 0.36], color=INK, linewidth=1.2, zorder=3)
            ax.text(mean_all + 0.02, i, f"{mean_all:.3f} ms" + (f"  ({share:.0f}% slow)" if share >= 1 else ""),
                    va="center", fontsize=6, color=INK)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([v for v, _ in order])
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_xlim(0, 2.3)
        ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
        ax.set_xlabel("mean pack time per rep, all reps (ms)")
        ax.set_title(f"{shape}, {tile_tex(tile)}" + (" (default tile)" if tile == "16x15" else ""), loc="left",
                     color=INK)
        ax.tick_params(axis="y", length=0)
    fig.legend([plt.Line2D([0], [0], color=INK, linewidth=1.2)], ["mean of the fast reps only"], loc="lower right",
               frameon=False, bbox_to_anchor=(0.98, 0.98), ncol=1)
    save(fig, "fig_pack_cost.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- figure 5: what one panel asks of memory
LINE = 64  # bytes, L1D and L2 line of the X60 (cache_config.py, jbpi2)


def panel_instructions(variant: str, NR: int, kc: int, ldb: int):
    """The loads and stores one full B panel issues, as lists of byte addresses per instruction.

    Follows the loop nests of templates/pack_b_variants.h (row-intr, col-gG, col-g1-m4) and of the current
    xDSL packer (row: one vlse32 per k per chunk of 8 columns, then one vse32), and pack_CB_scalar (scalar).
    B column j starts at j*ldb*4 and Bc element (k, j) is at (k*NR + j)*4; both bases are line aligned
    (ldb*4 = 1536 B and the panel size are multiples of 64 B)."""
    b = lambda k, j: (j * ldb + k) * 4  # noqa: E731
    c = lambda k, j: (1 << 30) + (k * NR + j) * 4  # noqa: E731
    loads, stores, live = [], [], 0
    if variant == "row":
        live = NR
        for k in range(kc):
            for c0 in range(0, NR, 8):
                cols = range(c0, min(NR, c0 + 8))
                loads.append([b(k, j) for j in cols])
                stores.append([c(k, j) for j in cols])
    elif variant == "scalar":
        live = NR
        for k in range(kc):
            for j in range(NR):
                loads.append([b(k, j)])
                stores.append([c(k, j)])
    else:
        g = {"col-g1": 1, "col-g2": 2, "col-g4": 4, "col-gall": min(NR, 24), "col-g1-m4": 1}[variant]
        strip = 32 if variant == "col-g1-m4" else 8
        live = g
        for j0 in range(0, NR, g):
            grp = range(j0, min(NR, j0 + g))
            for i in range(0, kc, strip):
                ks = range(i, min(kc, i + strip))
                for j in grp:
                    loads.append([b(k, j) for k in ks])
                for j in grp:
                    stores.append([c(k, j) for k in ks])
    return loads, stores, live


def lines_of(addrs):
    return len({a // LINE for a in addrs})


def fig_counts():
    NR, kc, ldb = 15, 128, 384  # B3 on the current 16x15 tile: kc = 128 from the cache model, ldb = K
    order = ["row", "scalar", "col-g1", "col-g1-m4", "col-g2", "col-g4", "col-gall"]
    label = {"row": "row, row-intr"}
    stats = {}
    for v in order:
        loads, stores, live = panel_instructions(v, NR, kc, ldb)
        per_col_bytes = statistics.fmean(
            4 * min(len(l), 32 if v == "col-g1-m4" else 8) if v.startswith("col") else 4 for l in loads)
        stats[v] = {
            "load instructions\nper panel": len(loads),
            "B lines touched\nby one load": statistics.fmean(lines_of(l) for l in loads),
            "bytes taken from one\ncolumn per touch (C)": per_col_bytes,
            "B columns read\nside by side (B)": live,
            "Bc lines touched\nby one store": statistics.fmean(lines_of(s) for s in stores),
        }
    metrics = list(stats[order[0]].keys())
    fig, axs = plt.subplots(1, len(metrics), figsize=(TEXTW, 2.15), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, m in zip(axs, metrics):
        vals = [stats[v][m] for v in order]
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        for i, (v, val) in enumerate(zip(order, vals)):
            ax.barh(i, val, height=0.62, color=COLOR.get(v, OTHER), zorder=2)
            txt = f"{val:.0f}" if abs(val - round(val)) < 0.05 else f"{val:.1f}"
            ax.text(val + max(vals) * 0.03, i, txt, va="center", fontsize=6, color=INK)
        ax.set_xlim(0, max(vals) * 1.35)
        ax.set_title(m, color=INK, fontsize=6.8)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=5.5)
        ax.locator_params(axis="x", nbins=3)
    axs[0].set_yticks(range(len(order)))
    axs[0].set_yticklabels([label.get(v, v) for v in order])
    axs[0].set_ylim(len(order) - 0.5, -0.5)
    save(fig, "fig_counts.pdf")
    plt.close(fig)
    return stats


# ---------------------------------------------------------------- figure 6: register, line and touch sizes
def fig_units():
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(TEXTW, 2.35))
    ax.set_xlim(-46, 136)
    ax.set_ylim(7.4, -0.9)
    ax.axis("off")
    rows = [
        ("one vector register\nVLEN = 256 bit = 32 B", [(0, 32, 8, BLUE, "8 × fp32 (LMUL = 1)")]),
        ("register group, LMUL = 4\nv4–v7 = 128 B",
         [(32 * r, 32, 8, BLUE, None) for r in range(4)] + [(None, None, None, None, "32 × fp32 (col-g1-m4)")]),
        ("cache line (L1D and L2)\n64 B", [(0, 64, 16, MUTED, None), (64, 64, 16, MUTED, None),
                                           (None, None, None, None, "16 × fp32 per line")]),
        ("row, row-intr, scalar:\nbytes from one column", [(0, 4, 1, ORANGE, "4 B, then the next column")]),
        ("col-g1, -g2, -g4, -gall:\nbytes from one column", [(0, 32, 8, BLUE, "32 B = half a line")]),
        ("col-g1-m4:\nbytes from one column", [(0, 128, 32, BLUE, "128 B = two whole lines")]),
    ]
    for r, (label, segs) in enumerate(rows):
        y = r * 1.2
        ax.text(-2, y + 0.35, label, ha="right", va="center", fontsize=6.5, color=INK)
        ax.add_patch(Rectangle((0, y), 128, 0.7, facecolor="white", edgecolor=GRID, linewidth=0.5))
        note, end = None, 0
        for x0, w, n, col, txt in segs:
            if x0 is None:
                note = txt
                continue
            for e in range(n):
                ax.add_patch(Rectangle((x0 + e * w / n, y), w / n, 0.7, facecolor=col, edgecolor="white",
                                       linewidth=0.5, alpha=0.9))
            ax.add_patch(Rectangle((x0, y), w, 0.7, facecolor="none", edgecolor=INK, linewidth=0.8))
            end = max(end, x0 + w)
            if txt:
                note = txt
        if note:
            ax.text(129.5, y + 0.35, note, ha="left", va="center", fontsize=6.2, color=INK2)
    for b in range(0, 129, 16):
        ax.text(b, -0.35, f"{b} B", ha="center", va="bottom", fontsize=5.5, color=MUTED)
        ax.plot([b, b], [-0.25, 7.0], color=GRID, linewidth=0.4, zorder=0)
    save(fig, "fig_units.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- figure 7: read and write order, first 3 instructions
def fig_read_order():
    from matplotlib.patches import Rectangle

    NK, NJ = 32, 8  # the first 32 k of one panel, NR = 8 (16x8 tile): one vlse32 per k for row
    shades = ["#0d366b", "#3987e5", "#9ec5f4"]
    oshades = ["#8a2f0c", "#eb6834", "#f6b89c"]

    def first3(variant):
        """(k, j) cells of the first three load instructions, in order."""
        if variant == "row":
            return [[(k, j) for j in range(NJ)] for k in range(3)]
        if variant == "col-g1":
            return [[(k, 0) for k in range(i, i + 8)] for i in (0, 8, 16)]
        if variant == "col-g1-m4":
            return [[(k, j) for k in range(32)] for j in (0, 1, 2)]
        if variant == "col-g2":
            return [[(k, 0) for k in range(8)], [(k, 1) for k in range(8)], [(k, 0) for k in range(8, 16)]]
        raise ValueError(variant)

    panels = [("row (current), row-intr", "row", "vlse32, stride 1536 B: 8 lines,\n4 B from each",
               "vse32: 32 B in a row, 1 line"),
              ("col-g1", "col-g1", "vle32 (LMUL 1): 32 B of one\ncolumn, half a line",
               "vsse32, stride 32 B: 8 rows,\n4 lines"),
              ("col-g2 (col-g4, col-gall alike)", "col-g2", "same loads as col-g1, but 2\ncolumns before any store",
               "same stores as col-g1"),
              ("col-g1-m4 (the fix)", "col-g1-m4", "vle32 (LMUL 4): 128 B of one\ncolumn, two whole lines",
               "vsse32, stride 32 B: 32 rows,\n16 lines")]
    fig, axs = plt.subplots(2, 4, figsize=(TEXTW, 3.9), gridspec_kw={"hspace": 0.6, "wspace": 0.35,
                                                                     "height_ratios": [1, 1]})
    for col, (title, v, load_txt, store_txt) in enumerate(panels):
        sh = oshades if v == "row" else shades
        order = first3(v)
        for row, where in enumerate(["B", "Bc"]):
            ax = axs[row][col]
            ax.set_xlim(-0.2, NJ + 0.2)
            ax.set_ylim(NK + 0.3, -0.3)
            ax.axis("off")
            for k in range(NK):
                for j in range(NJ):
                    ax.add_patch(Rectangle((j, k), 1, 1, facecolor="#f4f3ef", edgecolor="white", linewidth=0.4))
            for n, cells in enumerate(order):
                for (k, j) in cells:
                    ax.add_patch(Rectangle((j, k), 1, 1, facecolor=sh[n], edgecolor="white", linewidth=0.4))
            if where == "B":
                # column-major: each column is contiguous; a line holds 16 k of one column
                for j in range(NJ):
                    for k0 in range(0, NK, 16):
                        ax.add_patch(Rectangle((j + 0.06, k0 + 0.06), 0.88, 15.88, facecolor="none",
                                               edgecolor=INK2, linewidth=0.6))
                ax.set_title(f"{title}\nreads from B", loc="left", color=INK, fontsize=6.8)
                ax.text(0, NK + 1.2, load_txt, va="top", fontsize=5.9, color=INK)
            else:
                # Bc panel, row-major with NR = 8: one row is 32 B, a line holds two rows
                for k0 in range(0, NK, 2):
                    ax.add_patch(Rectangle((0.06, k0 + 0.06), NJ - 0.12, 1.88, facecolor="none",
                                           edgecolor=INK2, linewidth=0.6))
                ax.set_title("writes into Bc", loc="left", color=INK, fontsize=6.8)
                ax.text(0, NK + 1.2, store_txt, va="top", fontsize=5.9, color=INK)
            if col == 0:
                ax.text(-0.6, NK / 2, "k (0..31)  →" if where == "B" else "k (0..31)  →", rotation=90,
                        ha="right", va="center", fontsize=6, color=INK2)
                ax.text(NJ / 2, -2.4 if where == "B" else -2.4, "", fontsize=6)
    handles = [Rectangle((0, 0), 1, 1, facecolor=c) for c in shades] + \
              [Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=INK2)]
    fig.legend(handles, ["1st load", "2nd load", "3rd load", "one 64 B cache line"], loc="upper center",
               ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    save(fig, "fig_read_order.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- figure 8: reference vs ours vs the fix
def fig_ref_vs_ours():
    """The first 32 numbers each packer reads from B (K = 384, so columns are 1536 B apart), drawn twice: on the
    matrix (left) and on the memory addresses (right). Reference: gemm_blis.c pack_CB, one scalar load per number,
    row by row across the panel. Ours (current): one vlse32 per row of 8 columns. The fix: one vle32 (LMUL 4) of
    32 numbers down one column."""
    from matplotlib.patches import Rectangle

    NK, NJ, COLB = 16, 8, 1536
    designs = [
        ("Reference design (Adrián's gemm_blis.c)", AQUA, "scalar",
         "1 scalar load per number: 32 loads.\nEach load jumps 1536 B to the next column;\n"
         "after 8 it jumps back. 4 B used per touch."),
        ("Our design, current (xDSL vector packer)", ORANGE, "vector-row",
         "1 strided vector load per row: 4 loads,\neach sends 8 requests to 8 lines at once.\n"
         "Same addresses, same order, in bursts."),
        ("The fix, col-g1-m4", BLUE, "fix",
         "1 vector load (LMUL 4) for all 32 numbers:\n128 B in one piece, two whole cache lines,\n"
         "then the next column. No jumping."),
    ]

    def order(kind):
        if kind == "fix":
            return [(k, 0) for k in range(32)]
        return [(k, j) for k in range(4) for j in range(NJ)]

    fig, axs = plt.subplots(3, 2, figsize=(TEXTW, 5.6),
                            gridspec_kw={"width_ratios": [1, 2.1], "hspace": 0.55, "wspace": 0.12})
    for r, (title, col, kind, note) in enumerate(designs):
        seq = order(kind)
        # ---- left: the matrix B, column-major, first NK k of 8 columns, read order numbered
        ax = axs[r][0]
        ax.set_xlim(-0.3, NJ + 0.3)
        ax.set_ylim(NK + 0.2, -0.2)
        ax.axis("off")
        for k in range(NK):
            for j in range(NJ):
                ax.add_patch(Rectangle((j, k), 1, 1, facecolor="#f4f3ef", edgecolor="white", linewidth=0.5))
        for n, (k, j) in enumerate(seq):
            if k >= NK:
                continue
            ax.add_patch(Rectangle((j, k), 1, 1, facecolor=col, edgecolor="white", linewidth=0.5, alpha=0.85))
            ax.text(j + 0.5, k + 0.55, str(n + 1), ha="center", va="center", fontsize=4.6, color="white")
        for j in range(NJ):
            ax.add_patch(Rectangle((j + 0.05, 0.05), 0.9, NK - 0.1, facecolor="none", edgecolor=INK2, linewidth=0.6))
        if kind == "vector-row":
            for k in range(4):
                ax.add_patch(Rectangle((-0.12, k + 0.05), NJ + 0.24, 0.9, facecolor="none", edgecolor=INK,
                                       linewidth=1.0))
        if kind == "fix":
            ax.add_patch(Rectangle((-0.12, -0.1), 1.24, NK + 0.2, facecolor="none", edgecolor=INK, linewidth=1.0))
            ax.text(1.4, NK + 0.9, "…continues to k = 31", ha="left", va="top", fontsize=5.2, color=INK2)
        ax.set_title(title, loc="left", color=INK, fontsize=7)
        ax.text(-0.5, NK / 2, "k →", rotation=90, ha="right", va="center", fontsize=6, color=INK2)
        ax.text(NJ / 2, NK + 0.9, "columns of B →" if kind != "fix" else "", ha="center", va="top", fontsize=5.4,
                color=INK2)

        # ---- right: memory addresses; one row per load instruction, time going down
        ax = axs[r][1]
        ax.grid(False)
        for j in range(NJ):
            ax.axvspan(j * COLB, (j + 1) * COLB, color="#f4f3ef" if j % 2 == 0 else "#e9e8e2", zorder=0, linewidth=0)
            ax.text((j + 0.5) * COLB, -0.6, f"col {j}", ha="center", va="bottom", fontsize=5.2, color=INK2)
        if kind == "scalar":
            xs = [j * COLB + k * 4 for k, j in seq]
            ys = list(range(1, len(seq) + 1))
            ax.plot(xs, ys, color=col, linewidth=0.7, alpha=0.7, zorder=2)
            ax.scatter(xs, ys, s=9, color=col, edgecolor="white", linewidth=0.4, zorder=3)
        elif kind == "vector-row":
            for k in range(4):
                xs = [j * COLB + k * 4 for j in range(NJ)]
                ax.plot([xs[0], xs[-1]], [k + 1, k + 1], color=col, linewidth=0.7, alpha=0.6, zorder=2)
                ax.scatter(xs, [k + 1] * NJ, s=12, color=col, edgecolor="white", linewidth=0.4, zorder=3)
            ax.annotate("loads 1-4: each sends\n8 requests at once", xy=(5.5 * COLB, 4.6), xytext=(5.5 * COLB, 14),
                        fontsize=5.6, color=INK, ha="center", va="top",
                        arrowprops=dict(arrowstyle="->", color=INK, lw=0.7))
        else:
            ax.add_patch(Rectangle((0, 0.55), 128, 0.9, facecolor=col, edgecolor=INK, linewidth=0.6, zorder=3))
            ax.annotate("load 1: all 32 numbers, bytes 0-127 of column 0", xy=(200, 1), xytext=(1700, 1),
                        fontsize=5.6, color=INK, va="center", arrowprops=dict(arrowstyle="->", color=INK, lw=0.7))
        ax.set_xlim(-300, NJ * COLB + 5600)
        ax.set_ylim(33, -0.9)
        ax.set_xticks([j * COLB for j in range(NJ + 1)])
        ax.set_xticklabels(["0"] + [f"{j * 1.5:g}" for j in range(1, NJ + 1)])
        ax.set_xlabel("address in B (KB); a column is 1536 B", labelpad=1)
        ax.set_ylabel("load, in order ↓")
        ax.set_yticks([1, 8, 16, 24, 32])
        ax.text(NJ * COLB + 450, 16, note, ha="left", va="center", fontsize=5.8, color=INK, zorder=6)
        ax.spines["bottom"].set_bounds(-300, NJ * COLB + 150)
    save(fig, "fig_ref_vs_ours.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- figure 9: the fix, B -> registers -> Bc
def fig_fix_dataflow():
    """One load and one store of the current packer and of the fix, drawn as B -> vector register(s) -> Bc,
    and the finished Bc panel, which is the same for both. NR = 8, the first 32 k of one panel."""
    from matplotlib.patches import FancyArrowPatch, Rectangle

    NK, NJ = 32, 8
    cw, ch = 0.3, 0.15  # cell width / height in data units
    fig, ax = plt.subplots(figsize=(TEXTW, 5.0))
    ax.axis("off")
    ax.set_xlim(-1.2, 16.8)
    ax.set_ylim(-11.9, 1.3)

    def grid(x0, y0, nk, nj, fill, title, sub, lane_of=None, col=None):
        for k in range(nk):
            for j in range(nj):
                f = fill(k, j)
                ax.add_patch(Rectangle((x0 + j * cw, y0 - (k + 1) * ch), cw, ch,
                                       facecolor=f or "#f4f3ef", edgecolor="white", linewidth=0.4))
                if f and lane_of:
                    n = lane_of(k, j)
                    if n in (1, 8, 16, 24, 32) or (NJ == 8 and lane_of(k, j) <= 8 and nk_label_all):
                        ax.text(x0 + (j + 0.5) * cw, y0 - (k + 0.5) * ch, str(n), ha="center", va="center",
                                fontsize=3.8, color="white")
        ax.text(x0, y0 + 0.42, title, fontsize=6.6, color=INK, va="bottom", fontweight="bold")
        ax.text(x0, y0 + 0.08, sub, fontsize=5.6, color=INK2, va="bottom")

    def register(x0, y0, lanes, rows, col, label):
        lw = 0.3
        for r in range(rows):
            for e in range(lanes):
                n = r * lanes + e + 1
                ax.add_patch(Rectangle((x0 + e * lw, y0 - (r + 1) * 0.3), lw, 0.3, facecolor=col,
                                       edgecolor="white", linewidth=0.5))
                if lanes * rows <= 8 or n in (1, 8, 16, 24, 32):
                    ax.text(x0 + (e + 0.5) * lw, y0 - (r + 0.5) * 0.3, str(n), ha="center", va="center",
                            fontsize=4.6, color="white")
            ax.add_patch(Rectangle((x0, y0 - (r + 1) * 0.3), lanes * lw, 0.3, facecolor="none",
                                   edgecolor=INK, linewidth=0.7))
            if rows > 1:
                ax.text(x0 + lanes * lw + 0.08, y0 - (r + 0.5) * 0.3, f"v{4 + r}", fontsize=5.2, va="center",
                        color=INK2)
        ax.text(x0, y0 + 0.08, label, fontsize=5.6, color=INK2, va="bottom")

    def arrow(x0, y0, x1, y1, txt):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, color=INK, lw=0.8))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, txt, fontsize=5.4, ha="center", va="bottom", color=INK)

    global nk_label_all
    rows = [
        ("Current packer: one row at a time", ORANGE, lambda k, j: k == 0, lambda k, j: j + 1,
         "vlse32: 8 numbers, one from each\ncolumn, 1536 B apart", 8, 1, "one register (VLEN 256 b = 8 × fp32)",
         "vse32: side by side\ninto row 0 of Bc", True),
        ("The fix, col-g1-m4: one column at a time", BLUE, lambda k, j: j == 0, lambda k, j: k + 1,
         "vle32 (LMUL 4): 32 numbers, down\ncolumn 0, 128 B in one piece", 8, 4, "4 registers joined (LMUL 4 = 32 × fp32)",
         "vsse32: down column 0\nof Bc, one every 32 B", False),
    ]
    for r, (title, col, sel, lane, ld_txt, lanes, regs, reg_lbl, st_txt, all_lbl) in enumerate(rows):
        y0 = -r * 5.9
        nk_label_all = all_lbl
        ax.text(-1.1, y0 + 1.0, title, fontsize=7.5, color=INK, fontweight="bold", va="bottom")
        grid(0, y0, NK, NJ, lambda k, j: col if sel(k, j) else None, "B (column by column)",
             "first 32 k of 8 columns", lane)
        register(4.4, y0 - 1.0, lanes, regs, col, reg_lbl)
        grid(9.6, y0, NK, NJ, lambda k, j: col if sel(k, j) else None, "Bc panel (row by row)",
             "Bc[k*8 + j] = B(k, j)", lane)
        arrow(2.55, y0 - 1.6, 4.3, y0 - 1.6, "load")
        ax.text(3.55, y0 - 3.0, ld_txt, fontsize=5.2, ha="center", va="top", color=INK2)
        arrow(4.4 + lanes * 0.3 + (0.45 if regs > 1 else 0.1), y0 - 1.6, 9.5, y0 - 1.6, "store")
        ax.text(7.6, y0 - 3.0, st_txt, fontsize=5.2, ha="center", va="top", color=INK2)
        ax.text(-1.1, y0 - NK * ch / 2, "k →", rotation=90, fontsize=5.6, va="center", color=INK2)

    # the finished panel: the same Bc from both, coloured by the column of B each number came from
    x0, y0, w = 13.6, 0.0, 0.14
    ax.text(x0, y0 + 0.95, "After the whole panel", fontsize=6.6, color=INK, fontweight="bold", va="bottom")
    ax.text(x0, y0 + 0.62, "colour = the column of B a\nnumber came from", fontsize=5.6, color=INK2, va="top")
    for g, (name, gx) in enumerate([("current", x0), ("fix", x0 + NJ * w + 0.55)]):
        for k in range(NK * 2):
            for j in range(NJ):
                ax.add_patch(Rectangle((gx + j * w, y0 - 0.2 - (k + 1) * ch), w, ch, facecolor=RAMP[2 + j],
                                       edgecolor="white", linewidth=0.25))
        ax.text(gx + NJ * w / 2, y0 - 0.35 - NK * 2 * ch, name, ha="center", va="top", fontsize=6, color=INK)
    ax.text(x0 + NJ * w + 0.27, y0 - 0.2 - NK * ch, "=", ha="center", va="center", fontsize=12, color=INK,
            fontweight="bold")
    ax.text(x0 + NJ * w + 0.27, y0 - 0.85 - NK * 2 * ch, "the same Bc,\nbyte for byte", ha="center", va="top",
            fontsize=6.2, color=INK, fontweight="bold")
    save(fig, "fig_fix_dataflow.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- short-version figures: current packer and fix only
SHORT = ["row", "col-g1-m4"]
SHORT_NAME = {"row": "our current packer (xDSL)", "col-g1-m4": "the fix (one column at a time)"}


def _short_legend(fig, y=1.0):
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[v]) for v in SHORT]
    fig.legend(handles, [SHORT_NAME[v] for v in SHORT], loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, y))


def fig_short_reps():
    fig, axs = plt.subplots(1, 2, figsize=(TEXTW, 1.9), sharey=True, gridspec_kw={"wspace": 0.06})
    ops = GEMM_OPS["B3"]
    for ax, v, run in [(axs[0], "row", "row"), (axs[1], "col-g1-m4", "col-g1-m4")]:
        secs = gemm_reps(S2, 1, run, "16x15", "B3")
        gops = [ops / (s * 1e9) for s in secs]
        mask = slow_reps(ops, secs)
        share = 100 * sum(mask[1:]) / max(1, len(mask) - 1)
        t, acc = [], 0.0
        for s in secs:
            acc += s
            t.append(acc)
        ax.scatter(t, gops, s=0.6, color=COLOR[v], linewidths=0, rasterized=True)
        ax.set_title(f"{SHORT_NAME[v]}\n{share:.0f}% of reps slow, {len(secs) * ops / (sum(secs) * 1e9):.2f} GOPs",
                     loc="left", color=INK)
        ax.set_xlabel("time into the run (s)")
        ax.set_ylim(0, 10)
    axs[0].set_ylabel("GOPs of one rep")
    save(fig, "fig_short_reps.pdf", dpi=300)
    plt.close(fig)


def _short_cells():
    groups, keys = [], []
    for shape in ["B3", "B3_K256"]:
        for t in TILES:
            groups.append(f"{shape}\n{tile_tex(t)}" + ("\n(default tile)" if t == "16x15" else ""))
            keys.append((t, shape))
    return groups, keys


def fig_short_slow():
    groups, keys = _short_cells()
    vals = {v: [cell(R1, t, s, v)[2] for t, s in keys] for v in SHORT}
    fig, ax = plt.subplots(figsize=(TEXTW, 2.4))
    w = 0.36
    for vi, v in enumerate(SHORT):
        for gi in range(len(groups)):
            x = gi + (vi - 0.5) * w
            y = vals[v][gi]
            ax.bar(x, max(y, 0.6), width=w - 0.03, color=COLOR[v], edgecolor="white", linewidth=0.6, zorder=2)
            ax.text(x, max(y, 0.6) + 2, f"{y:.0f}%", ha="center", va="bottom", fontsize=6, color=INK)
            if y == 0:  # no slow rep: say what that means, so a short bar is read as good
                ax.text(x, 12, "no slow state:\nfull speed", ha="center", va="bottom", rotation=90, fontsize=5.2,
                        color=COLOR[v], linespacing=1.0)
            elif y >= 40:  # a tall bar is bad: say so, and why, inside the bar
                label = "slow state: poor speed\nscattered 4-byte reads" if y >= 70 else "slow: poor\nspeed"
                ax.text(x, y / 2, label, ha="center", va="center", rotation=90,
                        fontsize=5.0, color="white", fontweight="bold", linespacing=1.0)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 112)
    ax.set_ylabel("slow reps (%)\nlower is better")
    ax.text(0.0, 1.0, "A slow rep runs at about 4.3 GOPs instead of about 7: the more slow reps, the slower the layer.",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6, color=INK2)
    ax.axvline(2.5, color=AXIS, linewidth=0.6)
    _short_legend(fig, 1.05)
    save(fig, "fig_short_slow.pdf")
    plt.close(fig)


def fig_short_perf():
    fig, (a, b) = plt.subplots(2, 1, figsize=(TEXTW, 3.7), gridspec_kw={"hspace": 0.75})
    groups, keys = _short_cells()
    vals = {v: [cell(R1, t, s, v)[0] for t, s in keys] for v in SHORT}
    slow = {v: [cell(R1, t, s, v)[2] for t, s in keys] for v in SHORT}
    bars(a, groups, SHORT, vals, slow, 9.0)
    a.set_title("(a) The layer under test: B3 and B3_K256", loc="left", color=INK)
    a.axvline(2.5, color=AXIS, linewidth=0.6)
    groups, keys = [], []
    for shape, label in [("B2", "B2"), ("R24", "Reuse24")]:
        for t in TILES:
            groups.append(f"{label}\n{tile_tex(t)}")
            keys.append((t, shape))
    vals = {v: [cell(R1, t, s, v)[0] for t, s in keys] for v in SHORT}
    slow = {v: [cell(R1, t, s, v)[2] for t, s in keys] for v in SHORT}
    bars(b, groups, SHORT, vals, slow, 14.0, fs=5.2)
    b.set_title("(b) Other layers: nothing gets worse", loc="left", color=INK)
    b.axvline(2.5, color=AXIS, linewidth=0.6)
    _short_legend(fig, 1.02)
    save(fig, "fig_short_perf.pdf")
    plt.close(fig)


def fig_short_units():
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(TEXTW, 1.55))
    ax.set_xlim(-46, 136)
    ax.set_ylim(4.95, -0.9)
    ax.axis("off")
    rows = [
        ("one vector register\nVLEN = 256 bit = 32 B", [(0, 32, 8, BLUE)], "8 numbers (fp32)"),
        ("4 registers joined (LMUL = 4)\n= 128 B", [(32 * r, 32, 8, BLUE) for r in range(4)], "32 numbers"),
        ("one cache line = 64 B", [(0, 64, 16, MUTED), (64, 64, 16, MUTED)], "16 numbers per line"),
        ("current packer: from one\ncolumn per touch", [(0, 4, 1, ORANGE)], "4 B, then jump to the next column"),
        ("the fix: from one\ncolumn per touch", [(0, 128, 32, BLUE)], "128 B = two whole lines"),
    ]
    for r, (label, segs, note) in enumerate(rows):
        y = r * 1.0
        ax.text(-2, y + 0.3, label, ha="right", va="center", fontsize=6.3, color=INK)
        ax.add_patch(Rectangle((0, y), 128, 0.6, facecolor="white", edgecolor=GRID, linewidth=0.5))
        for x0, w, n, col in segs:
            for e in range(n):
                ax.add_patch(Rectangle((x0 + e * w / n, y), w / n, 0.6, facecolor=col, edgecolor="white",
                                       linewidth=0.5, alpha=0.9))
            ax.add_patch(Rectangle((x0, y), w, 0.6, facecolor="none", edgecolor=INK, linewidth=0.8))
        ax.text(129.5, y + 0.3, note, ha="left", va="center", fontsize=6.0, color=INK2)
    for bb in range(0, 129, 32):
        ax.text(bb, -0.3, f"{bb} B", ha="center", va="bottom", fontsize=5.5, color=MUTED)
    save(fig, "fig_short_units.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- figure 10: before / after at the C code level
def fig_code_diff():
    """The current read order and the fix as C intrinsics, read verbatim from templates/pack_b_variants.h, with the
    three changed places marked 1-3, and next to each a picture of the first loads it issues on B."""
    from matplotlib.patches import FancyBboxPatch, Rectangle

    hdr = (TESTS / "templates" / "pack_b_variants.h").read_text().splitlines()

    def take(a, b):
        return [(n, hdr[n - 1]) for n in range(a, b + 1)]

    blocks = [
        ("Before: the current read order (as C intrinsics; the same order as the xDSL packer)", ORANGE, "#fde3d6",
         take(93, 98) + [(None, "")] + take(151, 160),
         {155: 1, 156: 1, 96: 2, 95: 2, 97: 3, 157: 1}, "row"),
        ("After: the fix, col-g1-m4", BLUE, "#d7e6fa",
         take(117, 122) + [(None, "")] + take(196, 205),
         {200: 1, 201: 1, 119: 2, 120: 2, 121: 3, 202: 1}, "fix"),
    ]
    notes = {
        "row": ["1  loop: k outside, columns inside",
                "2  load: vlse32, m1 = 1 register = 8 numbers; stride ldb*4 bytes = 384 x 4 B = 1536 B",
                "   (one number from each of 8 columns)",
                "3  store: vse32, 8 numbers side by side in a Bc row"],
        "fix": ["1  loop: column outside, k (32 at a time) inside",
                "2  load: vle32, m4 = 4 registers joined = 32 numbers, side by side",
                "   (32 neighbours in one column, 128 B)",
                "3  store: vsse32, stride RR*4 bytes (NR numbers) down a Bc column"],
    }
    lh = 0.105  # line height, inches-ish in axes units below
    fig = plt.figure(figsize=(TEXTW, 6.1))
    tops = [0.985, 0.49]
    for (title, col, bg, lines, marks, kind), top in zip(blocks, tops):
        axc = fig.add_axes([0.0, top - 0.43, 0.74, 0.43])
        axc.axis("off")
        axc.set_xlim(0, 1)
        n = len(lines) + 2
        axc.set_ylim(n, 0)
        axc.text(0.0, 0.6, title, fontsize=7.2, fontweight="bold", color=col, va="center")
        for r, (ln, txt) in enumerate(lines):
            y = r + 1.6
            m = marks.get(ln)
            if m:
                axc.add_patch(Rectangle((0.075, y - 0.48), 0.925, 0.96, facecolor=bg, edgecolor="none", zorder=0))
                axc.text(0.995, y, str(m), fontsize=6.4, ha="right", va="center", color="white", fontweight="bold",
                         bbox=dict(boxstyle="circle,pad=0.18", facecolor=col, edgecolor="none"), zorder=3)
            if ln is not None:
                axc.text(0.0, y, f"{ln:4d}", fontsize=4.9, family="DejaVu Sans Mono", color=MUTED, va="center")
                axc.text(0.08, y, txt.replace("\t", "    "), fontsize=4.9, family="DejaVu Sans Mono", color=INK,
                         va="center")
        # notes under the code
        for k, t in enumerate(notes[kind]):
            axc.text(0.08, len(lines) + 2.0 + k * 0.95, t, fontsize=5.8, color=INK, va="center")
        axc.set_ylim(len(lines) + 2.0 + 4 * 0.95, 0)

        # picture: the first loads on B (8 columns x 32 k), numbered in issue order
        axd = fig.add_axes([0.775, top - 0.40, 0.22, 0.37])
        axd.axis("off")
        NK, NJ = 32, 8
        axd.set_xlim(-0.3, NJ + 0.3)
        axd.set_ylim(NK + 2.5, -2.2)
        for k in range(NK):
            for j in range(NJ):
                axd.add_patch(Rectangle((j, k), 1, 1, facecolor="#f4f3ef", edgecolor="white", linewidth=0.4))
        shades = ["#8a2f0c", "#eb6834", "#f6b89c"] if kind == "row" else ["#0d366b", "#3987e5", "#9ec5f4"]
        loads = ([[(k, j) for j in range(NJ)] for k in range(3)] if kind == "row"
                 else [[(k, j) for k in range(NK)] for j in range(3)])
        for li, cells in enumerate(loads):
            for k, j in cells:
                axd.add_patch(Rectangle((j, k), 1, 1, facecolor=shades[li], edgecolor="white", linewidth=0.4))
        for j in range(NJ):
            for k0 in (0, 16):
                axd.add_patch(Rectangle((j + 0.06, k0 + 0.06), 0.88, 15.88, facecolor="none", edgecolor=INK2,
                                        linewidth=0.5))
        axd.text(0, -0.6, "B: first 3 loads", fontsize=6, color=INK, va="bottom")
        axd.text(0, NK + 0.8, "columns →  (box = one 64 B line)", fontsize=5.2, color=INK2, va="top")
        axd.text(-0.5, NK / 2, "k →", rotation=90, fontsize=5.6, color=INK2, ha="right", va="center")
    save(fig, "fig_code_diff.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- reference design with and without the fix (issue54_ref series)
REF_SERIES = LOGS / "issue54_ref_2026-10-29_123319"


def ref_series_reps():
    """{(arm, tile): [per-launch list of B3 per-rep seconds]} from the issue54_ref series (timed launches only)."""
    out: dict[tuple[str, str], list[list[float]]] = {}
    path = REF_SERIES / "runs.tsv"
    if not path.exists():
        return out
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["status"] != "ok" or r["launch"] == "0" or r["arm"] not in ("ref", "ref-fix"):
                continue
            for tile in ("16x14", "16x16"):
                p = LOGS / r["result_dir"] / f"reps_ref_{tile}_macc.txt.gz"
                if p.exists():
                    with gzip.open(p, "rt") as g:
                        secs = [float(x["Seconds"]) for x in csv.DictReader(g) if x["Layer"] == "B3"]
                    out.setdefault((r["arm"], tile), []).append(secs)
    return out


def _launch_stats(launches, ops):
    """mean GOPs by mean time over launches, (lo, hi), mean slow share %"""
    g = [len(s) * ops / (sum(s) * 1e9) for s in launches]
    sh = []
    for s in launches:
        m = slow_reps(ops, s)
        sh.append(100 * sum(m[1:]) / max(1, len(m) - 1))
    return statistics.fmean(g), (min(g), max(g)), statistics.fmean(sh)


def fig_ref_compare():
    ops = GEMM_OPS["B3"]
    ref = ref_series_reps()
    groups = [("Our design (xDSL)\n16×15, series 1", "ours", None),
              ("Reference design\n16×14 (today)", "ref", "16x14"),
              ("Reference design\n16×16 (today)", "ref", "16x16")]
    fig, ax = plt.subplots(figsize=(TEXTW, 2.4))
    w = 0.36
    for gi, (label, kind, tile) in enumerate(groups):
        for vi, fixed in enumerate((False, True)):
            x = gi + (vi - 0.5) * w
            col = BLUE if fixed else ORANGE
            if kind == "ours":
                g, _, sl = cell(R1, "16x15", "B3", "col-g1-m4" if fixed else "row")
                rng, n = None, 3
            else:
                launches = ref.get(("ref-fix" if fixed else "ref", tile), [])
                if not launches:
                    ax.bar(x, 8.2, width=w - 0.03, color="none", edgecolor=MUTED, linestyle="--", linewidth=0.8, zorder=2)
                    ax.text(x, 4.1, "pending", rotation=90, ha="center", va="center", fontsize=6, color=MUTED)
                    continue
                g, rng, sl = _launch_stats(launches, ops)
                n = len(launches)
            ax.bar(x, g, width=w - 0.03, color=col, edgecolor="white", linewidth=0.6, zorder=2)
            if rng and n > 1:
                ax.plot([x, x], rng, color=INK, linewidth=0.8, zorder=3)
            ax.text(x, g + 0.25, f"{g:.2f}", ha="center", va="bottom", fontsize=6, color=INK)
            inner = f"{sl:.0f}% slow" if sl >= 1 else "no slow state"
            ax.text(x, g / 2, inner, rotation=90, ha="center", va="center", fontsize=5.4, color="white", fontweight="bold")
            if n < 3:
                ax.text(x, 0.3, f"{n} of 3", ha="center", va="bottom", fontsize=4.8, color="white")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 9.2)
    ax.set_ylabel("B3 GOPs (by mean time)\nhigher is better")
    ax.axvline(0.5, color=AXIS, linewidth=0.6)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ORANGE), plt.Rectangle((0, 0), 1, 1, color=BLUE)]
    fig.legend(handles, ["as it is (row-by-row B packer)", "with the fix (one column at a time)"], loc="upper center",
               ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    save(fig, "fig_ref_compare.pdf")
    plt.close(fig)


def fig_ref_reps():
    """Every rep of launch 1, B3, reference 16x14: as it is and with the fix (as fig_short_reps for ours)."""
    ops = GEMM_OPS["B3"]
    ref = ref_series_reps()
    fig, axs = plt.subplots(1, 2, figsize=(TEXTW, 1.9), sharey=True, gridspec_kw={"wspace": 0.06})
    for ax, arm, col, name in [(axs[0], "ref", ORANGE, "reference design, as it is"),
                               (axs[1], "ref-fix", BLUE, "reference design, with the fix")]:
        launches = ref.get((arm, "16x14"), [])
        if not launches:
            ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center", va="center", color=MUTED)
            continue
        secs = launches[0]
        gops = [ops / (s * 1e9) for s in secs]
        mask = slow_reps(ops, secs)
        share = 100 * sum(mask[1:]) / max(1, len(mask) - 1)
        t, acc = [], 0.0
        for s in secs:
            acc += s
            t.append(acc)
        ax.scatter(t, gops, s=0.6, color=col, linewidths=0, rasterized=True)
        ax.set_title(f"{name} (16×14)\n{share:.0f}% of reps slow, {len(secs) * ops / (sum(secs) * 1e9):.2f} GOPs",
                     loc="left", color=INK)
        ax.set_xlabel("time into the run (s)")
        ax.set_ylim(0, 10)
    axs[0].set_ylabel("GOPs of one rep")
    save(fig, "fig_ref_reps.pdf", dpi=300)
    plt.close(fig)


nk_label_all = False

if __name__ == "__main__":
    fig_ref_compare()
    fig_ref_reps()
    fig_code_diff()
    fig_short_reps()
    fig_short_slow()
    fig_short_perf()
    fig_short_units()
    fig_fix_dataflow()
    fig_ref_vs_ours()
    fig_units()
    fig_read_order()
    for v, s in fig_counts().items():
        print(v, {k.replace(chr(10), " "): round(x, 2) for k, x in s.items()})
    fig_headline()
    fig_reps()
    fig_heatmap()
    fig_pack_cost()
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("fig_*.pdf"))))
