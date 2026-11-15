"""Three simple figures for the short guide to the issue #54 fix, in English and Spanish.

    .venv/bin/python .scratch/reports/issue-54-figs/make_simple.py

Writes simple/en/*.pdf and simple/es/*.pdf. The two result bars use the committed series numbers quoted in
issue-54-brief.tex (Table 6 and the reference series of 2026-10-29); everything else is a drawing of the data layout.
"""
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parent / "simple"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#e9e8e2"
INK, INK2, EDGE = "#0b0b0b", "#52514e", "#b5b3ab"
MONO = {"family": "monospace"}

T = {
    "en": {
        "cur": "Current packer: reads across columns",
        "fix": "The fix: reads down one column",
        "cur1": "1 load = 8 numbers,\n1 from each of 8 columns\n= 8 cache lines touched,\n4 bytes used from each",
        "fix1": "1 load = 32 numbers,\nall from one column\n= 128 bytes = 2 cache lines,\nevery byte used",
        "k": "k (rows of B)", "cols": "columns of B (each one is contiguous in memory)",
        "m1": "m1: 1 register\n8 numbers (32 B)", "m4": "m4: 4 registers joined\n32 numbers (128 B)",
        "lane": "lanes", "reg": "Vector register(s), VLEN = 256 bits",
        "bc": "Bc (the packed copy of B)", "bcrow": "k", "bccol": "column in the panel",
        "st": "vsse32: one number per row of Bc,\nstep RR*4 bytes (4 = bytes per fp32)",
        "gops": "GOPs (higher is better)", "ours": "Our design\n(16x15 tile)", "ref": "Reference design\n(16x14 tile)",
        "before": "before the fix", "after": "with the fix", "slow": "slow", "dec": ".",
    },
    "es": {
        "cur": "Empaquetador actual: lee entre columnas",
        "fix": "La solución: lee una columna entera",
        "cur1": "1 carga = 8 números,\n1 de cada una de 8 columnas\n= se tocan 8 líneas de caché,\nse usan 4 bytes de cada una",
        "fix1": "1 carga = 32 números,\ntodos de una columna\n= 128 bytes = 2 líneas de caché,\nse usan todos los bytes",
        "k": "k (filas de B)", "cols": "columnas de B (cada una es contigua en memoria)",
        "m1": "m1: 1 registro\n8 números (32 B)", "m4": "m4: 4 registros unidos\n32 números (128 B)",
        "lane": "carriles", "reg": "Registro(s) vectorial(es), VLEN = 256 bits",
        "bc": "Bc (la copia empaquetada de B)", "bcrow": "k", "bccol": "columna del panel",
        "st": "vsse32: un número por fila de Bc,\npaso RR*4 bytes (4 = bytes por fp32)",
        "gops": "GOPs (más es mejor)", "ours": "Nuestro diseño\n(tesela 16x15)", "ref": "Diseño de referencia\n(tesela 16x14)",
        "before": "antes de la solución", "after": "con la solución", "slow": "lentas", "dec": ",",
    },
}


def grid(ax, x0, y0, ncol, nrow, hl, colour, cell=1.0):
    """ncol x nrow cells, hl = set of (col, row) to colour."""
    for c in range(ncol):
        for r in range(nrow):
            on = (c, r) in hl
            ax.add_patch(Rectangle((x0 + c * cell, y0 - (r + 1) * cell), cell, cell,
                                   fc=colour if on else GREY, ec="white", lw=0.8))


def fig_before_after(lang):
    L = T[lang]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.7))
    rows, cols = 36, 8
    for ax, which in zip(axs, ("cur", "fix")):
        hl = {(c, 0) for c in range(cols)} if which == "cur" else {(0, r) for r in range(32)}
        grid(ax, 0, 0, cols, rows, hl, ORANGE if which == "cur" else BLUE)
        ax.set_xlim(-2.2, cols + 0.3)
        ax.set_ylim(-rows - 6.5, 3.2)
        ax.axis("off")
        ax.set_title(L[which], fontsize=9, color=INK, pad=2, fontweight="bold")
        ax.text(-0.6, -rows / 2, L["k"], rotation=90, ha="right", va="center", fontsize=8, color=INK2)
        ax.text(cols / 2, 0.7, L["cols"], ha="center", va="bottom", fontsize=6.3, color=INK2)
        ax.text(cols / 2, -rows - 1.0, L[which + "1"], ha="center", va="top", fontsize=7.6, color=INK)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(OUT / lang / "fig_simple_read.pdf")
    plt.close(fig)


def fig_lmul_store(lang):
    L = T[lang]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw={"width_ratios": [1.25, 1]})
    # left: m1 against m4
    a.set_xlim(0, 33)
    a.set_ylim(-9, 6)
    a.axis("off")
    a.set_title(L["reg"], fontsize=8.5, color=INK, pad=2, fontweight="bold")
    shades = ["#2a78d6", "#5598e7", "#2a78d6", "#5598e7"]
    for i in range(8):
        a.add_patch(Rectangle((i, 1.5), 1, 1.5, fc=ORANGE, ec="white", lw=0.8))
    a.text(0, 3.4, L["m1"], fontsize=7.6, va="bottom", color=INK)
    for g in range(4):
        for i in range(8):
            a.add_patch(Rectangle((g * 8 + i, -4), 1, 1.5, fc=shades[g], ec="white", lw=0.8))
    a.text(0, -1.9, L["m4"], fontsize=7.6, va="bottom", color=INK)
    a.text(16, -8, "32 " + L["lane"] + " = 4 x 8", fontsize=7, ha="center", color=INK2)
    # right: strided store into Bc
    nc, nr = 8, 34
    grid(b, 0, 0, nc, nr, {(0, r) for r in range(32)}, BLUE, cell=1.0)
    b.set_xlim(-3, nc + 0.5)
    b.set_ylim(-nr - 6.5, 4.5)
    b.axis("off")
    b.set_title(L["bc"], fontsize=8.5, color=INK, pad=2, fontweight="bold")
    b.text(-0.6, -nr / 2, L["bcrow"], rotation=90, ha="right", va="center", fontsize=8, color=INK2)
    b.text(nc / 2, 0.7, L["bccol"], ha="center", va="bottom", fontsize=6.5, color=INK2)
    for r in (0, 1, 2):
        b.add_patch(FancyArrowPatch((-2.4, -r - 0.5), (-0.1, -r - 0.5), arrowstyle="-|>", mutation_scale=7, color=INK2, lw=0.7))
    b.text(nc / 2, -nr - 1.0, L["st"], ha="center", va="top", fontsize=7.4, color=INK)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / lang / "fig_simple_lmul_store.pdf")
    plt.close(fig)


def fig_result(lang):
    L = T[lang]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    data = [(L["ours"], 4.63, 86, 7.03, 0), (L["ref"], 4.49, 81, 7.14, 0)]
    w = 0.34
    for i, (name, b0, s0, b1, s1) in enumerate(data):
        for dx, v, s, col in ((-w / 2 - 0.02, b0, s0, ORANGE), (w / 2 + 0.02, b1, s1, BLUE)):
            ax.bar(i + dx, v, w, color=col)
            ax.text(i + dx, v + 0.12, f"{v:.2f}".replace(".", L["dec"]), ha="center", fontsize=8.5, color=INK, fontweight="bold")
            ax.text(i + dx, v / 2, f"{s}%\n{L['slow']}", ha="center", va="center", fontsize=7.5, color="white")
        ax.text(i, 8.35, f"+{round((b1 / b0 - 1) * 100)}%", ha="center", fontsize=10, color=BLUE, fontweight="bold")
    ax.set_xticks(range(2))
    ax.set_xticklabels([d[0] for d in data], fontsize=8)
    ax.set_ylim(0, 9.3)
    ax.set_ylabel(L["gops"], fontsize=8)
    ax.tick_params(labelsize=8, length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(EDGE)
    ax.spines["bottom"].set_color(EDGE)
    ax.yaxis.grid(True, color="#e1e0d9", lw=0.6)
    ax.set_axisbelow(True)
    h = [Rectangle((0, 0), 1, 1, fc=ORANGE), Rectangle((0, 0), 1, 1, fc=BLUE)]
    ax.legend(h, [L["before"], L["after"]], fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / lang / "fig_simple_result.pdf")
    plt.close(fig)


if __name__ == "__main__":
    for lang in T:
        (OUT / lang).mkdir(parents=True, exist_ok=True)
        fig_before_after(lang)
        fig_lmul_store(lang)
        fig_result(lang)
    print("wrote", sorted(p.name for p in (OUT / "en").glob("*.pdf")))
