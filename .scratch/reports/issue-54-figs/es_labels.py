"""Spanish labels for the issue #54 figures (make_figs.save writes the Spanish copy to es/).

tr(s) translates one figure text: first an exact match in EXACT, then the phrase rules of PHRASES (longest first), then
the decimal point becomes a comma. Code text is never passed here (make_figs skips monospace text).
"""
import re

EXACT = {
    "=": "=",
    "rep": "repetición",
    "load": "carga",
    "store": "escritura",
    "scalar": "scalar",  # a variant name, as in the text
    "current": "actual",
    "fix": "solución",
    "1st load": "1.ª carga",
    "2nd load": "2.ª carga",
    "3rd load": "3.ª carga",
    "k →": "k →",
    "k (0..31)  →": "k (0..31)  →",
    "no slow state": "sin estado lento",
    "no slow state:\nfull speed": "sin estado lento:\nvelocidad plena",
    "slow: poor\nspeed": "lento: poca\nvelocidad",
    "slow state: poor speed\nscattered 4-byte reads": "estado lento: poca velocidad\nlecturas de 4 bytes dispersas",
    "slow reps (%)\nlower is better": "repeticiones lentas (%)\ncuanto menos, mejor",
    "A slow rep runs at about 4.3 GOPs instead of about 7: the more slow reps, the slower the layer.":
        "Una repetición lenta va a unos 4,3 GOPs en vez de unos 7: cuantas más repeticiones lentas, más lenta la capa.",
    "GOPs (by mean time)": "GOPs (por tiempo medio)",
    "GOPs by mean time (darker is faster)": "GOPs por tiempo medio (más oscuro es más rápido)",
    "GOPs of one rep": "GOPs de una repetición",
    "B3 GOPs (by mean time)\nhigher is better": "GOPs en B3 (por tiempo medio)\ncuanto más, mejor",
    "time into the run (s)": "tiempo de ejecución transcurrido (s)",
    "pack time of one rep (ms)": "tiempo de empaquetado de una repetición (ms)",
    "mean pack time per rep, all reps (ms)": "tiempo medio de empaquetado por repetición, todas (ms)",
    "mean of the fast reps only": "media de las repeticiones rápidas solamente",
    "address in B (KB); a column is 1536 B": "dirección en B (KB); una columna ocupa 1536 B",
    "load, in order ↓": "carga, en orden ↓",
    "columns of B →": "columnas de B →",
    "columns →  (box = one 64 B line)": "columnas →  (recuadro = una línea de 64 B)",
    "…continues to k = 31": "…sigue hasta k = 31",
    "B: first 3 loads": "B: 3 primeras cargas",
    "one 64 B cache line": "una línea de caché de 64 B",
    "first 32 k of 8 columns": "primeros 32 k de 8 columnas",
    "B (column by column)": "B (por columnas)",
    "Bc panel (row by row)": "panel Bc (por filas)",
    "writes into Bc": "escribe en Bc",
    "After the whole panel": "Con el panel terminado",
    "colour = the column of B a\nnumber came from": "color = la columna de B de la\nque viene cada número",
    "the same Bc,\nbyte for byte": "el mismo Bc,\nbyte a byte",
    "take 4 B\nper column": "toman 4 B\npor columna",
    "take whole\ncolumns\n(32 or 128 B)": "toman columnas\nenteras\n(32 o 128 B)",
    # register / line sizes
    "one vector register\nVLEN = 256 bit = 32 B": "un registro vectorial\nVLEN = 256 bits = 32 B",
    "register group, LMUL = 4\nv4–v7 = 128 B": "grupo de registros, LMUL = 4\nv4–v7 = 128 B",
    "cache line (L1D and L2)\n64 B": "línea de caché (L1D y L2)\n64 B",
    "row, row-intr, scalar:\nbytes from one column": "row, row-intr, scalar:\nbytes de una columna",
    "col-g1, -g2, -g4, -gall:\nbytes from one column": "col-g1, -g2, -g4, -gall:\nbytes de una columna",
    "col-g1-m4:\nbytes from one column": "col-g1-m4:\nbytes de una columna",
    "8 × fp32 (LMUL = 1)": "8 × fp32 (LMUL = 1)",
    "32 × fp32 (col-g1-m4)": "32 × fp32 (col-g1-m4)",
    "16 × fp32 per line": "16 × fp32 por línea",
    "4 B, then the next column": "4 B, y luego la siguiente columna",
    "32 B = half a line": "32 B = media línea",
    "128 B = two whole lines": "128 B = dos líneas enteras",
    "8 numbers (fp32)": "8 números (fp32)",
    "32 numbers": "32 números",
    "16 numbers per line": "16 números por línea",
    "4 registers joined (LMUL = 4)\n= 128 B": "4 registros unidos (LMUL = 4)\n= 128 B",
    "one cache line = 64 B": "una línea de caché = 64 B",
    "current packer: from one\ncolumn per touch": "empaquetador actual: de una\ncolumna por acceso",
    "the fix: from one\ncolumn per touch": "la solución: de una\ncolumna por acceso",
    "4 B, then jump to the next column": "4 B, y luego salto a la siguiente columna",
    # counts figure
    "load instructions\nper panel": "instrucciones de carga\npor panel",
    "B lines touched\nby one load": "líneas de B tocadas\npor una carga",
    "bytes taken from one\ncolumn per touch (C)": "bytes tomados de una\ncolumna por acceso (C)",
    "B columns read\nside by side (B)": "columnas de B leídas\na la vez (B)",
    "Bc lines touched\nby one store": "líneas de Bc tocadas\npor una escritura",
    # read order figure
    "row (current), row-intr\nreads from B": "row (actual), row-intr\nlee de B",
    "col-g1\nreads from B": "col-g1\nlee de B",
    "col-g2 (col-g4, col-gall alike)\nreads from B": "col-g2 (igual col-g4, col-gall)\nlee de B",
    "col-g1-m4 (the fix)\nreads from B": "col-g1-m4 (la solución)\nlee de B",
    "vlse32, stride 1536 B: 8 lines,\n4 B from each": "vlse32, salto de 1536 B: 8 líneas,\n4 B de cada una",
    "vle32 (LMUL 1): 32 B of one\ncolumn, half a line": "vle32 (LMUL 1): 32 B de una\ncolumna, media línea",
    "same loads as col-g1, but 2\ncolumns before any store": "mismas cargas que col-g1, pero 2\ncolumnas antes de escribir",
    "vle32 (LMUL 4): 128 B of one\ncolumn, two whole lines": "vle32 (LMUL 4): 128 B de una\ncolumna, dos líneas enteras",
    "vse32: 32 B in a row, 1 line": "vse32: 32 B en una fila, 1 línea",
    "vsse32, stride 32 B: 8 rows,\n4 lines": "vsse32, salto de 32 B: 8 filas,\n4 líneas",
    "same stores as col-g1": "mismas escrituras que col-g1",
    "vsse32, stride 32 B: 32 rows,\n16 lines": "vsse32, salto de 32 B: 32 filas,\n16 líneas",
    # data-flow figure
    "Current packer: one row at a time": "Empaquetador actual: una fila cada vez",
    "The fix, col-g1-m4: one column at a time": "La solución, col-g1-m4: una columna cada vez",
    "one register (VLEN 256 b = 8 × fp32)": "un registro (VLEN 256 bits = 8 × fp32)",
    "4 registers joined (LMUL 4 = 32 × fp32)": "4 registros unidos (LMUL 4 = 32 × fp32)",
    "vlse32: 8 numbers, one from each\ncolumn, 1536 B apart": "vlse32: 8 números, uno de cada\ncolumna, a 1536 B entre sí",
    "vse32: side by side\ninto row 0 of Bc": "vse32: seguidos\nen la fila 0 de Bc",
    "vle32 (LMUL 4): 32 numbers, down\ncolumn 0, 128 B in one piece": "vle32 (LMUL 4): 32 números, bajando\npor la columna 0, 128 B de una vez",
    "vsse32: down column 0\nof Bc, one every 32 B": "vsse32: bajando por la columna 0\nde Bc, uno cada 32 B",
    # reference vs ours
    "Reference design (Adrián's gemm_blis.c)": "Diseño de referencia (gemm_blis.c de Adrián)",
    "Our design, current (xDSL vector packer)": "Nuestro diseño, actual (empaquetador vectorial xDSL)",
    "The fix, col-g1-m4": "La solución, col-g1-m4",
    "1 scalar load per number: 32 loads.\nEach load jumps 1536 B to the next column;\nafter 8 it jumps back. 4 B used per touch.":
        "1 carga escalar por número: 32 cargas.\nCada carga salta 1536 B a la siguiente columna;\ntras 8 vuelve atrás. Usa 4 B por acceso.",
    "1 strided vector load per row: 4 loads,\neach sends 8 requests to 8 lines at once.\nSame addresses, same order, in bursts.":
        "1 carga vectorial con salto por fila: 4 cargas,\ncada una envía 8 peticiones a 8 líneas a la vez.\nMismas direcciones, mismo orden, en ráfagas.",
    "1 vector load (LMUL 4) for all 32 numbers:\n128 B in one piece, two whole cache lines,\nthen the next column. No jumping.":
        "1 carga vectorial (LMUL 4) para los 32 números:\n128 B de una vez, dos líneas de caché enteras,\ny luego la siguiente columna. Sin saltos.",
    "loads 1-4: each sends\n8 requests at once": "cargas 1-4: cada una envía\n8 peticiones a la vez",
    "load 1: all 32 numbers, bytes 0-127 of column 0": "carga 1: los 32 números, bytes 0-127 de la columna 0",
    # code figure notes (the code itself is monospace and is not translated)
    "Before: the current read order (as C intrinsics; the same order as the xDSL packer)":
        "Antes: el orden de lectura actual (en intrínsecos de C; el mismo orden que el empaquetador xDSL)",
    "After: the fix, col-g1-m4": "Después: la solución, col-g1-m4",
    "1  loop: k outside, columns inside": "1  bucle: k fuera, columnas dentro",
    "2  load: vlse32, m1 = 1 register = 8 numbers; stride ldb*4 bytes = 384 x 4 B = 1536 B":
        "2  carga: vlse32, m1 = 1 registro = 8 números; salto ldb*4 bytes = 384 x 4 B = 1536 B",
    "   (one number from each of 8 columns)": "   (un número de cada una de 8 columnas)",
    "3  store: vse32, 8 numbers side by side in a Bc row": "3  escritura: vse32, 8 números seguidos en una fila de Bc",
    "1  loop: column outside, k (32 at a time) inside": "1  bucle: columna fuera, k (de 32 en 32) dentro",
    "2  load: vle32, m4 = 4 registers joined = 32 numbers, side by side":
        "2  carga: vle32, m4 = 4 registros unidos = 32 números seguidos",
    "   (32 neighbours in one column, 128 B)": "   (32 vecinos en una columna, 128 B)",
    "3  store: vsse32, stride RR*4 bytes (NR numbers) down a Bc column":
        "3  escritura: vsse32, salto RR*4 bytes (NR números) bajando por una columna de Bc",
    # chart titles and legends
    "(a) Shape under test: B3 (64×384×384) and B3_K256 (64×384×256)":
        "(a) Forma estudiada: B3 (64×384×384) y B3_K256 (64×384×256)",
    "(b) Controls that must not get worse: B2 (384×384×64) and Reuse24 (384³)":
        "(b) Controles que no deben empeorar: B2 (384×384×64) y Reuse24 (384³)",
    "(a) The layer under test: B3 and B3_K256": "(a) La capa estudiada: B3 y B3_K256",
    "(b) Other layers: nothing gets worse": "(b) Otras capas: nada empeora",
    "row (current xDSL packer)": "row (empaquetador xDSL actual)",
    "col-g1-m4 (one column at a time)": "col-g1-m4 (una columna cada vez)",
    "our current packer (xDSL)": "nuestro empaquetador actual (xDSL)",
    "the fix (one column at a time)": "la solución (una columna cada vez)",
    "as it is (row-by-row B packer)": "tal cual (empaquetador de B por filas)",
    "with the fix (one column at a time)": "con la solución (una columna cada vez)",
    "Our design (xDSL)\n16×15, series 1": "Nuestro diseño (xDSL)\n16×15, serie 1",
    "Reference design\n16×14 (today)": "Diseño de referencia\n16×14 (hoy)",
    "Reference design\n16×16 (today)": "Diseño de referencia\n16×16 (hoy)",
}

# Phrases inside labels that also carry numbers (longest first).
PHRASES = [
    ("GEMM, current row packer", "GEMM, empaquetador actual por filas"),
    ("B packer alone", "empaquetador de B solo"),
    ("reference design, as it is", "diseño de referencia, tal cual"),
    ("reference design, with the fix", "diseño de referencia, con la solución"),
    ("our current packer (xDSL)", "nuestro empaquetador actual (xDSL)"),
    ("the fix (one column at a time)", "la solución (una columna cada vez)"),
    ("% of reps slow", "% de repeticiones lentas"),
    ("GOPs by mean time", "GOPs por tiempo medio"),
    ("(default tile)", "(tesela por defecto)"),
    ("(default)", "(por defecto)"),
    ("% slow", "% lentas"),
    (" reps,", " repeticiones,"),
    ("mean ", "media "),
    ("col ", "col. "),
]


def tr(s: str) -> str:
    if s in EXACT:
        out = EXACT[s]
    else:
        out = s
        for a, b in PHRASES:
            out = out.replace(a, b)
    return re.sub(r"(\d)\.(\d)", r"\1,\2", out)
