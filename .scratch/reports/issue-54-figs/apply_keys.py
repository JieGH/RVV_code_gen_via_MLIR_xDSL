"""Mark the key points of the issue #54 brief with \\key{...}, in the English brief and in the Spanish body, as pairs.

    python3 .scratch/reports/issue-54-figs/apply_keys.py

Each pair is (English phrase, Spanish phrase). A phrase matches across line breaks (any run of spaces/newlines). It is
wrapped once per occurrence and never twice (already-marked text is skipped), so the script can be run again safely.
It fails if a phrase is found in one language and not in the other, so the two versions keep the same highlights.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EN = HERE.parent / "issue-54-brief.tex"
ES = HERE / "body_es.tex"

PAIRS = [
    (r"The trigger is reading 4 bytes from each column of B per touch.",
     r"El disparador es leer 4 bytes de cada columna de B en cada acceso."),
    (r"gives no slow rep in any GEMM cell, on any tile, K, board or night.",
     r"no da ninguna repetición lenta en ninguna celda GEMM, con ninguna tesela, K, placa o noche."),
    (r"B3 goes from 4.53--4.63 to 7.01--7.05 GOPs by mean time",
     r"B3 pasa de 4,53--4,63 a 7,01--7,05 GOPs por tiempo medio"),
    (r"each instruction touches 8 cache lines and takes 4 B from each.",
     r"cada instrucción toca 8 líneas de caché y toma 4 B de cada una."),
    (r"both Bc are the same byte for byte",
     r"los dos Bc son idénticos byte a byte"),
    (r"None of them shows a loss.",
     r"En ninguno se ve una pérdida."),
    (r"with the fix it had no slow rep, 7.14 GOPs (7.11--7.18), +59\%.",
     r"con la solución no tuvo ninguna repetición lenta, 7,14 GOPs (7,11--7,18), +59\%."),
    (r"The fix removes the slow state in both designs.",
     r"La solución elimina el estado lento en los dos diseños."),
    (r"+52\% on B3 and +69\% on B3\_K256",
     r"+52\% en B3 y +69\% en B3\_K256"),
    (r"The fix is the only one of the three that is never slow and never clearly slower",
     r"La solución es la única de las tres que nunca es lenta y nunca es claramente más lenta"),
    (r"Port the fix if that share is significant;",
     r"Portar la solución si esa proporción es significativa;"),
    (r"One column at a time is the only order that is safe on every cell we ran.",
     r"Una columna cada vez es el único orden seguro en todas las celdas que probamos."),
    (r"About 9 reps out of 10 are slow.",
     r"Unas 9 de cada 10 repeticiones son lentas."),
    (r"it stays at about 7.1--7.2 GOPs with no slow rep.",
     r"se mantiene en unos 7,1--7,2 GOPs sin ninguna repetición lenta."),
    (r"So one load touches 8 different cache lines and uses only 4 bytes from each.",
     r"Así que una carga toca 8 líneas de caché distintas y solo usa 4 bytes de cada una."),
    (r"Reading B one column at a time removes it:",
     r"Leer B una columna cada vez lo elimina:"),
    (r"So the finished Bc is the same byte for byte,",
     r"Así que el Bc terminado es idéntico byte a byte,"),
    (r"about 4.6 $\to$ 7.0 GOPs (+50\%).",
     r"de unos 4,6 a 7,0 GOPs (+50\%)."),
    (r"about 4.8 $\to$ 7.9 GOPs (+60\%).",
     r"de unos 4,8 a 7,9 GOPs (+60\%)."),
    (r"It depends on how much of a real model's time is in small-M, large-K layers like B3.",
     r"Depende de cuánto tiempo de un modelo real se pasa en capas con M pequeña y K grande, como B3."),
]


def wrap(text: str, phrase: str) -> tuple[str, int]:
    pat = r"\s+".join(re.escape(w) for w in phrase.split())
    n = 0

    def sub(m: re.Match) -> str:
        nonlocal n
        start = m.start()
        if text[max(0, start - 5):start] == "\\key{":
            return m.group(0)
        n += 1
        return "\\key{" + m.group(0) + "}"

    return re.sub(pat, sub, text), n


def main() -> int:
    en, es = EN.read_text(), ES.read_text()
    bad = 0
    for e, s in PAIRS:
        en2, ne = wrap(en, e)
        es2, ns = wrap(es, s)
        found_e = ne or ("\\key{" + e.split()[0]) in en
        found_s = ns or ("\\key{" + s.split()[0]) in es
        if not (found_e and found_s):
            print(f"  [MISSING] en={bool(found_e)} es={bool(found_s)}: {e[:60]}")
            bad += 1
        en, es = en2, es2
    EN.write_text(en)
    ES.write_text(es)
    print(f"key points: {len(PAIRS)} pairs, {bad} missing; \\key in English {en.count(chr(92) + 'key{')}, in Spanish {es.count(chr(92) + 'key{')}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
