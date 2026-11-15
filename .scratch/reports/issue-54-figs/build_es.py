"""Build the Spanish brief (issue-54-brief-es.tex) from the English one and the hand-translated body.

    python3 .scratch/reports/issue-54-figs/build_es.py

The prose is translated by hand in body_es.tex. The numeric tables (%%TABn%%, n = position of the tabular in the English
file) and the code listings (%%VERBn%%) are copied from issue-54-brief.tex, so every number and every code line stays
exactly as in the English version; only the table headers are swapped through HEADERS.
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
EN = HERE.parent / "issue-54-brief.tex"
OUT = HERE.parent / "issue-54-brief-es.tex"

HEADERS = [
    ("variant & ", "variante & "),
    ("layer & tile & current (vector) & scalar & the fix & fix vs current & fix vs scalar",
     "capa & tesela & actual (vectorial) & escalar & la solución & solución vs actual & solución vs escalar"),
    ("(default)", "(por defecto)"),
]

TITLE = r"""\title{\vspace{-1.2cm}\textbf{Qué provoca, en las lecturas del empaquetador de B, el estado lento de B3,\\ y un empaquetador que
mantiene B3 estable}\\[2pt]
\large Informe del issue \iss{54} (especificación \iss{55}), continuación de \iss{44}, \iss{48}, \iss{49}, \iss{50} y \iss{53}}
\author{RVV\_code\_gen\_via\_MLIR\_xDSL, \texttt{pure-xdsl}; rama \texttt{issue55-pack-b-variants}\\
SpacemiT X60 (RVV 1.0, VLEN 256), \texttt{jbpi2}; serie 1 el 2026-10-22, serie 2 el 2026-10-23}
\date{23 de octubre de 2026; figuras añadidas el 29 de octubre de 2026. Versión en español}
"""

SPANISH = r"""% Spanish version: no babel/polyglossia in the offline cache, so names are set by hand and hyphenation is kept rare.
\renewcommand{\figurename}{Figura}
\hyphenpenalty=2000
\emergencystretch=2em
"""


def main() -> None:
    en = EN.read_text()
    body = (HERE / "body_es.tex").read_text()
    pre = en[:en.index("\\begin{document}")]
    pre = re.sub(r"\\title\{.*?\n\\date\{[^\n]*\}\n", lambda _: TITLE, pre, count=1, flags=re.S)
    assert "Versión en español" in pre, "title block not replaced"
    pre = pre.replace("\\graphicspath{{issue-54-figs/}}", "\\graphicspath{{issue-54-figs/es/}}")
    assert "issue-54-figs/es/" in pre, "graphicspath not switched to the Spanish figures"
    pre += SPANISH

    tabs = re.findall(r"\\begin\{tabular\}.*?\\end\{tabular\}", en, flags=re.S)
    verbs = re.findall(r"\{\\(?:footnotesize|scriptsize)\\begin\{verbatim\}.*?\\end\{verbatim\}\}", en, flags=re.S)
    assert len(tabs) == 9 and len(verbs) == 11, (len(tabs), len(verbs))

    def tab(m: re.Match) -> str:
        t = tabs[int(m.group(1)) - 1]
        for a, b in HEADERS:
            t = t.replace(a, b)
        return t

    body = re.sub(r"%%TAB(\d+)%%", tab, body)
    body = re.sub(r"%%VERB(\d+)%%", lambda m: verbs[int(m.group(1)) - 1], body)
    assert "%%" not in body, "a placeholder was left"
    # T1 + XeTeX puts U+00BF/U+00A1 in the wrong font slot (prints a pound sign): use the LaTeX names instead.
    body = body.replace("¿", "\\textquestiondown{}").replace("¡", "\\textexclamdown{}")
    OUT.write_text(pre + body)
    print(f"wrote {OUT.name}: {len(tabs)} tables seen, 5 copied; {len(verbs)} code listings copied")


if __name__ == "__main__":
    main()
