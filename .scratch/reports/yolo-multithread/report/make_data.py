#!/usr/bin/env python3
"""Combined YOLO report data: GOPs by mean time only (the user's choice, 2026-11-18).

Usage: python3 make_data.py   (run after the issue-91 make_data.py)
Reads the issue folders' data and writes mean-time-only copies to data/:
  ../../issue-91/deck/data/{paper_sv,yolo_sv_rows}.tex, ../../issue-91/deck/p85/data/tab_*.tex
                                                   "mean / median" cells -> mean (the ~ mark stays)
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent.parent
OUT = HERE / "data"
PAIR = re.compile(r"(\d+\.\d+) / \d+\.\d+(\$\^\{\\sim\}\$)?")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    srcs = [R / "issue-91/deck/data/paper_sv.tex", R / "issue-91/deck/data/yolo_sv_rows.tex",
            *sorted((R / "issue-91/deck/p85/data").glob("tab_*.tex"))]
    for src in srcs:
        s = PAIR.sub(lambda m: m.group(1) + (m.group(2) or ""), src.read_text())
        assert " / " not in s, src
        (OUT / src.name).write_text(s)
    print(f"wrote {len(srcs)} files to {OUT}")


if __name__ == "__main__":
    main()
