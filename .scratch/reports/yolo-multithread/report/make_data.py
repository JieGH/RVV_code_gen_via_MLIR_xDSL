#!/usr/bin/env python3
"""Combined YOLO report: put GOPs by mean time and by median rep in one cell ("mean / median").

Usage: python3 make_data.py   (run after the issue-94 and issue-95 make_data.py)
Reads ../../issue-94/report/data/{pass_rows,by_n_mean_rows,by_n_median_rows}.tex and
../../issue-95/report/data/probe_rows.tex; writes the merged rows to data/.
A value with no median (Héctor's driver keeps no per-call time) prints alone (by mean time).
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
E = HERE / "../../issue-94/report/data"
F = HERE / "../../issue-95/report/data"
OUT = HERE / "data"


def cells(line: str) -> list[str]:
    return [c.strip() for c in line.replace("\\\\", "").split("&")]


def rows(path: Path) -> list[list[str]]:
    return [cells(ln) for ln in path.read_text().splitlines() if "&" in ln]


def pair(a: str, b: str | None) -> str:
    return f"{a} / {b}" if b else a


def row(cs: list[str]) -> str:
    return " & ".join(cs) + " \\\\"


def main() -> None:
    OUT.mkdir(exist_ok=True)

    # one pass: rows come as "<setup>, by mean time" then (ours only) "<setup>, by median rep"
    mean, med = {}, {}
    order = []
    for cs in rows(E / "pass_rows.tex"):
        name, kind = cs[0].rsplit(", by ", 1)
        (mean if kind == "mean time" else med)[name] = cs
        if name not in order:
            order.append(name)
    out = []
    for name in order:
        m, d = mean[name], med.get(name)
        # ms, GOPs, range (mean only), ms, GOPs, range, speedup
        out.append(row([name] + [m[i] if i in (3, 6) else pair(m[i], d[i] if d else None) for i in range(1, 8)]))
    (OUT / "pass_rows.tex").write_text("\n".join(out) + "\n")

    # by n: V and S have both metrics, H only by mean time
    mrows, drows = rows(E / "by_n_mean_rows.tex"), rows(E / "by_n_median_rows.tex")
    assert [r[0] for r in mrows] == [r[0] for r in drows]
    out = [row(m[:2] + [pair(m[i], d[i]) for i in range(2, 8)] + [pair(m[i], None) for i in range(8, 11)])
           for m, d in zip(mrows, drows)]
    (OUT / "by_n_rows.tex").write_text("\n".join(out) + "\n")

    # #95 probe runs: ours by mean time / by median rep in one cell
    out = [row(p[:5] + [pair(p[5], p[6]), p[7]]) for p in rows(F / "probe_rows.tex")]
    (OUT / "probe_rows.tex").write_text("\n".join(out) + "\n")
    print(f"wrote {OUT}/pass_rows.tex, by_n_rows.tex, probe_rows.tex")


if __name__ == "__main__":
    main()
