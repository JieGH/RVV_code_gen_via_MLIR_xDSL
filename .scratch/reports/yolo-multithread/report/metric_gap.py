#!/usr/bin/env python3
"""GOPs by mean time against GOPs by median rep: how far apart are they, per case?

Usage: python3 metric_gap.py   -> prints a table and writes data/metric_gap_rows.tex
One case = one shape x one tile x one launch, from the committed per-rep times (reps_*.txt[.gz]).
Ratio = GOPs by median rep / GOPs by mean time (>= 1 means a few slow reps pull the mean time down).
Runs: the uninstrumented, reference-protocol runs behind the combined YOLO report.
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = HERE.parents[3] / "tests" / "pure_xdsl_blis_rvv"
LOGS = TESTS / "logs"
sys.path.insert(0, str(TESTS))
from reps_report import case_stats, read_reps, reps_files  # noqa: E402


def issue94(setup: str, threads: int) -> list[str]:
    """Run folders of one #94 setup (V or S) at one thread count, from the series steps.tsv."""
    out = []
    for ln in (LOGS / "issue94_2026-11-18_115554/steps.tsv").read_text().splitlines()[1:]:
        _step, s, t, series, _exit = ln.split("\t")
        if s == setup and int(t) == threads:
            out += [r.split("\t")[4] for r in (LOGS / series / "runs.tsv").read_text().splitlines()[1:]]
    return sorted(out)


def runs(*stamps: str) -> list[str]:
    return [f"run_2026-11-{s}" for s in stamps]


GROUPS = [  # label, threads, run folders
    ("\\#94 V: YOLO, 24x8, vector", "1T", issue94("V", 1)),
    ("\\#94 S: YOLO, 24x8, scalar", "1T", issue94("S", 1)),
    ("\\#61: YOLO, 19 tiles, vector", "1T", runs("15_220501", "15_231549", "16_002638")),
    ("\\#83 T1: YOLO, 19 tiles, scalar", "1T", runs("17_171901", "17_182951", "17_194103")),
    ("\\#71: paper, 19 tiles, vector", "1T", runs("17_020326", "17_043612", "17_053929")),
    ("\\#83 T3: paper, 19 tiles, scalar", "1T", runs("17_133114", "17_160334", "17_170643")),
    ("\\#94 V: YOLO, 24x8, vector", "8T", issue94("V", 8)),
    ("\\#94 S: YOLO, 24x8, scalar", "8T", issue94("S", 8)),
    ("\\#63: YOLO, 19 tiles, vector", "8T", runs("16_222551", "16_234245", "17_005337")),
    ("\\#83 T2: YOLO, 19 tiles, scalar", "8T", runs("17_133645", "17_145239", "17_160305")),
    ("\\#71: paper, 19 tiles, vector", "8T", runs("17_024808", "17_033254", "17_062411")),
    ("\\#83 T3: paper, 19 tiles, scalar", "8T", runs("17_141548", "17_150024", "17_175117")),
]


def ratios(folders: list[str]) -> list[float]:
    out = []
    for r in folders:
        for f in reps_files(LOGS / r):
            for c in read_reps(f).values():
                st = case_stats(c.ops, c.seconds)
                out.append(st.gops_median / st.gops_mean_time)
    return out


def main() -> None:
    lines, tex = [], []
    for label, th, folders in GROUPS:
        assert folders, label
        r = sorted(ratios(folders))
        n = len(r)
        within = [sum(x <= lim for x in r) for lim in (1.01, 1.03)]
        row = [th, label, str(len(folders)), str(n), f"{r[0]:.3f}", f"{statistics.median(r):.3f}",
               f"{r[int(0.95 * (n - 1))]:.3f}", f"{r[-1]:.3f}", str(within[0]), str(within[1])]
        tex.append(" & ".join(row) + " \\\\")
        lines.append("  ".join(row).replace("\\#", "#"))
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data/metric_gap_rows.tex").write_text("\n".join(tex) + "\n")
    print("threads  group  launches  cases  min  median  95th  max  cases<=1.01  cases<=1.03")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
