#!/usr/bin/env python3
"""Report on a scalar-pack sweep (run_scalar_pack_sweep.sh, issue #26), finished or not.

Reads the runs a sweep has completed so far (`logs/sweep_scalar_<stamp>/runs.txt`) and the
vector-pack runs of the overnight sweep of 19-20 Sep (any run dir with a stamp inside
--vs-from .. --vs-to that has the vector-pack log), and writes a Markdown report:

  - which runs are in, which are still to come, and what differs between the two sets
    (time of day, boards);
  - per layer: e2e GOPs by mean time (2*M*N*K / mean time per GEMM call, packing included)
    for both packers, the difference, a verdict against the spread between runs, and the
    published figure;
  - per board, so a board imbalance in a partial sweep shows;
  - B3 run by run, because its reps have two speeds and its median is not stable.

Usage: python3 report_scalar_sweep.py [--sweep DIR] [--out FILE]
Without --sweep it takes the newest logs/sweep_scalar_*. Without --out it writes
<sweep dir>/report_<HHMM>.md and prints it.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import rawlog

HERE = Path(__file__).resolve().parent
LOGS = HERE.parent / "logs"

LAYER_ORDER = ["S1", "S2", "S3", "S4", "S5", "B1", "B2", "B3", "B4", "B5"]
# The published table (issue 11), GOPs per layer.
PUBLISHED = {"S1": 8.5, "S2": 8.7, "S3": 8.9, "S4": 9.0, "S5": 9.0,
             "B1": 8.0, "B2": 12.5, "B3": 8.8, "B4": 8.5, "B5": 7.5}

VECTOR_GLOB = rawlog.raw_glob(no_nc_override=True)
SCALAR_GLOB = rawlog.raw_glob(scalar_pack=True, no_nc_override=True)


@dataclass
class Point:
    run: str
    board: str
    gops: float
    median: float
    p5: float
    p95: float
    peak: float
    reps: int


@dataclass
class Layer:
    shape: str = ""
    cfg: str = ""
    vector: list[Point] = field(default_factory=lambda: [])
    scalar: list[Point] = field(default_factory=lambda: [])


def board_of(run_dir: Path) -> str:
    prov = run_dir / "provenance.txt"
    if prov.is_file():
        m = re.search(r"board\s*:\s*(?:\w+@)?(\w+)", prov.read_text())
        if m:
            return m.group(1)
    return "?"


def read_run(run_dir: Path, glob: str) -> dict[str, tuple[str, str, Point]]:
    """layer -> (shape, config, point) for one run dir."""
    out: dict[str, tuple[str, str, Point]] = {}
    for raw in sorted(run_dir.glob(glob)):
        for r in rawlog.read_rows(raw):
            if r["Status"] == "FAIL" or not float(r["RVV_Time_ms"]) > 0:
                continue
            m, n, k = int(r["M"]), int(r["N"]), int(r["K"])
            cfg = re.match(r"MR(\d+)_NR(\d+)", r["Variant"])
            out[r["Layer"]] = (
                f"{m}x{n}x{k}",
                f"{cfg.group(1)}x{cfg.group(2)}" if cfg else r["Variant"],
                Point(run_dir.name, board_of(run_dir), 2.0 * m * n * k / (float(r["RVV_Time_ms"]) * 1e6),
                      float(r["KerGOPs_Median"]), float(r["KerGOPs_P5"]), float(r["KerGOPs_P95"]),
                      float(r["KerGOPs_Max"]), int(r["N_Reps"])),
            )
    return out


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def spread(xs: list[float]) -> float:
    return 100.0 * (max(xs) - min(xs)) / mean(xs) if len(xs) > 1 else 0.0


def rng(xs: list[float]) -> str:
    return f"{min(xs):.2f}-{max(xs):.2f}" if len(xs) > 1 else "-"


def clock(run: str) -> str:
    m = re.search(r"_(\d\d)(\d\d)\d\d$", run)
    return f"{m.group(1)}:{m.group(2)}" if m else run


def verdict(d: float, noise: float, n_v: int, n_s: int) -> str:
    if n_v < 2 or n_s < 2:
        return "too few runs for a noise estimate"
    return "within noise" if abs(d) <= noise else ("scalar faster" if d > 0 else "scalar slower")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", type=Path, help="sweep dir (default: the newest logs/sweep_scalar_*)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--vs-from", default="2026-09-19_230000")
    ap.add_argument("--vs-to", default="2026-09-20_070000")
    args = ap.parse_args()

    sweeps = sorted(LOGS.glob("sweep_scalar_*"))
    sweep: Path | None = args.sweep or (sweeps[-1] if sweeps else None)
    if sweep is None or not (sweep / "runs.txt").is_file():
        print("no sweep dir with a runs.txt found", file=sys.stderr)
        return 1
    plan = (sweep / "plan.txt").read_text() if (sweep / "plan.txt").is_file() else ""
    planned = re.search(r"(\d+) runs", plan)
    n_planned = int(planned.group(1)) if planned else 0

    layers: dict[str, Layer] = {}
    scalar_runs: list[tuple[str, str]] = []
    for line in (sweep / "runs.txt").read_text().splitlines():
        board, name = line.split("\t")
        scalar_runs.append((board, name))
        for lid, (shape, cfg, p) in read_run(LOGS / name, SCALAR_GLOB).items():
            lay = layers.setdefault(lid, Layer(shape, cfg))
            lay.scalar.append(p)
    vector_runs: list[tuple[str, str]] = []
    for d in sorted(LOGS.glob("run_*")):
        stamp = d.name[len("run_"):]
        if not (args.vs_from <= stamp <= args.vs_to) or not list(d.glob(VECTOR_GLOB)):
            continue
        vector_runs.append((board_of(d), d.name))
        for lid, (shape, cfg, p) in read_run(d, VECTOR_GLOB).items():
            lay = layers.setdefault(lid, Layer(shape, cfg))
            lay.vector.append(p)

    now = datetime.now()
    out: list[str] = []
    w = out.append
    w("# Scalar-pack sweep, report so far\n")
    w(f"Generated {now:%a %d %b %Y %H:%M}. Sweep dir `{sweep.relative_to(HERE.parents[2]) if sweep.is_relative_to(HERE.parents[2]) else sweep}`.\n")
    done = len(scalar_runs)
    status = "partial, the sweep is still running" if n_planned and done < n_planned else "complete"
    w(f"**Scalar-pack runs in: {done} of {n_planned or '?'} ({status}).** Runs still to come, and a run in progress, are not in the numbers.\n")
    w("| | runs | boards | started |\n|---|---|---|---|")
    for label, runs in (("scalar pack (this sweep)", scalar_runs), ("vector pack (overnight 19-20 Sep)", vector_runs)):
        bs = {b: sum(1 for x, _ in runs if x == b) for b, _ in runs}
        clocks = [clock(n) for _, n in runs]
        w(f"| {label} | {len(runs)} | {', '.join(f'{b} {c}' for b, c in bs.items()) or '-'} | "
          f"{clocks[0] + ' to ' + clocks[-1] if clocks else '-'} |")
    w("\nBoth sets: uninstrumented build, reference protocol, `tmin` 10 s, `--no-nc-overrides`, core 0, the full `config.json`. "
      "e2e GOPs = 2*M*N*K / mean time per whole GEMM call, packing included.\n")
    w("**What is not the same between the two sets:** the vector runs were made overnight (23:00-06:30) and the scalar runs "
      "at the times in the table above (afternoon and evening), so the time of day and the board's temperature differ. "
      "A gap of a few percent can come from that. A partial sweep may also have run on one board only (see the boards column).\n")

    w("## Per layer\n")
    w("| Layer | Shape | Cfg | vector mean | vector range | vector runs | scalar mean | scalar range | scalar runs "
      "| scalar vs vector | verdict | published | scalar vs published | scalar reps per run |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for lid in LAYER_ORDER:
        lay = layers.get(lid)
        if lay is None or not lay.vector:
            continue
        v = [p.gops for p in lay.vector]
        s = [p.gops for p in lay.scalar]
        pub = PUBLISHED.get(lid)
        pub_cell = f"{pub:.1f}" if pub else ""
        head = [lid, lay.shape, lay.cfg, f"{mean(v):.2f}", rng(v), str(len(v))]
        if s:
            d = 100.0 * (mean(s) - mean(v)) / mean(v)
            tail = [f"{mean(s):.2f}", rng(s), str(len(s)), f"{d:+.1f}%",
                    verdict(d, max(spread(v), spread(s)), len(v), len(s)), pub_cell,
                    f"{100.0 * (mean(s) - pub) / pub:+.1f}%" if pub else "", f"{mean([p.reps for p in lay.scalar]):.0f}"]
        else:
            tail = ["-", "-", "0", "", "no scalar run yet", pub_cell, "", "-"]
        w("| " + " | ".join(head + tail) + " |")
    w("\nLayers with 1-2 reps per run (S3, S4, S5) have no spread inside a run; their noise is the spread between runs only.\n")

    w("## Per board\n")
    w("Mean e2e GOPs per board and packer. A board with no scalar run yet reads `-`.\n")
    w("| Layer | vector jbpi | vector jbpi2 | scalar jbpi | scalar jbpi2 |\n|---|---|---|---|---|")
    for lid in LAYER_ORDER:
        lay = layers.get(lid)
        if lay is None:
            continue
        def cell(pts: list[Point], b: str) -> str:
            xs = [p.gops for p in pts if p.board == b]
            return f"{mean(xs):.2f} (n={len(xs)})" if xs else "-"
        w(f"| {lid} | {cell(lay.vector, 'jbpi')} | {cell(lay.vector, 'jbpi2')} | {cell(lay.scalar, 'jbpi')} | {cell(lay.scalar, 'jbpi2')} |")

    b3 = layers.get("B3")
    if b3 and b3.scalar:
        w("\n## B3, run by run\n")
        w("B3 runs at two speeds (about 4.2 and about 9 GOPs); quote it by mean, the median moves between identical runs.\n")
        w("| packer | run | board | mean | median rep | 5th pct | 95th pct | fastest | reps |\n|---|---|---|---|---|---|---|---|---|")
        for label, pts in (("scalar", b3.scalar), ("vector", b3.vector)):
            for p in pts:
                w(f"| {label} | {clock(p.run)} | {p.board} | {p.gops:.2f} | {p.median:.2f} | {p.p5:.2f} | {p.p95:.2f} | {p.peak:.2f} | {p.reps} |")

    text = "\n".join(out) + "\n"
    dest = args.out or (sweep / f"report_{now:%H%M}.md")
    dest.write_text(text)
    print(text)
    print(f"[written to {dest}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
