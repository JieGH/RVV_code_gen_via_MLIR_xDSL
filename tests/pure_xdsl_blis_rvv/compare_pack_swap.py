#!/usr/bin/env python3
"""Vector-pack GEMM against scalar-pack GEMM, end to end (issue #26).

Reads the merged raw logs of two kinds of run of the same build and protocol:

  vector-live : all_raw_*_noinstr_refproto[_no_nc_override].txt
  scalar-live : all_raw_*_noinstr_refproto_scalarpack[_no_nc_override].txt   (--scalar-pack)

and prints, per case, e2e GOPs = 2*M*N*K / `RVV_Time_ms` (mean time per whole GEMM call,
packing included, the reference design's own formula) for each side.

Give several runs per side to see the noise: `run_pack_swap.sh` runs vector, scalar, scalar,
vector. `spread` is (best run - worst run) / mean of the runs on one side; a difference
between the sides smaller than the larger spread is called "within noise".

Usage:
  python3 compare_pack_swap.py --vector RUN [RUN ...] --scalar RUN [RUN ...] [--shipped-nc]

RUN is a run directory name under logs/ or a path. --shipped-nc reads the logs of runs made
without --no-nc-overrides.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE / "logs"

PROTOCOL = "_noinstr_refproto"


@dataclass
class Case:
    layer: str
    variant: str
    shape: str
    gops: list[float]  # one entry per run, in the order the runs were given


def raw_glob(scalar: bool, no_nc_overrides: bool) -> str:
    """The merged log a run leaves, named the way the driver names it."""
    suffix = PROTOCOL + ("_scalarpack" if scalar else "") + ("_no_nc_override" if no_nc_overrides else "")
    return f"all_raw_*{suffix}.txt"


def load_side(
    runs: list[Path], scalar: bool, no_nc_overrides: bool, failed: list[str]
) -> dict[tuple[str, str], Case]:
    """(layer, variant) -> the e2e GOPs of each run that measured it. A FAIL row is left out
    of the numbers and named in `failed`, so a broken run cannot pass as a slow one."""
    cases: dict[tuple[str, str], Case] = {}
    for run in runs:
        raws = sorted(run.glob(raw_glob(scalar, no_nc_overrides)))
        if not raws:
            raise SystemExit(f"{run}: no {raw_glob(scalar, no_nc_overrides)} to read")
        for raw in raws:
            with raw.open() as fh:
                for row in csv.DictReader(fh):
                    if row["Status"] == "FAIL":
                        failed.append(f"{run.name}: {row['Layer']}/{row['Variant']}")
                        continue
                    m, n, k = int(row["M"]), int(row["N"]), int(row["K"])
                    ms = float(row["RVV_Time_ms"])
                    if not ms > 0:
                        continue
                    key = (row["Layer"], row["Variant"])
                    case = cases.setdefault(key, Case(row["Layer"], row["Variant"], f"{m}x{n}x{k}", []))
                    case.gops.append(2.0 * m * n * k / (ms * 1e6))
    return cases


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def spread_pct(xs: list[float]) -> float:
    return 100.0 * (max(xs) - min(xs)) / mean(xs) if len(xs) > 1 else 0.0


def verdict(diff_pct: float, noise_pct: float, single_run: bool) -> str:
    if single_run:
        return "one run per side: no noise estimate"
    if abs(diff_pct) <= noise_pct:
        return "within noise"
    return "scalar pack faster" if diff_pct > 0 else "scalar pack slower"


def render(vector: dict[tuple[str, str], Case], scalar: dict[tuple[str, str], Case]) -> str:
    headers = ["Case", "Shape", "Variant", "vector GOPs", "scalar GOPs", "scalar vs vector",
               "spread vec/scl", "verdict"]
    rows: list[list[str]] = []
    for key in sorted(vector.keys() & scalar.keys()):
        v, s = vector[key], scalar[key]
        v_mean, s_mean = mean(v.gops), mean(s.gops)
        diff = 100.0 * (s_mean - v_mean) / v_mean
        noise = max(spread_pct(v.gops), spread_pct(s.gops))
        single = len(v.gops) < 2 and len(s.gops) < 2
        rows.append([
            v.layer, v.shape, v.variant,
            " / ".join(f"{g:.2f}" for g in v.gops) + f" (mean {v_mean:.2f})",
            " / ".join(f"{g:.2f}" for g in s.gops) + f" (mean {s_mean:.2f})",
            f"{diff:+.1f}%",
            f"{spread_pct(v.gops):.1f}% / {spread_pct(s.gops):.1f}%",
            verdict(diff, noise, single),
        ])
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    only = sorted(vector.keys() ^ scalar.keys())
    if only:
        lines.append("")
        lines.append("Only one side measured: " + ", ".join(f"{a}/{b}" for a, b in only))
    return "\n".join(lines)


def resolve(run: str) -> Path:
    p = Path(run)
    return p if p.is_dir() else LOGS / run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vector", nargs="+", required=True, metavar="RUN", help="vector-pack-live run dirs")
    ap.add_argument("--scalar", nargs="+", required=True, metavar="RUN", help="scalar-pack-live run dirs")
    ap.add_argument("--shipped-nc", action="store_true",
                    help="read runs made without --no-nc-overrides (default: runs made with it)")
    args = ap.parse_args(argv)

    no_nc = not args.shipped_nc
    failed: list[str] = []
    vector = load_side([resolve(r) for r in args.vector], False, no_nc, failed)
    scalar = load_side([resolve(r) for r in args.scalar], True, no_nc, failed)
    print(f"e2e GOPs by mean time per GEMM call, packing included; vector runs: {len(args.vector)}, "
          f"scalar runs: {len(args.scalar)}")
    print()
    print(render(vector, scalar))
    if failed:
        print()
        print("WARNING: rows that FAILED the correctness check and are left out above: " + ", ".join(failed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
