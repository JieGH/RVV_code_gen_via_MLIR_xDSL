#!/usr/bin/env python3
"""Join a fresh reference-design baseline against our own run logs.

The two harnesses do NOT report the same metric:

  reference : GFLOPS = 2mnk / wall(gemm_blis_B3A2C0)        -> pack INCLUDED
              (ext/RVV_ukernels_benchmark/gemm_blis_family/src/test_gemm.c:399-400)
  ours      : KerGOPs = 2mnk / (total_kernel_time + total_pack_time) -> pack INCLUDED (Fixed via Issue #13)
              (tests/pure_xdsl_blis_rvv/templates/driver_benchmark.cpp:695)

So this script reports BOTH: our raw KerGOPs (what the dashboards plot) and our
end-to-end 2mnk/(vector_pack + compute). Since Issue #13, these two should be
nearly identical. Comparing either to the reference column is now apples-to-apples.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE.parent / "logs"


def load_plan(result_dir: Path) -> list[dict]:
    plan = result_dir / "run_plan.tsv"
    if not plan.is_file():
        plan = HERE / "generated" / "run_plan.tsv"
    rows = []
    for line in plan.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("\t")
        rows.append({
            "case": f[0], "numid": int(f[1]),
            "m": int(f[2]), "n": int(f[3]), "k": int(f[4]),
            "pinned": f"{f[5]}x{f[6]}", "built": f"{f[7]}x{f[8]}",
            "kind": f[9], "mc": f[10], "nc": f[11], "kc": f[12], "tag": f[13],
        })
    return rows


def load_reference(result_dir: Path) -> dict[tuple[str, int], tuple[float, str]]:
    """(tag, numid) -> (best GFLOPS, winning kernel variant).

    Files are result_<tag>_<variant>.csv (or legacy result_<tag>.csv). When several
    kernel variants were measured we keep the BEST per case: the reference design
    should be judged at its strongest, the same way the archived tables reported
    'BEST PERFORMANCE' over a sweep.
    """
    out: dict[tuple[str, int], tuple[float, str]] = {}
    for csv_path in sorted(result_dir.glob("result_*.csv")):
        stem = csv_path.stem[len("result_"):]
        if "_" in stem:
            tag, variant = stem.rsplit("_", 1)
        else:
            tag, variant = stem, "?"
        for line in csv_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(";")
            if len(parts) < 5:
                continue
            try:
                numid = int(parts[0])
                gflops = float(parts[4])
            except ValueError:
                continue  # the 'CCC_NN;...' echo rows
            prev = out.get((tag, numid))
            if prev is None or gflops > prev[0]:
                out[(tag, numid)] = (gflops, variant)
    return out


def discover_runs(explicit: list[str] | None) -> list[Path]:
    if explicit:
        return [LOGS / r for r in explicit]
    runs = sorted((d for d in LOGS.glob("run_*") if
                   (d / "all_raw_fp32_AXPY_FP32_Pack.txt").is_file()),
                  key=lambda d: d.name)
    return runs[-5:]


def load_ours(runs: list[Path]) -> dict[str, dict]:
    acc: dict[str, list[dict]] = {}
    for run in runs:
        raw = run / "all_raw_fp32_AXPY_FP32_Pack.txt"
        if not raw.is_file():
            continue
        with raw.open() as fh:
            for row in csv.DictReader(fh):
                acc.setdefault(row["Layer"], []).append(row)
    out: dict[str, dict] = {}
    for case, rows in acc.items():
        m, n, k = (int(rows[0][x]) for x in "MNK")
        vp = statistics.mean(float(r["Vector_Pack_Time_ms"]) for r in rows)
        cp = statistics.mean(float(r["Compute_Time_ms"]) for r in rows)
        out[case] = {
            "ker": statistics.mean(float(r["KerGOPs_Median"]) for r in rows),
            "vpack_ms": vp, "compute_ms": cp,
            "pack_frac": vp / (vp + cp) if (vp + cp) else 0.0,
            "e2e": 2.0 * m * n * k / ((vp + cp) * 1e6) if (vp + cp) else 0.0,
            "nruns": len(rows),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result-dir", type=Path, required=True,
                    help="logs/refbase_<stamp>/ produced by run_ref_baseline.sh")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="our run_* log dirs to average (default: latest 5)")
    args = ap.parse_args()

    if not args.result_dir.is_dir():
        print(f"ERROR: no such result dir: {args.result_dir}", file=sys.stderr)
        return 1

    plan = load_plan(args.result_dir)
    ref = load_reference(args.result_dir)
    runs = discover_runs(args.runs)
    ours = load_ours(runs)

    if not ref:
        print("  No reference results found (no result_*.csv). Did the board run fail?")
        return 1

    print()
    print(f"  Ours      : mean of {len(runs)} run(s): "
          + ", ".join(r.name for r in runs))
    print(f"  Reference : {args.result_dir.name}")
    print()
    print("  'ref' is pack-INCLUSIVE. Compare it against 'e2e', not against 'Ker'.")
    print("  'kern' is the reference kernel variant that won this case.")
    print()

    hdr = (f"  {'case':4} {'shape':16} {'pin':>5} {'built':>5} {'kind':>7} "
           f"{'ref':>6} {'kern':>6} | {'Ker':>6} {'Ker/ref':>8} | "
           f"{'pack%':>6} {'e2e':>6} {'e2e/ref':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    order = {r["case"]: i for i, r in enumerate(plan)}
    for row in sorted(plan, key=lambda r: (r["case"][0] != "S", order[r["case"]])):
        hit = ref.get((row["tag"], row["numid"]))
        o = ours.get(row["case"])
        if hit is None:
            continue
        g, variant = hit
        shape = f"{row['m']}x{row['n']}x{row['k']}"
        if o is None:
            print(f"  {row['case']:4} {shape:16} {row['pinned']:>5} {row['built']:>5} "
                  f"{row['kind']:>7} {g:6.2f} {variant:>6} | {'--':>6} {'--':>8} | "
                  f"{'--':>6} {'--':>6} {'--':>8}   (no local run data)")
            continue
        print(f"  {row['case']:4} {shape:16} {row['pinned']:>5} {row['built']:>5} "
              f"{row['kind']:>7} {g:6.2f} {variant:>6} | "
              f"{o['ker']:6.2f} {100*(o['ker']/g-1):+7.1f}% | "
              f"{100*o['pack_frac']:5.1f}% {o['e2e']:6.2f} {100*(o['e2e']/g-1):+7.1f}%")

    bracketed = sorted({r["case"] for r in plan if r["kind"] == "bracket"})
    if bracketed:
        print()
        print("  NOTE: the reference ships no kernel at the pinned MR/NR for "
              + ", ".join(bracketed) + ".")
        print("        Those rows are the nearest even-NR neighbours, so they bracket")
        print("        the true reference number rather than matching it exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
