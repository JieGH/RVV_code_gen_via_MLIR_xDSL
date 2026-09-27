#!/usr/bin/env python3
"""Join a fresh reference-design baseline against our own run logs.

The two harnesses do NOT report the same metric:

  reference : GFLOPS = 2mnk / wall(gemm_blis_B3A2C0)        -> pack INCLUDED
              (ext/RVV_ukernels_benchmark/gemm_blis_family/src/test_gemm.c:399-400)
  ours      : KerGOPs = 2mnk / (total_kernel_time + total_pack_time) -> pack INCLUDED (Fixed via Issue #13)
              (tests/pure_xdsl_blis_rvv/templates/driver_benchmark.cpp:695)

So this script reports BOTH: our raw KerGOPs (what the dashboards plot) and our
end-to-end `e2e`.

`e2e` is 2mnk / `RVV_Time_ms` — the outer wall clock around the whole N_REPS loop
(Issue #14). It used to be 2mnk / (`Vector_Pack_Time_ms` + `Compute_Time_ms`), which
silently dropped the time falling *between* the per-tile `chrono::now()` pairs: no
timer runs there, so it is charged to neither column. On B1 that gap is 2.6% and made
us print -2.2% against the reference where the honest figure is -4.8%. `RVV_Time_ms`
cannot lose it, and it is the same shape as the reference's own single wall-clock
read, so comparing it to the reference column is apples-to-apples.

`fringe` is the extra multiply-adds our full-size fringe kernels do that the reference's
right-sized ones skip (issue #15, `fringe.py`). It is a bias against us that no other
column includes, so read it beside `e2e/ref`, not folded into it.

An uninstrumented run (`make INSTRUMENTED=0`, Issue #14 part 2) compiles the inner
timers out entirely: its `Vector_Pack_Time_ms`/`Compute_Time_ms` are nan, `RVV_Time_ms`
is not, and `pack%` reads `--`. Point `--raw-name` at that run's merged CSV to compare
it. It carries no timer tax at all, so it is the fairest number we can produce.

Both sides must also have been measured under the same conditions (issue #21). The table is
preceded by `Gap` and `Pinning` lines, from the timestamps in the two dir names and the
`cpu pin` line of each `provenance.txt`, and by a `WARNING` when the runs are more than an
hour apart, when a side was not pinned, when the two were pinned to different cores, or when
a run predates provenance (its pinning is reported unknown, not guessed).
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

from fringe import our_fringe

HERE = Path(__file__).resolve().parent
LOGS = HERE.parent / "logs"

# The merged per-run CSV that compile_macro_bench.sh writes. An uninstrumented build
# writes the same schema under a `_noinstr` name; see --raw-name.
DEFAULT_RAW_NAME = "all_raw_fp32_AXPY_FP32_Pack.txt"
NO_NC_OVERRIDE_SUFFIX = "_no_nc_override.txt"


def load_plan(result_dir: Path) -> list[dict]:
    plan = result_dir / "run_plan.tsv"
    if not plan.is_file():
        plan = HERE / "generated" / "run_plan.tsv"
    rows = []
    for line in plan.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("\t")
        rows.append(
            {
                "case": f[0],
                "numid": int(f[1]),
                "m": int(f[2]),
                "n": int(f[3]),
                "k": int(f[4]),
                "pinned": f"{f[5]}x{f[6]}",
                "built": f"{f[7]}x{f[8]}",
                "kind": f[9],
                "mc": f[10],
                "nc": f[11],
                "kc": f[12],
                "tag": f[13],
            }
        )
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
        stem = csv_path.stem[len("result_") :]
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


def discover_runs(
    explicit: list[str] | None, raw_name: str = DEFAULT_RAW_NAME
) -> list[Path]:
    if explicit:
        return [LOGS / r for r in explicit]
    runs = sorted(
        (d for d in LOGS.glob("run_*") if (d / raw_name).is_file()),
        key=lambda d: d.name,
    )
    return runs[-5:]


def load_ours(runs: list[Path], raw_name: str = DEFAULT_RAW_NAME) -> dict[str, dict]:
    acc: dict[str, list[dict]] = {}
    for run in runs:
        raw = run / raw_name
        if not raw.is_file():
            continue
        with raw.open() as fh:
            for row in csv.DictReader(fh):
                acc.setdefault(row["Layer"], []).append(row)
    out: dict[str, dict] = {}
    for case, rows in acc.items():
        m, n, k = (int(rows[0][x]) for x in "MNK")
        # Breakdown columns only: an uninstrumented run leaves them nan, which flows
        # through to pack% as `--` and must not touch e2e.
        vp = statistics.mean(float(r["Vector_Pack_Time_ms"]) for r in rows)
        cp = statistics.mean(float(r["Compute_Time_ms"]) for r in rows)
        # Issue #14: the outer wall clock, which is the only one of our numbers that
        # cannot lose the time between the per-tile timers.
        wall = statistics.mean(float(r["RVV_Time_ms"]) for r in rows)
        out[case] = {
            "ker": statistics.mean(float(r["KerGOPs_Median"]) for r in rows),
            "vpack_ms": vp,
            "compute_ms": cp,
            "pack_frac": vp / (vp + cp) if (vp + cp) else 0.0,
            "e2e": 2.0 * m * n * k / (wall * 1e6) if wall else 0.0,
            "nruns": len(rows),
        }
    return out


# Issue #21: how far apart the two runs may be before the comparison says so. The board is
# shared and its thermal state drifts, so a pair taken a day apart carries an
# unknown-sized variable; one run straight after the other does not.
MAX_GAP_HOURS = 1.0

STAMP = re.compile(r"_(\d{4}-\d{2}-\d{2}_\d{6})$")
PIN_LINE = re.compile(r"^\s*cpu pin\s*:\s*(\S+)", re.MULTILINE)


def stamp_of(d: Path) -> datetime | None:
    found = STAMP.search(d.name)
    return datetime.strptime(found.group(1), "%Y-%m-%d_%H%M%S") if found else None


def pin_of(d: Path) -> str | None:
    """The core `d` was pinned to ("none" if unpinned), or None if it never said."""
    try:
        found = PIN_LINE.search((d / "provenance.txt").read_text())
    except OSError:
        return None
    return found.group(1) if found else None


def pin_words(pin: str | None) -> str:
    return "unknown" if pin is None else "unpinned" if pin == "none" else f"cpu {pin}"


def conditions(result_dir: Path, runs: list[Path]) -> list[str]:
    """What the two sides ran under, one line each, `WARNING` where the pair is not sound.

    Reads only the dir names (their timestamps) and the `cpu pin` line of each
    provenance.txt (issue #21): `run_ref_baseline.sh` and `compile_macro_bench.sh` write it.
    """
    lines: list[str] = []

    ref_time = stamp_of(result_dir)
    gaps = [
        abs((ref_time - t).total_seconds()) / 3600
        for t in map(stamp_of, runs)
        if ref_time is not None and t is not None
    ]
    if gaps:
        worst = max(gaps)
        lines.append(f"Gap      : the run dir stamps are up to {worst:.1f} h apart")
        if worst > MAX_GAP_HOURS:
            lines.append(
                f"WARNING: {worst:.1f} h apart (more than {MAX_GAP_HOURS:g} h): the board's "
                "thermal state and load can differ between them. Run both back to back "
                "(run_head_to_head.sh)."
            )
    else:
        lines.append("Gap      : unknown (no timestamp in a run dir name)")

    ref_pin = pin_of(result_dir)
    our_pins = {pin_of(r) for r in runs}
    lines.append(
        "Pinning  : ours "
        + ", ".join(sorted(pin_words(p) for p in our_pins))
        + f"; reference {pin_words(ref_pin)}"
    )
    if ref_pin is None or None in our_pins:
        unknown = [n for n, unk in (("ours", None in our_pins), ("reference", ref_pin is None)) if unk]
        lines.append(
            f"WARNING: pinning unknown for {' and '.join(unknown)} (no provenance.txt, or a run "
            "from before issue #21): cannot tell that both sides ran on the same core."
        )
    elif ref_pin == "none" or "none" in our_pins:
        lines.append("WARNING: a side was not pinned: the scheduler may have moved its run between cores.")
    elif our_pins != {ref_pin}:
        lines.append("WARNING: the two sides were pinned to different cores.")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="logs/refbase_<stamp>/ produced by run_ref_baseline.sh",
    )
    ap.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="our run_* log dirs to average (default: latest 5)",
    )
    ap.add_argument(
        "--raw-name",
        default=DEFAULT_RAW_NAME,
        help=(
            "merged raw CSV to read inside each run dir "
            f"(default: {DEFAULT_RAW_NAME}; an uninstrumented run writes a _noinstr name)"
        ),
    )
    args = ap.parse_args()

    if not args.result_dir.is_dir():
        print(f"ERROR: no such result dir: {args.result_dir}", file=sys.stderr)
        return 1

    # The driver's --no-nc-overrides run logs under this suffix (issue #16), and the
    # fringe column has to be computed at the same nc it ran at.
    no_nc_overrides = args.raw_name.endswith(NO_NC_OVERRIDE_SUFFIX)

    plan = load_plan(args.result_dir)
    ref = load_reference(args.result_dir)
    runs = discover_runs(args.runs, args.raw_name)
    ours = load_ours(runs, args.raw_name)

    if not ref:
        print("  No reference results found (no result_*.csv). Did the board run fail?")
        return 1

    print()
    print(
        f"  Ours      : mean of {len(runs)} run(s): " + ", ".join(r.name for r in runs)
    )
    print(f"  Reference : {args.result_dir.name}")
    print()
    for line in conditions(args.result_dir, runs):
        print(f"  {line}")
    print()
    print("  'ref' is pack-INCLUSIVE. Compare it against 'e2e', not against 'Ker'.")
    print("  'e2e' is 2mnk / RVV_Time_ms (outer wall clock), timer gap included.")
    print("  'kern' is the reference kernel variant that won this case.")
    print("  'fringe' is the extra multiply-adds our full-size fringe kernels do that the")
    print("  reference's right-sized ones skip (issue #15): a bias against us, not counted")
    print("  in e2e/ref. It leaves out the memset and copy through tmp_C, which cost time too.")
    if no_nc_overrides:
        print("  nc overrides are OFF (a --no-nc-overrides run): every case ran at the heuristic's nc,")
        print("  like the reference. This is the head-to-head configuration.")
    else:
        print("  nc overrides are ON (config.json, e.g. B3 at nc=128): the reference has none, so a")
        print("  case with one is not the same configuration as the reference's. Compare a")
        print("  --no-nc-overrides run (its *_no_nc_override.txt log) for the head-to-head number.")
    print()

    hdr = (
        f"  {'case':4} {'shape':16} {'pin':>5} {'built':>5} {'kind':>7} "
        f"{'ref':>6} {'kern':>6} | {'Ker':>6} {'Ker/ref':>8} | "
        f"{'pack%':>6} {'e2e':>6} {'e2e/ref':>8} | {'fringe':>7}"
    )
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
        pinned_mr, pinned_nr = (int(x) for x in row["pinned"].split("x"))
        fringe = our_fringe(
            row["case"], row["m"], row["n"], row["k"], pinned_mr, pinned_nr,
            use_overrides=not no_nc_overrides,
        )
        fringe_col = f"{100 * fringe:+6.1f}%"
        if o is None:
            print(
                f"  {row['case']:4} {shape:16} {row['pinned']:>5} {row['built']:>5} "
                f"{row['kind']:>7} {g:6.2f} {variant:>6} | {'--':>6} {'--':>8} | "
                f"{'--':>6} {'--':>6} {'--':>8} | {fringe_col}   (no local run data)"
            )
            continue
        pack_col = (
            f"{100 * o['pack_frac']:5.1f}%"
            if math.isfinite(o["pack_frac"])
            else f"{'--':>6}"  # uninstrumented run: no pack/compute breakdown exists
        )
        print(
            f"  {row['case']:4} {shape:16} {row['pinned']:>5} {row['built']:>5} "
            f"{row['kind']:>7} {g:6.2f} {variant:>6} | "
            f"{o['ker']:6.2f} {100 * (o['ker'] / g - 1):+7.1f}% | "
            f"{pack_col} {o['e2e']:6.2f} {100 * (o['e2e'] / g - 1):+7.1f}% | {fringe_col}"
        )

    bracketed = sorted({r["case"] for r in plan if r["kind"] == "bracket"})
    if bracketed:
        print()
        print(
            "  NOTE: the reference ships no kernel at the pinned MR/NR for "
            + ", ".join(bracketed)
            + "."
        )
        print("        Those rows are the nearest even-NR neighbours, so they bracket")
        print("        the true reference number rather than matching it exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
