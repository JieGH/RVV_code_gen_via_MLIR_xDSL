#!/usr/bin/env python3
"""The reference design alone, over many runs, beside our two packers (issue #29).

`run_ref_sweep.sh` runs the reference design (`run_ref_baseline.sh`, nothing of ours in it)
once every 30 min on two boards, the way the #26 sweeps ran ours. This reads the reference runs
it made and puts them per layer next to two sets of our runs of the same layers:

  vector pack : the overnight sweep of 19-20 Sep (`run_*` dirs with a stamp in --vector-from ..
                --vector-to, log `all_raw_*_noinstr_refproto_no_nc_override.txt`)
  scalar pack : the runs of a scalar-pack sweep (`logs/sweep_scalar_*/runs.txt`, issue #26)

All three are GFLOPS = 2*M*N*K / mean time per whole GEMM call, packing included: the reference's
own formula (`src/test_gemm.c`), and our `RVV_Time_ms`. Ours is uninstrumented, reference
protocol, `--no-nc-overrides`, so it blocks the way the reference does.

The reference ships no odd-NR kernel, so a layer pinned at 32x7 or 16x15 has two rows, the even
neighbours either side (`bracket`). Those are not "the reference number" and the table says so.

`vector/ref` and `scalar/ref` are (our mean - reference mean) / reference mean: positive means we
are faster. A difference no larger than the larger of the two spreads between runs (best - worst
over mean, the reference's own is the `spread ref` column) is called "within noise"; a larger one
is called faster or slower. With one run on both sides there is no spread and no call.

Usage:
  python3 compare_ref_sweep.py --sweep logs/sweep_ref_<stamp>
  python3 compare_ref_sweep.py --ref logs/refbase_<stamp> [logs/refbase_<stamp> ...]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH_DIR = HERE.parent  # tests/pure_xdsl_blis_rvv
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(HERE))

from compare_pack_swap import Case, load_side, mean, spread_pct  # noqa: E402
from compare_ref_baseline import load_plan, load_reference, pin_of, pin_words, stamp_of  # noqa: E402
from report_scalar_sweep import LAYER_ORDER, PUBLISHED, VECTOR_GLOB, board_of  # noqa: E402

LOGS = BENCH_DIR / "logs"
VECTOR_FROM = "2026-09-19_230000"
VECTOR_TO = "2026-09-20_070000"


@dataclass
class RefRow:
    """One (layer, built kernel) of the reference: its GFLOPS in every run that measured it."""

    layer: str
    shape: str
    pinned: str  # what we pin this layer at, e.g. "32x7"
    built: str  # what the reference built, e.g. "32x8"
    kind: str  # "exact" or "bracket"
    gflops: list[float] = field(default_factory=lambda: [])


def load_ref_runs(dirs: list[Path]) -> dict[tuple[str, str], RefRow]:
    """(layer, built) -> the reference's GFLOPS by run. A run that left no CSV for a row leaves
    that row's list one entry short, and the table shows the run count, so a hole is visible."""
    rows: dict[tuple[str, str], RefRow] = {}
    for d in dirs:
        if not (d / "run_plan.tsv").is_file():
            raise SystemExit(f"{d}: no run_plan.tsv (is this a run_ref_baseline.sh result dir?)")
        gflops = load_reference(d)
        for p in load_plan(d):
            hit = gflops.get((p["tag"], p["numid"]))
            row = rows.setdefault(
                (p["case"], p["built"]),
                RefRow(p["case"], f"{p['m']}x{p['n']}x{p['k']}", p["pinned"], p["built"], p["kind"]),
            )
            if hit is not None:
                row.gflops.append(hit[0])
    return rows


def runs_in_window(logs: Path, lo: str, hi: str, glob: str) -> list[Path]:
    """run_* dirs of `logs` whose stamp is in lo..hi and that hold a log matching `glob`."""
    out: list[Path] = []
    for d in sorted(logs.glob("run_*")):
        stamp = d.name[len("run_") :]
        if lo <= stamp <= hi and any(d.glob(glob)):
            out.append(d)
    return out


def runs_of_sweep(logs: Path, sweep: Path) -> list[Path]:
    """The run dirs a sweep's runs.txt lists (`board<TAB>run dir`)."""
    return [logs / line.split("\t")[1] for line in (sweep / "runs.txt").read_text().splitlines() if "\t" in line]


def ours_for(side: dict[tuple[str, str], Case], layer: str, pinned: str) -> Case | None:
    """Our runs of `layer` at the MR/NR it is pinned at."""
    mr, nr = pinned.split("x")
    return side.get((layer, f"MR{mr}_NR{nr}"))


def diff_cell(ours: Case | None, ref: RefRow) -> str:
    """Our mean against the reference's, with a call against the spread between runs."""
    if ours is None or not ours.gops or not ref.gflops:
        return "-"
    diff = 100.0 * (mean(ours.gops) - mean(ref.gflops)) / mean(ref.gflops)
    if len(ours.gops) < 2 and len(ref.gflops) < 2:
        return f"{diff:+.1f}% (one run each: no noise estimate)"
    noise = max(spread_pct(ours.gops), spread_pct(ref.gflops))
    call = "within noise" if abs(diff) <= noise else ("faster" if diff > 0 else "slower")
    return f"{diff:+.1f}% ({call}, noise {noise:.1f}%)"


def mean_range_n(xs: list[float]) -> str:
    """mean, range, run count, in one cell."""
    if not xs:
        return "-"
    rng = f"{min(xs):.2f}-{max(xs):.2f}, " if len(xs) > 1 else ""
    return f"{mean(xs):.2f} ({rng}n={len(xs)})"


def render(
    ref: dict[tuple[str, str], RefRow],
    vector: dict[tuple[str, str], Case],
    scalar: dict[tuple[str, str], Case],
) -> str:
    headers = ["Layer", "Shape", "Pinned", "Ref build", "Kind", "reference", "vector pack", "scalar pack",
               "vector/ref", "scalar/ref", "spread ref", "published", "ref/published"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    order = {name: i for i, name in enumerate(LAYER_ORDER)}
    for (layer, built), row in sorted(ref.items(), key=lambda kv: (order.get(kv[0][0], 99), kv[0][1])):
        v = ours_for(vector, layer, row.pinned)
        s = ours_for(scalar, layer, row.pinned)
        pub = PUBLISHED.get(layer)
        cells = [
            layer, row.shape, row.pinned, built, row.kind,
            mean_range_n(row.gflops), mean_range_n(v.gops if v else []), mean_range_n(s.gops if s else []),
            diff_cell(v, row), diff_cell(s, row),
            f"{spread_pct(row.gflops):.1f}%" if len(row.gflops) > 1 else "-",
            f"{pub:.1f}" if pub else "",
            # Not for a bracket row: the published figure is for the pinned MR/NR, not the neighbour's.
            f"{100.0 * (mean(row.gflops) - pub) / pub:+.1f}%" if pub and row.gflops and row.kind == "exact" else "",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def conditions(ref_dirs: list[Path], vector_dirs: list[Path], scalar_dirs: list[Path]) -> list[str]:
    """What each set ran under: how many runs, on which boards, from when to when, on which core,
    and a `WARNING` when the sets were not all pinned to one core (as compare_ref_baseline.py does)."""
    lines: list[str] = []
    all_pins: set[str | None] = set()
    for label, dirs in (("reference", ref_dirs), ("vector pack", vector_dirs), ("scalar pack", scalar_dirs)):
        if not dirs:
            lines.append(f"{label:12}: no runs")
            continue
        boards: dict[str, int] = {}
        for d in dirs:
            b = board_of(d)
            boards[b] = boards.get(b, 0) + 1
        stamps = sorted(t for t in map(stamp_of, dirs) if t is not None)
        span = f"{stamps[0]:%d %b %H:%M} to {stamps[-1]:%d %b %H:%M}" if stamps else "unknown"
        all_pins |= {pin_of(d) for d in dirs}
        pins = sorted({pin_words(pin_of(d)) for d in dirs})
        lines.append(
            f"{label:12}: {len(dirs)} runs ({', '.join(f'{b} {n}' for b, n in sorted(boards.items()))}), "
            f"{span}, pinned: {', '.join(pins)}"
        )
    if None in all_pins:
        lines.append("WARNING: pinning unknown for some runs (no provenance.txt, or a run from before issue #21): "
                     "cannot tell that every set ran on the same core.")
    elif "none" in all_pins:
        lines.append("WARNING: some runs were not pinned: the scheduler may have moved them between cores.")
    elif len(all_pins) > 1:
        lines.append("WARNING: the sets were pinned to different cores.")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", type=Path, help="a logs/sweep_ref_<stamp> dir (reads its runs.txt)")
    ap.add_argument("--ref", nargs="+", type=Path, metavar="DIR", help="refbase_<stamp> dirs, instead of --sweep")
    ap.add_argument("--scalar-sweep", type=Path,
                    help="a scalar-pack sweep dir (default: the newest logs/sweep_scalar_*)")
    ap.add_argument("--vector-from", default=VECTOR_FROM, help=f"first vector-pack run stamp (default {VECTOR_FROM})")
    ap.add_argument("--vector-to", default=VECTOR_TO, help=f"last vector-pack run stamp (default {VECTOR_TO})")
    ap.add_argument("--logs", type=Path, default=LOGS, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if bool(args.sweep) == bool(args.ref):
        ap.error("give exactly one of --sweep and --ref")
    ref_dirs: list[Path] = args.ref or runs_of_sweep(args.logs, args.sweep)
    if not ref_dirs:
        print("no reference runs to read", file=sys.stderr)
        return 1

    scalar_sweep = args.scalar_sweep
    if scalar_sweep is None:
        found = sorted(args.logs.glob("sweep_scalar_*"))
        scalar_sweep = found[-1] if found else None
    scalar_dirs = runs_of_sweep(args.logs, scalar_sweep) if scalar_sweep and (scalar_sweep / "runs.txt").is_file() else []
    vector_dirs = runs_in_window(args.logs, args.vector_from, args.vector_to, VECTOR_GLOB)

    failed: list[str] = []
    ref = load_ref_runs(ref_dirs)
    vector = load_side(vector_dirs, False, True, failed) if vector_dirs else {}
    scalar = load_side(scalar_dirs, True, True, failed) if scalar_dirs else {}

    print("GFLOPS = 2*M*N*K / mean time per whole GEMM call, packing included, on all three sides.")
    print("reference = the reference design alone: its scalar packers, its Exo kernels (the set and variant are")
    print("in each run's provenance.txt), its blocking.")
    print("Cells read: mean (lowest-highest run, n = runs that measured it).")
    print()
    for line in conditions(ref_dirs, vector_dirs, scalar_dirs):
        print(line)
    print()
    print(render(ref, vector, scalar))
    bracketed = sorted({r.layer for r in ref.values() if r.kind == "bracket"})
    if bracketed:
        print()
        print("`bracket` rows: the reference ships no kernel at the pinned MR/NR for " + ", ".join(bracketed)
              + " (NR must be even), so these are the nearest even-NR kernels. They bracket the number a")
        print("true reference kernel would give; do not quote one as 'the reference number'. The two neighbours are")
        print("not equally close: 32x8 and 16x16 hold 32 C accumulators plus the A registers, more than the 32 vector")
        print("registers there are, so they spill and ran at about half the speed of 32x6 and 16x14 in the first")
        print("reference run. The lower-NR row is the nearer one.")
    print()
    print("Limits:")
    print("  - The three sets were not made in the same hours; time of day and board temperature are not controlled.")
    print("  - At tmin 10, S3-S5 are about one rep per run on both sides (a single cold rep); the reference has no warm-up.")
    print("  - The reference writes its GFLOPS with 3 significant digits, so its figures carry a rounding of about 0.1%.")
    if failed:
        print()
        print("WARNING: rows of ours that FAILED the correctness check and are left out: " + ", ".join(failed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
