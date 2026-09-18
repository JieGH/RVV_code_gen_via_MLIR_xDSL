#!/usr/bin/env python3
"""Generate the reference design's per-(MR,NR) .dat files for our exact case set.

The reference driver (ext/RVV_ukernels_benchmark/gemm_blis_family) bakes MR and NR
in at compile time (-DMR/-DNR, and the kernel source path
kernels/RVV_256_<OPT>/fp32/<MR>x<NR>/), so one binary can only cover the cases that
share a pinning. This script therefore splits cases.tsv into one .dat per (MR,NR)
group and reports, for each group, whether the reference actually ships that kernel.

It also predicts MC/NC/KC using tests/pure_xdsl_blis_rvv/cache_config.py, which is a
straight port of the reference's own modelLevel/model_level.c (identical constants and
identical clamp order), so the predicted blocking is what the reference will pick.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH_DIR = HERE.parent                      # tests/pure_xdsl_blis_rvv
sys.path.insert(0, str(BENCH_DIR))
from cache_config import get_optim_mc_nc_kc  # noqa: E402

DATA_SIZE = 4  # fp32


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, numid, m, n, k, mr, nr = line.split()
        cases.append({
            "case": name, "numid": int(numid),
            "m": int(m), "n": int(n), "k": int(k),
            "mr": int(mr), "nr": int(nr),
        })
    return cases


def kernel_dir(kernels_root: Path, opt: str, mr: int, nr: int,
               swap: int, gather: int) -> Path:
    load = "loadBA" if swap else "loadAB"
    mode = {0: "bcast", 1: "gather", 2: "macc"}[gather]
    return kernels_root / f"RVV_256_{opt}" / "fp32" / f"{mr}x{nr}" / load / mode


def nearest_available(kernels_root: Path, opt: str, mr: int, nr: int,
                      swap: int, gather: int) -> list[tuple[int, int]]:
    """Even-NR neighbours that bracket an unavailable odd NR."""
    out = []
    for cand_nr in (nr - 1, nr + 1):
        if cand_nr < 2:
            continue
        if kernel_dir(kernels_root, opt, mr, cand_nr, swap, gather).is_dir():
            out.append((mr, cand_nr))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=Path, default=HERE / "cases.tsv")
    ap.add_argument("--kernels-root", type=Path,
                    default=HERE.parents[3] / "ext" / "RVV_ukernels_benchmark" / "kernels",
                    help="ext/RVV_ukernels_benchmark/kernels")
    ap.add_argument("--out-dir", type=Path, default=HERE / "generated",
                    help="where the .dat files are written")
    ap.add_argument("--opt", default="OPT", choices=["BASE", "LDX", "OPT", "ALL"])
    ap.add_argument("--swap", type=int, default=0, choices=[0, 1])
    ap.add_argument("--gather", type=int, default=2, choices=[0, 1, 2],
                    help="0=bcast 1=gather 2=macc. Default macc: it is the "
                         "vfmacc.vf variant that matches our AXPY kernel. "
                         "bcast materializes B into 6 extra vector regs "
                         "(34 live > 32) and provably spills.")
    ap.add_argument("--tmin", type=float, default=3.0,
                    help="the reference's TIMIN (run_gemm.sh): it repeats a shape "
                         "until cumulative time exceeds this, then divides by the "
                         "rep count. Used here only to warn about under-sampling.")
    ap.add_argument("--assumed-gflops", type=float, default=8.0,
                    help="rough throughput used to estimate the reference's rep count")
    ap.add_argument("--bracket", action="store_true",
                    help="for a pinning the reference does not ship (odd NR), also emit "
                         "datasets for the nearest even-NR neighbours so the exact "
                         "config can be bracketed instead of skipped")
    args = ap.parse_args()

    if not args.kernels_root.is_dir():
        print(f"ERROR: kernels root not found: {args.kernels_root}", file=sys.stderr)
        return 1

    cases = load_cases(args.cases)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.out_dir.glob("*.dat"):
        stale.unlink()
    for stale in args.out_dir.glob("*.plan"):
        stale.unlink()

    groups: dict[tuple[int, int], list[dict]] = {}
    for c in cases:
        groups.setdefault((c["mr"], c["nr"]), []).append(c)

    plan_rows: list[str] = []
    # A single built (MR,NR) can serve more than one pinning group — e.g. 32x6 is
    # the exact kernel for B1/B5 AND the lower bracket for the 32x7 cases. Collect
    # members per built tag first so each .dat is written once with everything it
    # must cover, instead of a later group clobbering an earlier one.
    by_tag: dict[tuple[int, int], list[tuple[dict, str]]] = {}

    print(f"  Reference kernels : {args.kernels_root}")
    print(f"  Variant           : OPT={args.opt} SWAP={args.swap} GATHER={args.gather} "
          f"({'loadBA' if args.swap else 'loadAB'}/"
          f"{ {0:'bcast',1:'gather',2:'macc'}[args.gather] })")
    print()

    for (mr, nr), members in sorted(groups.items()):
        names = ",".join(m["case"] for m in members)
        exact = kernel_dir(args.kernels_root, args.opt, mr, nr, args.swap, args.gather)
        have_exact = exact.is_dir()

        builds: list[tuple[int, int, str]] = []
        if have_exact:
            builds.append((mr, nr, "exact"))
        elif args.bracket:
            for bmr, bnr in nearest_available(args.kernels_root, args.opt, mr, nr,
                                              args.swap, args.gather):
                builds.append((bmr, bnr, "bracket"))

        status = "OK" if have_exact else ("BRACKETED" if builds else "UNAVAILABLE")
        print(f"  [{status:^11}] {mr}x{nr}  cases: {names}")
        if not have_exact:
            print(f"                 reference ships no {mr}x{nr} kernel "
                  f"(NR must be even, 2..32; MR in 8,16,24,32,40,48)")
            if builds:
                print("                 bracketing with: "
                      + ", ".join(f"{a}x{b}" for a, b, _ in builds))
            else:
                print("                 re-run with --bracket to measure the "
                      "nearest available neighbours instead")

        for bmr, bnr, kind in builds:
            for c in members:
                by_tag.setdefault((bmr, bnr), []).append((c, kind))

    print()
    for (bmr, bnr), entries in sorted(by_tag.items()):
        tag = f"{bmr}x{bnr}"
        dat = args.out_dir / f"xdsl_cases_{tag}.dat"
        # The reference parses this with strtok(line, "\t") — TABS ONLY — and
        # atoi() on column 0, so labels must be numeric (see src/inutils.c).
        lines = ["#layer\tm\tn\tk\tFlops"]
        for c, _kind in entries:
            lines.append(f"{c['numid']}\t{c['m']}\t{c['n']}\t{c['k']}\t0")
        dat.write_text("\n".join(lines) + "\n")
        covered = ",".join(c["case"] for c, _ in entries)
        print(f"  build {tag:>6} -> {dat.relative_to(HERE)}  ({len(entries)} cases: {covered})")

        for c, kind in entries:
            mc, nc, kc = get_optim_mc_nc_kc(
                "jbpi2", DATA_SIZE, bmr, bnr,
                mmax=c["m"], nmax=c["n"], kmax=c["k"])
            plan_rows.append(
                f"{c['case']}\t{c['numid']}\t{c['m']}\t{c['n']}\t{c['k']}\t"
                f"{c['mr']}\t{c['nr']}\t{bmr}\t{bnr}\t{kind}\t{mc}\t{nc}\t{kc}\t{tag}")

    # ---- Under-sampling warning -------------------------------------------
    # The reference has NO warm-up: it runs `while (time <= tmin) { nreps++; ... }`
    # and divides the total by nreps (src/test_gemm.c:340-360), so the cold first
    # rep is always inside the average. When one rep already exceeds tmin the loop
    # stops at nreps=1 and the reported GFLOPS is a single cold sample.
    # This bites the large Square shapes and not the BERT ones, which is exactly
    # the kind of asymmetry that makes one family look like it "matches" and the
    # other not.
    seen: set[str] = set()
    warn: list[tuple[str, float, int]] = []
    for row in plan_rows:
        f = row.split("\t")
        if f[0] in seen:
            continue
        seen.add(f[0])
        m, n, k = int(f[2]), int(f[3]), int(f[4])
        t_rep = 2.0 * m * n * k / (args.assumed_gflops * 1e9)
        nreps = max(1, int(args.tmin // t_rep) + 1)
        warn.append((f[0], t_rep, nreps))

    thin = [w for w in warn if w[2] < 3]
    if thin:
        print()
        print(f"  !! Under-sampling warning (reference TIMIN={args.tmin}s, "
              f"assuming ~{args.assumed_gflops:g} GFLOPS):")
        print(f"     {'case':5} {'est s/rep':>10} {'est reps':>9}")
        for case, t_rep, nreps in warn:
            mark = "  <-- single cold sample" if nreps == 1 else (
                   "  <-- thin" if nreps < 3 else "")
            print(f"     {case:5} {t_rep:10.2f} {nreps:9d}{mark}")
        worst = max(w[1] for w in thin)
        print()
        print(f"     The reference does no warm-up, so a 1-rep case is one cold run.")
        print(f"     For >=5 reps on the slowest of these, pass --tmin {worst*5:.0f}")
        print(f"     to run_ref_baseline.sh (costs board time, buys comparability).")

    plan = args.out_dir / "run_plan.tsv"
    plan.write_text(
        "# case\tnumid\tM\tN\tK\tpinned_MR\tpinned_NR\tbuilt_MR\tbuilt_NR\tkind\t"
        "pred_MC\tpred_NC\tpred_KC\ttag\n" + "\n".join(plan_rows) + "\n")
    print()
    print(f"  Run plan          : {plan.relative_to(HERE)}  ({len(plan_rows)} rows)")
    print()
    print("  Predicted blocking (reference recomputes this itself from MR/NR via")
    print("  modelLevel/model_level.c; shown here so a mismatch is obvious):")
    print(f"    {'case':5} {'shape':18} {'built':>7} {'MC':>6} {'NC':>6} {'KC':>5}")
    for row in plan_rows:
        f = row.split("\t")
        print(f"    {f[0]:5} {f[2]+'x'+f[3]+'x'+f[4]:18} {f[7]+'x'+f[8]:>7} "
              f"{f[10]:>6} {f[11]:>6} {f[12]:>5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
