#!/usr/bin/env python3
"""Issue #46: per-rep times of the reference design, from a patched STAGED copy of test_gemm.c.

The reference driver (ext/RVV_ukernels_benchmark/gemm_blis_family/src/test_gemm.c) times a case
with `while (time <= tmin) { nreps++; gemm_blis_B3A2C0(...); t2 = dclock(); time = t2 - t1; }`
and keeps only the total, so two speeds inside a run average into one number. This module

  patch    edits a staged copy of test_gemm.c so the clock read after every call is stored in a
           static array (in .bss, touched before the timed loop) and writes it with open(2) and
           write(2) from a static buffer, so nothing is taken from the heap and the driver's own
           buffers land where they do in the unchanged driver; nothing printed inside the loop and written out after the loop, to `<result csv>.reps`;
  convert  turns the driver's numeric layer id back into our label, adds the columns of our own
           reps_*.txt so reps_report.py reads it unchanged, and gzips it.

Like repair_ref_kernels.py it only ever rewrites a copy that run_ref_baseline.sh has made, and
refuses anything under an `ext/` path. The timed loop itself is not touched: the patch adds one
compare and one store after the clock read the driver already does.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import sys
from pathlib import Path

REP_CAP = 131072  # calls kept per case (1 MiB of doubles); the shortest case runs ~2 ms a call, so ~5000 in 10 s

HEADER = "Layer,M,N,K,Variant,MC,NC,KC,Rep,Seconds"
OUR_HEADER = HEADER + ",Pack_Seconds,Compute_Seconds,Pack_Cycles,Compute_Cycles"
MARK = "issue #46: per-rep record"


class PatchError(Exception):
    pass


ANCHOR_INC = "#include <string.h>"
ANCHOR_DECL = "FILE *fd_csv;"
ANCHOR_OPEN = 'fprintf(fd_csv, "#l;m;n;k;Gflops\\n");'
ANCHOR_STAMP = "t2   = dclock();"
ANCHOR_AFTER = "time = time/nreps;"

INC = ANCHOR_INC + """
#include <fcntl.h>  /* issue #46 */"""

# Nothing here touches the heap: the driver aligns A, B, C, Ac, Bc to 32 bytes only, so where each
# lands depends on everything malloc'd before it, and B3-like cases move by up to 30% with that (a
# malloc'd array and a second FILE both did, issue #46). The array and the row buffer are static, the
# file is open(2) + write(2).
DECL = f"""FILE *fd_csv;
  /* {MARK}: clock after every call, kept in memory until the timed loop is over */
  #define REP_CAP {REP_CAP}
  static double rep_stamp[REP_CAP];  /* .bss */
  static char   rep_row[4096];
  int     fd_reps = -1;
  char    reps_name[1024];"""

OPEN = ANCHOR_OPEN + f"""
  /* {MARK}: file open and the array's pages touched, all before any timing */
  snprintf(reps_name, sizeof(reps_name), "%s%s", argv[argc - 1], ".reps");
  fd_reps = open(reps_name, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  memset(rep_stamp, 0, sizeof(rep_stamp));
  {{
    int hdr = snprintf(rep_row, sizeof(rep_row), "{HEADER}\\n");
    if ( write(fd_reps, rep_row, hdr) != hdr ) fprintf(stderr, "per-rep header not written\\n");
  }}"""

STAMP = ANCHOR_STAMP + """
	    if ( nreps <= REP_CAP ) rep_stamp[nreps - 1] = t2;  /* issue #46 */"""

AFTER = ANCHOR_AFTER + """
	  /* issue #46: one row per call, seconds = difference of the clock reads (the first from t1) */
	  {
	    size_t used = 0;
	    for ( i = 0; i < nreps && i < REP_CAP; i++ ) {
	      used += snprintf(rep_row + used, sizeof(rep_row) - used, "%d,%zu,%zu,%zu,MR%d_NR%d,%zu,%zu,%zu,%zu,%.9f\\n",
	                       cnn_enable ? testConf->cnn[cnn_i].layer : 0, m, n, k, MR, NR, mc, nc, kc, i,
	                       rep_stamp[i] - ( i ? rep_stamp[i - 1] : t1 ));
	      if ( used > sizeof(rep_row) - 256 ) {
	        if ( write(fd_reps, rep_row, used) != (ssize_t) used ) fprintf(stderr, "per-rep rows not written\\n");
	        used = 0;
	      }
	    }
	    if ( used && write(fd_reps, rep_row, used) != (ssize_t) used ) fprintf(stderr, "per-rep rows not written\\n");
	  }"""


def _once(src: str, anchor: str) -> None:
    n = src.count(anchor)
    if n != 1:
        raise PatchError(f"anchor {anchor!r} found {n} times in test_gemm.c, expected once: the reference driver changed")


def patch_test_gemm(src: str) -> str:
    """The reference's test_gemm.c with the per-rep record added. PatchError if it is already
    patched or any anchor is not found exactly once."""
    if MARK in src:
        raise PatchError("test_gemm.c is already patched")
    for a in (ANCHOR_INC, ANCHOR_DECL, ANCHOR_OPEN, ANCHOR_STAMP, ANCHOR_AFTER):
        _once(src, a)
    for old, new in ((ANCHOR_INC, INC), (ANCHOR_DECL, DECL), (ANCHOR_OPEN, OPEN), (ANCHOR_STAMP, STAMP), (ANCHOR_AFTER, AFTER)):
        src = src.replace(old, new)
    return src


def load_labels(cases: Path) -> dict[int, str]:
    """{numeric layer id: label} from a cases.tsv (`case numid M N K MR NR`)."""
    out: dict[int, str] = {}
    for line in cases.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, numid, *_ = line.split()
            out[int(numid)] = name
    return out


def convert_reps(raw: str, labels: dict[int, str]) -> str:
    """The driver's per-rep file in the layout of our reps_*.txt: label for layer id, and the
    instrumented-only columns as nan (the reference has no pack/compute split)."""
    rows = list(csv.reader(io.StringIO(raw)))
    if not rows or ",".join(rows[0]) != HEADER:
        raise PatchError(f"unexpected header in the per-rep file: {rows[:1]}")
    out = [OUR_HEADER]
    for row in rows[1:]:
        if not row:
            continue
        layer = int(row[0])
        if layer not in labels:
            raise PatchError(f"layer id {layer} is not in the cases file")
        out.append(",".join([labels[layer], *row[1:], "nan", "nan", "nan", "nan"]))
    return "\n".join(out) + "\n"


def total_matches_driver(seconds: list[float], ops: float, driver_gflops: str, tol: float = 0.02) -> bool:
    """The per-rep seconds, summed, against the GFLOPS the driver printed for the case (it prints
    3 significant digits and uses gettimeofday, hence the tolerance)."""
    mine = len(seconds) * ops / (sum(seconds) * 1e9)
    return abs(mine - float(driver_gflops)) <= tol * float(driver_gflops)


def rows_may_be_truncated(n_rows: int) -> bool:
    return n_rows >= REP_CAP


def _cmd_patch(args: argparse.Namespace) -> int:
    path = args.file.resolve()
    if "ext" in path.parts:
        print(f"REFUSING to modify the vendored reference tree: {path}", file=sys.stderr)
        print("Point this at a staged copy instead.", file=sys.stderr)
        return 1
    try:
        path.write_text(patch_test_gemm(path.read_text()))
    except PatchError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"  [PATCHED] {path.name}: per-rep record after each call (issue #46)")
    return 0


def _driver_gflops(result_dir: Path, tag_variant: str, numid: int) -> str | None:
    """The `Gflops` column of result_<tag>_<variant>.csv for one layer id."""
    f = result_dir / f"result_{tag_variant}.csv"
    if not f.is_file():
        return None
    for line in f.read_text().splitlines():
        parts = line.split(";")
        if len(parts) == 5 and parts[0] == str(numid):
            return parts[4]
    return None


def _cmd_convert(args: argparse.Namespace) -> int:
    labels = load_labels(args.cases)
    by_label = {v: k for k, v in labels.items()}
    bad = 0
    for raw_path in sorted(args.result_dir.glob("result_*.csv.reps")):
        tag_variant = raw_path.name[len("result_"):-len(".csv.reps")]
        text = convert_reps(raw_path.read_text(), labels)
        out = args.result_dir / f"reps_ref_{tag_variant}.txt.gz"
        out.write_bytes(gzip.compress(text.encode()))
        raw_path.unlink()
        # Cross-check each case's total against what the driver itself wrote.
        rows = list(csv.DictReader(io.StringIO(text)))
        for label in dict.fromkeys(r["Layer"] for r in rows):
            mine = [r for r in rows if r["Layer"] == label]
            secs = [float(r["Seconds"]) for r in mine]
            ops = 2.0 * int(mine[0]["M"]) * int(mine[0]["N"]) * int(mine[0]["K"])
            gf = _driver_gflops(args.result_dir, tag_variant, by_label[label])
            note = ""
            if gf is None:
                note = "  (no driver GFLOPS to check against)"
            elif not total_matches_driver(secs, ops, gf):
                note = f"  MISMATCH: driver says {gf} GFLOPS"
                bad += 1
            if rows_may_be_truncated(len(secs)):
                note += f"  WARNING: {len(secs)} rows = the array size; later reps may be missing"
                bad += 1
            print(f"  [OK] {out.name} {label}: {len(secs)} reps, {sum(secs):.2f} s{note}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("patch", help="patch a staged test_gemm.c in place")
    p.add_argument("file", type=Path)
    p.set_defaults(fn=_cmd_patch)
    c = sub.add_parser("convert", help="turn result_*.csv.reps of a result dir into reps_ref_*.txt.gz")
    c.add_argument("result_dir", type=Path)
    c.add_argument("--cases", type=Path, required=True, help="the cases.tsv the run used")
    c.set_defaults(fn=_cmd_convert)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
