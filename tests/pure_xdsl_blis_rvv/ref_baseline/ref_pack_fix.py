#!/usr/bin/env python3
"""Issue #54: the reference design with our one-column B packer, from a patched STAGED copy of gemm_blis.c.

The reference packs B with `pack_CB` (gemm_blis.c): for each k it reads one number from each of the
panel's columns, one scalar load each, `ldB * 4` bytes apart (the same order as our current xDSL packer).
This module

  patch    renames that function to `pack_CB_orig` (kept, and still used for every case the fix does
           not cover) and appends a new `pack_CB` that packs a column-major fp32 B one column at a
           time, 32 numbers per load: `vle32` (LMUL 4) down the column, `vsse32` with stride RR*4
           into the Bc panel. This is col-g1-m4 of templates/pack_b_variants.h, written in the
           reference's own C. Every number lands where pack_CB_orig puts it
           (Mc[j*mc + i*RR + jj] = M(i, j+jj)), and the pad slots of a narrow last panel are not
           written, as in the original. Off RISC-V (host tests) the same loops copy with plain C.

Like ref_per_rep.py it only ever rewrites a copy that run_ref_baseline.sh has made, and refuses
anything under an `ext/` path. Nothing else in gemm_blis.c changes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

MARK = "issue #54: one-column B packer"

ANCHOR = "void pack_CB( char orderM, char transM, int mc, int nc, DTYPE *M, int ldM, DTYPE *Mc, int RR ){"
RENAMED = "static void pack_CB_orig( char orderM, char transM, int mc, int nc, DTYPE *M, int ldM, DTYPE *Mc, int RR ){"

STRIP = 32  # numbers per load: e32, LMUL 4, VLEN 256

FIX = f"""

/* {MARK} (col-g1-m4 of RVV_code_gen_via_MLIR_xDSL). A column-major fp32 B is read one
   column at a time, {STRIP} numbers per load, and each strip is stored down its column of the Bc panel.
   Same destination for every number as pack_CB_orig; every other case still goes to pack_CB_orig. */
#include <stddef.h>
#if defined(FP32) && defined(__riscv_vector)
#include <riscv_vector.h>
#endif
void pack_CB( char orderM, char transM, int mc, int nc, DTYPE *M, int ldM, DTYPE *Mc, int RR ){{
#if defined(FP32)
  if ( ((transM=='N')&&( orderM=='C'))||
       ((transM=='T')&&( orderM=='R')) ) {{
    int j, jj, i, nr, len;
    for ( j=0; j<nc; j+=RR ) {{
      DTYPE *Bp = &Mc[j*mc];
      nr = min( nc-j, RR );
      for ( jj=0; jj<nr; jj++ )
        for ( i=0; i<mc; i+={STRIP} ) {{
          len = min( mc-i, {STRIP} );
          DTYPE *src = &Mcol(i,j+jj);
          DTYPE *dst = &Bp[i*RR + jj];
#if defined(__riscv_vector)
          size_t vl = __riscv_vsetvl_e32m4( (size_t) len );
          vfloat32m4_t v = __riscv_vle32_v_f32m4( src, vl );
          __riscv_vsse32_v_f32m4( dst, (ptrdiff_t) RR * 4, v, vl );
#else
          int e;
          for ( e=0; e<len; e++ ) dst[e*RR] = src[e];
#endif
        }}
    }}
    return;
  }}
#endif
  pack_CB_orig( orderM, transM, mc, nc, M, ldM, Mc, RR );
}}
"""


class PatchError(Exception):
    pass


def patch_gemm_blis(src: str) -> str:
    """The reference's gemm_blis.c with the one-column B packer. PatchError if it is already patched
    or the pack_CB definition is not found exactly once."""
    if MARK in src:
        raise PatchError("gemm_blis.c is already patched")
    n = src.count(ANCHOR)
    if n != 1:
        raise PatchError(f"pack_CB definition found {n} times in gemm_blis.c, expected once: the reference changed")
    return src.replace(ANCHOR, RENAMED) + FIX


def _cmd_patch(args: argparse.Namespace) -> int:
    path = args.file.resolve()
    if "ext" in path.parts:
        print(f"REFUSING to modify the vendored reference tree: {path}", file=sys.stderr)
        print("Point this at a staged copy instead.", file=sys.stderr)
        return 1
    try:
        path.write_text(patch_gemm_blis(path.read_text()))
    except PatchError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"  [PATCHED] {path.name}: pack_CB reads B one column at a time, {STRIP} numbers per load (issue #54)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("patch", help="patch a staged gemm_blis.c in place")
    p.add_argument("file", type=Path)
    p.set_defaults(fn=_cmd_patch)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
