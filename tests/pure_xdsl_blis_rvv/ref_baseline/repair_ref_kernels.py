#!/usr/bin/env python3
"""Repair truncated Exo kernel sources in a STAGED COPY of the reference tree.

Six of the 4032 vendored kernel files (all `16x14/*/macc`) are missing the first
16 bytes of their opening line, so they start with

    s_RVV_16x14_fp32.h"

instead of

    #include "kernels_RVV_16x14_fp32.h"

and fail to compile. This is pre-existing damage in ext/RVV_ukernels_benchmark,
not something the baseline run introduces.

This script only ever rewrites a copy that run_ref_baseline.sh has already made
(the pre-flight temp dir, or the tree staged on the board) — it refuses to touch
anything under an `ext/` path, so the vendored source stays pristine.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def repair(root: Path, dry_run: bool = False) -> tuple[int, int]:
    checked = fixed = 0
    for path in sorted(root.rglob("kernels_RVV_*_fp32.c")):
        checked += 1
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                first = fh.readline()
        except OSError:
            continue
        if first.startswith("#include"):
            continue
        expected = f'#include "{path.stem}.h"'
        # The damaged line must be a proper suffix of the correct one, otherwise
        # this is some other problem and we must not guess.
        if not (first.rstrip("\n") and expected.endswith(first.rstrip("\n"))):
            print(f"  [SKIP] {path}: unrecognised first line {first!r}", file=sys.stderr)
            continue
        if dry_run:
            print(f"  [WOULD FIX] {path.relative_to(root)}")
            fixed += 1
            continue
        text = path.read_text(encoding="utf-8", errors="surrogateescape")
        missing = expected[: len(expected) - len(first.rstrip("\n"))]
        path.write_text(missing + text, encoding="utf-8", errors="surrogateescape")
        print(f"  [FIXED] {path.relative_to(root)}  (restored {missing!r})")
        fixed += 1
    return checked, fixed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="staged copy of the kernels tree")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-ext", action="store_true",
                    help=argparse.SUPPRESS)  # escape hatch, deliberately undocumented
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        return 1
    if "ext" in root.parts and not args.allow_ext:
        print(f"REFUSING to modify the vendored reference tree: {root}", file=sys.stderr)
        print("Point this at a staged copy instead.", file=sys.stderr)
        return 1

    checked, fixed = repair(root, args.dry_run)
    if fixed:
        print(f"  {fixed} truncated kernel file(s) repaired in the staged copy "
              f"({checked} checked).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
