# One Reference Design patch step

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/120

**Parent:** #115, codebase architecture optimization.
**Role / Label:** `needs-triage`. Spec, 2026-11-20. Strength: **Speculative** (low churn, small gain).
**Status:** open. Not started. Host-only.
**Branch:** cut from `pure-xdsl` when work starts, named after this title.

---

## 1. Problem

Four modules in `ref_baseline/` change the Reference Design's C source by string match:

- `ref_pack_fix.py`
- `repair_ref_kernels.py`
- `ref_per_rep.py` (219 lines; `patch_gemm_blis`, `patch_test_gemm`)
- `fringe.py`

Each takes a path and rewrites text. Each has its own test (`test_ref_pack_fix.py`, `test_ref_per_rep.py`, `test_fringe_overhead.py`). The interface is about as big as what each does (shallow). If the reference source changes, a match can fail quietly and the run uses an unpatched file.

`compare_ref_baseline.py` (358 lines) and `compare_ref_sweep.py` (239 lines) both parse the Reference Design result files separately.

## 2. What to build

- One **patch step** that applies every fix to a Reference Design checkout and **fails loudly** when any expected match is missing, naming the fix and the file.
- Optional (decided in grilling): one shared reader for the Reference Design result files, used by both compare scripts. It could live in the reader from #117.

## 3. Out of scope

- Changing what any patch does.
- Touching `../ext/` (read-only).

## 4. Tests

- One test: patching a pinned copy of the reference source gives the same output as the four patchers today.
- One test per fix: a source with the match removed makes the step fail with that fix's name.

## 5. Closure criteria

- [ ] Patched output byte-identical to today's for the pinned reference source.
- [ ] Missing-match failure tested for every fix.
- [ ] Host pytest passes.
