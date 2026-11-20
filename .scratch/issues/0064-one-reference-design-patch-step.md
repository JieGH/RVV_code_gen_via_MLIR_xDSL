# One Reference Design patch step

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/120

**Parent:** #115, codebase architecture optimization.
**Role / Label:** `needs-triage`. Spec, 2026-11-20. Strength: **Speculative** (low churn, small gain).
**Status:** open. **Analysed 2026-11-20, nothing built: the spec's premises are wrong** (section 6). Recommend `wontfix` + an ADR; waiting for the user.
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

## 6. Findings (2026-11-20, agent, before writing any code)

Each claim in section 1 was checked against the files:

| Spec said | The code says |
|---|---|
| Four modules patch the Reference Design's C by string match | Three do, each a different file for a different issue: `ref_pack_fix.py` patches `gemm_blis.c` (#54), `ref_per_rep.py` patches `test_gemm.c` and converts rep logs (#54/#55), `repair_ref_kernels.py` restores 16 cut bytes in six kernel files. **`fringe.py` patches nothing**: it computes our fringe-tile extra multiply-adds. |
| A missing match can pass silently and the run uses an unpatched file | `ref_pack_fix.patch_gemm_blis` raises `PatchError` unless the anchor occurs exactly once ("the reference changed"). `ref_per_rep._once` does the same for each of its five anchors. Both exit 1 with the message. `repair_ref_kernels.py` prints `[SKIP] <file>: unrecognised first line` on stderr and does not guess; that is on purpose (its docstring). |
| The interface is as small as the implementation (shallow) | Each takes a path (or a source string in tests) and owns its own anchors and replacement text. Merging them gives one module with the union of three unrelated anchor sets: a bigger interface, no behaviour moves (deletion test: nothing concentrates). |
| `compare_ref_baseline.py` and `compare_ref_sweep.py` parse the result CSVs separately | `compare_ref_sweep.py:42` imports `load_plan, load_reference, pin_of, pin_words, stamp_of` from `compare_ref_baseline.py`. One parser already. |

So there is no loud-failure gap and no duplicated parse to remove. The tests cover the anchors (`test_ref_pack_fix.py`, `test_ref_per_rep.py`, `test_fringe_overhead.py`).

**Recommendation:** close as `wontfix` and record in `docs/adr/` that the Reference Design patchers stay separate (one file each, each fails loudly on a changed anchor), so later architecture reviews skip them. Not done: the user decides, and the ADR is only written on a yes.
