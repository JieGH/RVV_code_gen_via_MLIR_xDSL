# Reference Design scripts fail on a fresh clone: they need `../ext/`, which the README never mentions

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/149

**Role / Label:** `needs-triage`. Found 2026-11-22 during a newcomer walk-through of the README (fresh clone, no board).
**Status:** fixed on branch `reference-scripts-fail-without-sibling-ext` (`450b3813`), not merged yet. Both closure criteria met (section 7).
**Target:** host only (no board used).
**Primary Sources:** `tests/pure_xdsl_blis_rvv/ref_baseline/run_ref_baseline.sh`, `ref_baseline/run_direct_check.sh`, `runners/run_head_to_head.sh`, `tests/test_board_pin.py`, `tests/test_layout_and_scripts.py`, `README.md`

---

## 1. Symptom

In a fresh clone of `5dae0f10` (cloned to a temp folder, no `../ext/` beside it), the README's host test command fails 3 tests:

```
cd tests/pure_xdsl_blis_rvv && ../../.venv/bin/python -m pytest
3 failed, 803 passed, 3 skipped in 976.19s (0:16:16)

FAILED tests/test_board_pin.py::test_every_script_refuses_core_0_before_anything_runs[run_direct_check.sh]
FAILED tests/test_board_pin.py::test_a_wrapper_passes_allow_core_0_on_to_the_scripts_it_runs[run_head_to_head.sh]
FAILED tests/test_layout_and_scripts.py::test_runner_dry_run[run_head_to_head.sh]
```

Error text:

```
.../fresh/../ext/RVV_ukernels_benchmark: No such file or directory
```

In the normal checkout, `../ext/` exists, so these tests pass there.

## 2. Root cause

The Reference Design source is not in the repo. Two scripts find it by a fixed path outside the repo:

- `ref_baseline/run_ref_baseline.sh:107`: `REF_SRC="$(cd "${REPO_ROOT}/../ext/RVV_ukernels_benchmark" && pwd)"`
- `ref_baseline/run_direct_check.sh:36`: same line.

`runners/run_head_to_head.sh:77,85` calls `run_ref_baseline.sh`, so it fails too. The lookup runs before argument checks, so even `--dry-run` and the core-0 refusal fail.

## 3. Where the docs fall short

- `README.md` lists `run_head_to_head.sh` under "Other workflows" ("Compare with the Reference Design on the same core") with no note that it needs `../ext/`.
- The README test command ("Host tests, about 17 minutes") does not say that 3 tests need `../ext/`.
- The only mention is one line in `docs/user_guide.md:149`. It does not say where a newcomer gets the tree.

## 4. Not done

- No board run.
- No check of `runners/run_ref_sweep.sh` beyond its comment (line 7 names the same tree).
- No decision on where the Reference Design should come from (submodule, download script, or documented path).

## 5. Options

- **A.** README: add a "Needs `../ext/RVV_ukernels_benchmark`" note to the head-to-head row and the host-test command, plus where to get it.
- **B.** Tests: skip the three tests with a clear reason when `../ext/RVV_ukernels_benchmark` is missing.
- **C.** Scripts: when the tree is missing, print one clear line ("Reference Design not found at ...; see README") instead of the raw `cd` error.

## 6. Closure criteria

- [x] A fresh clone without `../ext/` runs the host tests with 0 failures (skips allowed).
- [x] The README says which workflows need `../ext/` and how to get it.

## 7. Fix (2026-11-22)

Options A, B and C, plus one change the options did not list:

- **C, scripts.** `run_ref_baseline.sh`, `run_direct_check.sh` and `run_head_to_head.sh` print one line when the tree is missing: `[FAIL] Reference Design not found at <path>; see README.md (Reference Design)`.
  - `run_direct_check.sh` now looks the tree up after the core-0 refusal, so the refusal still comes first. This fixes `test_every_script_refuses_core_0_before_anything_runs[run_direct_check.sh]` without a skip.
  - `run_head_to_head.sh` checks before our side runs. Before, a real run would benchmark our side on the board and fail only at the reference step.
- **B, tests.** The two head-to-head `--dry-run` tests skip with a `needs_ref` marker when `../ext/RVV_ukernels_benchmark` is missing.
- **A, docs.** The README has a new "Reference Design" section: clone `https://github.com/adcastel/RVV_ukernels_benchmark.git` into `../ext/` and check out `eb84ded`, the commit in our `../ext/`. The head-to-head row and the host-test line point to it, and so does `docs/user_guide.md`.

**Checks (host only, no board):**

- Fresh `git clone` of `450b3813`, no `../ext/`: `804 passed, 5 skipped in 976.41s`. The skips are the 2 new head-to-head ones plus 3 older ones (`test_ref_pack_fix.py` x2, `test_ref_per_rep.py`) that already skipped without the tree.
- Normal checkout (with `../ext/`): `test_board_pin.py` + `test_layout_and_scripts.py`, 161 passed.

**Not done:** no test checks the new `[FAIL]` line; it was checked by hand. The path and message are written out in all three scripts, not shared.
