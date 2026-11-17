# Issue #62: full test suite (2026-11-16)

**Result: 583 passed, 0 failed, 0 skipped**, in 937.7 s (15 min 38 s). Raw output: [`full_test_suite_2026-11-16.txt`](full_test_suite_2026-11-16.txt).

- **What ran:** every test under `tests/pure_xdsl_blis_rvv/` (40 files), on the #62 branch at commit `dc984ccb`, with the project venv: `.venv/bin/python -m pytest -q tests/pure_xdsl_blis_rvv`.
- **Board:** `test_pack_correctness.py` and `test_perf_counters.py` compile and check on the board through `fej`. They skip when the board can't be reached; none skipped, so both ran on the board. They are correctness checks; nothing is timed. No board series ran meanwhile.
- **Covers #62's new code:** `test_run_hector_gemm.py` (6 tests: the runner against a stand-in board) and `test_hector_gemm_report.py` (3: the report and figure). Also the HectorYolo dump in `test_run_yolo_sweep.py` (11).

## Side effects, cleaned up

- The suite rewrote three tracked files under `tests/pure_xdsl_blis_rvv/ref_baseline/generated/` (`run_plan.tsv`, `xdsl_cases_16x14.dat`, `xdsl_cases_16x16.dat`). These are known side effects of the `--dry-run` tests. They were restored with `git checkout`.
- A test from before #62 left `tests/pure_xdsl_blis_rvv/logs/issue30_direct_check_2026-11-16_120124.txt` in the real logs folder. The same happened in the earlier session's run (`…_110202.txt`). Both are left untracked; this is outside #62.
