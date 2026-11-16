# Full test suite, issue #56 branch (2026-11-16)

Branch `make-the-design-parallel-for-the-gemm-test-and-test-it-on-one-specific-program`, run before the board series, from `tests/pure_xdsl_blis_rvv/`: `pytest -q -p no:cacheprovider`.

**604 passed, 0 failed, 0 skipped** in 942.91 s (15 min 43 s). The board-side test files (`test_pack_correctness.py`, `test_perf_counters.py`) ran on the board.

After the code review: `--no-ref-protocol` refused with `--threads`, and two small edits. The three touched test files were run again: `test_parallel_gemm.py`, `test_parallel_gemm_report.py`, `test_pack_b_switch.py`: 66 passed.

Raw tail: [`full_test_suite_2026-11-16.txt`](full_test_suite_2026-11-16.txt).
