# Raw-log reader: one Python module finds and reads the driver's logs for every report

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/117

**Parent:** #115, codebase architecture optimization.
**Role / Label:** `needs-triage`. Spec, 2026-11-20. Strength: **Strong**.
**Status:** open. Not started. Host-only, no board runs.
**Branch:** cut from `pure-xdsl` when work starts, named after this title.

---

## 1. Problem

The driver names its raw log from six optional parts (`templates/driver_benchmark.cpp:711`):

```
_refproto  _scalarpack  _zeroline  _padldb<N>  _packb-<name>  _no_nc_override
```

Report scripts spell that name again by hand, each knowing only some parts:

- `compare_pack_swap.py:34` (`PROTOCOL = "_noinstr_refproto"`)
- `report_scalar_sweep.py:39-40` (two fixed globs)
- `ref_baseline/compare_ref_sweep.py:9`
- `ref_baseline/compare_ref_baseline.py:56`: `DEFAULT_RAW_NAME = "all_raw_fp32_AXPY_FP32_Pack.txt"`, with no `_noinstr`. *(Correction 2026-11-20: this is deliberate. Issue #27's test `test_a_ref_protocol_run_is_read_by_name_and_never_by_the_default_name` keeps a Reference Protocol log out of a default run's numbers. It is not a bug.)*
- `parallel_gemm_report.py:80`, `summarize_run.py:180,187`

Rows are read 7 ways: `csv.DictReader` in 5 files, `split(",")` in `parallel_gemm_report.py`, `split("\t")` in `hector_gemm_report.py`; `runs.tsv` hand-read in 4 files (`issue46_report.py:61`, `issue48_report.py:87`, `parallel_gemm_report.py:40`, `hector_gemm_report.py:40`).

**Cost today:** a new log part or a renamed column breaks reports one by one. A report can pick the wrong variant's log without error (a past cause of mixed numbers; see the reference-comparison trap in memory and #13).

## 2. What to build

One Python module, the **Raw-log reader**, that every report uses:

1. **Find a run's raw log by variant** (protocol, packer, nc overrides, ...). Unknown or missing variant: error naming what was found.
2. **Read its rows** with typed columns.
3. **Read a series' `runs.tsv` and `plan.txt`.**

The log-name rule lives in one place, next to a comment that points to `driver_benchmark.cpp:711`. Exact names and arguments are settled in grilling.

## 3. Out of scope

- Changing the log format or file names (#118 may change how the shell learns the name; the name stays).
- Rewriting finished reports of closed issues beyond swapping their loader.
- The `.scratch/reports/` figure scripts (separate, per-report).

## 4. Tests

- Fixture logs (small, real rows copied from committed logs) for each variant the driver can write.
- One test per rule: name from variant, variant from name, wrong-variant error, row types.
- Each switched report: its existing test output unchanged.

## 5. Decisions (2026-11-20, by the agent, no grilling)

- **Module:** `tests/pure_xdsl_blis_rvv/rawlog.py`. Functions: `raw_glob(**variant)`, `find_raw`, `require_raw` (stops and lists what the run holds), `any_raw`, `resolve_raw`, `read_rows`, `read_tsv`. Parts are in the driver's order (`_noinstr _refproto _scalarpack _zeroline _padldb<N> _packb-<name> _no_nc_override`); a test reads `driver_benchmark.cpp` and fails if the driver's order differs.
- **`compare_ref_baseline.py` keeps its default.** A first version fell back to the uninstrumented Reference Protocol log; test #27 failed, because that is exactly what the guard forbids. Reverted: the default reads only the old instrumented name; a Reference Protocol log is read only with `--raw-name`. The closure line "finds the default (uninstrumented) log without `--raw-name`" is therefore **dropped as wrong**, not met.
- **Switched to the reader:** `compare_pack_swap.py`, `report_scalar_sweep.py` (`VECTOR_GLOB` / `SCALAR_GLOB` still exported for `compare_ref_sweep.py`), `compare_ref_baseline.py`, `parallel_gemm_report.py`, `hector_gemm_report.py`, `summarize_run.py`.
- **Not switched:** `issue46_report.py`, `issue48_report.py` (closed experiments, `runs.tsv` read into different tuple shapes), `issue55_report.py`, `reps_report.py` (reads `reps_*.txt`, a different file family; `reps_report.py:185` tests `"_scalarpack" in f.name`).

## 6. Result

- Report output on committed logs, before and after: identical for 6 reports (table in `.scratch/reports/issue-117/README.md`).
- New tests: `test_rawlog.py`, 9 passed.

## 7. Closure criteria

- [x] No `_refproto` / `_noinstr` / `_scalarpack` / `_no_nc_override` string in report code outside the reader (grep, `*.py` minus tests). Left only in two docstrings and `compare_ref_sweep.py`'s docstring, which describe the file names to a reader. `reps_report.py` and `slow_state_survey.py` now use `rawlog.*_PART`.
- [x] ~~`compare_ref_baseline.py` finds the default log without `--raw-name`~~ Dropped: it contradicts the #27 guard (section 5). Not a defect.
- [x] Reports' output on committed logs identical before and after (`.scratch/reports/issue-117/README.md`).
- [x] Host pytest passes: 567 passed in the conda Python plus 147 passed in `.venv` for the ten files that need `xdsl`. Earlier "failures" in `test_parallel_gemm.py` and four other files were the conda Python lacking `xdsl` (same on the pre-change tree), not code.
