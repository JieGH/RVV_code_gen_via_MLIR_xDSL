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
- `ref_baseline/compare_ref_baseline.py:56`: `DEFAULT_RAW_NAME = "all_raw_fp32_AXPY_FP32_Pack.txt"`, with no `_noinstr`. The default build is uninstrumented, so the default name misses it; callers must pass `--raw-name`.
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

## 5. Closure criteria

- [ ] No `_refproto` / `_noinstr` / `_scalarpack` / `_no_nc_override` string literal in report scripts outside the reader (grep).
- [ ] `compare_ref_baseline.py` finds the default (uninstrumented) log without `--raw-name`.
- [ ] Reports' output on committed logs byte-identical before and after (diff saved in this issue).
- [ ] Host pytest passes.
