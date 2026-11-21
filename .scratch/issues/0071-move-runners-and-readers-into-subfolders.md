# Move runners and readers out of the project top level into runners/ and readers/

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/137

**Role / Label:** `needs-triage`. Opened 2026-11-21 at the user's request, after #134 (user guide) and #127 (organize by feature).
**Status:** open.
**Branch:** `move-runners-and-readers-into-subfolders`, cut from `pure-xdsl`.

---

## 1. Problem

The top level of `tests/pure_xdsl_blis_rvv/` still holds 31 files. Most are side scripts: eight `run_*.sh` runners, the overnight cron helpers, and eleven Python readers of run logs. A new developer sees them first, though the core workflow is only: generate a kernel project (data type, packing method, tile) and run it with `compile_macro_bench.sh`.

## 2. What to build

Top level keeps only the main workflow:

```
generate_pure_blis_project.py  pack_builder.py  config.py  cache_config.py   the Project Generator
compile_macro_bench.sh  board_pin.sh                                          generate, build, run on the board
config.json  README.md  __init__.py
```

- `runners/`: `run_*.sh`, `cron_bench_run.sh`, `gather_sweep_report.sh`, `hector_yolo_dump.sh`, `gen_handover_ukernels.py`.
- `readers/`: `compare_pack_swap.py`, `hector_gemm_report.py`, `parallel_gemm_report.py`, `rawlog.py`, `report_scalar_sweep.py`, `reps_report.py`, `search_report.py`, `shape_breakdown.py`, `slow_state_survey.py`, `summarize_run.py`, `pack_dims_table.py`.
- `git mv` each file; fix every path that names it: `HERE`/`sys.path` in the moved scripts, `compile_macro_bench.sh`, `ref_baseline/`, `experiments/`, `tests/`, configs' comments, `README.md`s, `docs/`.
- Logs and closed issue files stay as they are (history).

## 3. Out of scope

- Any change to what a script does.
- Moving `board_pin.sh`: `compile_macro_bench.sh` and every runner source it.

## 4. Tests

- Snapshot compare against the #128 baseline: generated C and `--dry-run` overview identical.
- Full host suite passes, same test count.
- Each moved runner: `--help` or `--dry-run` from the repo root. Each reader: run on a committed log folder.
- One correctness pass on the board through `compile_macro_bench.sh` (it calls `readers/summarize_run.py`).

## 5. Closure criteria

- [ ] Top level holds only the files in §2.
- [ ] Snapshot compare clean; full host suite passes.
- [ ] Every moved script runs from its new place.
