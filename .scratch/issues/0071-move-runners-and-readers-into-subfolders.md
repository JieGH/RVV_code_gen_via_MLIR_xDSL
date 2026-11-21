# Move runners and readers out of the project top level into runners/ and readers/

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/137

**Role / Label:** `needs-triage`. Opened 2026-11-21 at the user's request, after #134 (user guide) and #127 (organize by feature).
**Status:** closed 2026-11-21. Moves clean (681 passed, snapshot compare no difference); feature baseline on jbpi 15/15 PASS; READMEs rewritten.
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

Added during the issue, at the user's request (2026-11-21):

- **Feature baseline** `runners/run_feature_baseline.sh`: one board correctness run per user-facing option (data type, packer, `--pack-b` variant, `--zero-pack-lines`, `--pad-ldb`, pinned and search tiles, `--threads 8`), each against the scalar oracle; every row must PASS. New configs `configs/correctness/config_feature_{dot,int8,search}.json`. Host test `tests/test_run_feature_baseline.py`.
- **Layout and scripts test** `tests/test_layout_and_scripts.py`: top level holds only the main workflow; no path in code or docs points at a missing file; every runner and reader starts from its folder; each folder README lists its scripts.
- **Test regime** in the folder README: host tests, snapshot compare, feature baseline.
- **READMEs rewritten** to the essentials (what, commands, options, where results land): root, project folder, `runners/`, `readers/`, `configs/`, `experiments/`, `ref_baseline/` (opening only; its caveats stay).

## 3. Out of scope

- Any change to what a script does.
- Moving `board_pin.sh`: `compile_macro_bench.sh` and every runner source it.

## 4. Tests

- Snapshot compare against the #128 baseline: generated C and `--dry-run` overview identical.
- Full host suite passes, same test count.
- Each moved runner: `--help` or `--dry-run` from the repo root. Each reader: run on a committed log folder.
- One correctness pass on the board through `compile_macro_bench.sh` (it calls `readers/summarize_run.py`).

## 5. Closure criteria

- [x] Top level holds only the files in §2.
- [x] Snapshot compare clean; full host suite passes.
- [x] Every moved script runs from its new place.
- [x] Feature baseline passes on the board.
- [ ] READMEs reviewed by the user (not blocking).

## 6. Results (2026-11-21)

- **Moves:** 12 scripts to `runners/`, 11 to `readers/`; top level from 31 files to 9. Paths fixed in 59 files.
- **Full host suite:** 681 passed, 16.7 min (same count as #131). Run inside the snapshot below.
- **Snapshot compare** against `.scratch/reports/issue-128/after131`: no difference (generated C for 45 configs, every `compile_macro_bench.sh --dry-run` plan, passing test IDs). Files: `.scratch/reports/issue-137/after137/`, `compare_after137.txt`.
- **Each script from its new place:** every runner `--help`; seven runners `--dry-run` (`run_hector_gemm.sh` needs Héctor's clone at `b292c92`: covered by its stubbed test); readers on `logs/run_2026-11-21_021027` (`summarize_run`, `reps_report`, `search_report`), `shape_breakdown` on a dataset, `pack_dims_table`, `slow_state_survey` over all logs; `gen_handover_ukernels.py` wrote a full hand-over set.
- **New tests:** `test_layout_and_scripts.py` and `test_run_feature_baseline.py`, 57 passed with `test_house_rules.py`.
- **Feature baseline on jbpi** (idle board, core 4, `--tmin 0.2`, correctness only: no number from it is quoted), commit `8e558de7`, `logs/feature_baseline_2026-11-21_183802.tsv`:

| Case | Run | Rows PASS |
|---|---|---|
| fp32 AXPY, RVV pack | `run_2026-11-21_184415` | 9/9 |
| fp32 DOT | `run_2026-11-21_184616` | 3/3 |
| f16f16f32, scalar pack | `run_2026-11-21_185514` | 12/12 |
| int8 | `run_2026-11-21_185713` | 3/3 |
| fp32 scalar pack | `run_2026-11-21_190427` | 9/9 |
| `--pack-b` row-intr, col-g1, col-g2, col-g4, col-gall, col-g1-m4 | `run_2026-11-21_191043` .. `_194208` | 9/9 each |
| `--zero-pack-lines` | `run_2026-11-21_194824` | 9/9 |
| `--pad-ldb 16` | `run_2026-11-21_195439` | 9/9 |
| search mode (every tile x shape) | `run_2026-11-21_201243` | 1056/1056 |
| `--threads 8` | `run_2026-11-21_202309` | 9/9 |

- **Runner bug found and fixed:** the first version read the 6th column of the merged raw log as `Status`, but the merged log puts `Project` first, so it read `Variant` and counted the header: every case printed FAIL in the series console log (`logs/feature_baseline_issue137.out`, gitignored). The run folders were fine. Fixed to find `Status` by its header name; the `.tsv` was recomputed from the run folders; two tests pin it (a made-up log with a FAIL row, and the real `run_2026-11-21_184415`).
- **Per-rep log too big for GitHub:** the search-mode run wrote a 120.7 MB `reps_*.txt.gz` (over the 100 MB limit), so it is gitignored and kept on this host only; the verdicts come from `all_raw_*.txt`, which is committed. The runner now passes `--no-save-reps` (correctness runs need no per-rep timings).
- **Not done:** the user's crontab still has September entries pointing at the old `cron_bench_run.sh` path. Removed in #140.
