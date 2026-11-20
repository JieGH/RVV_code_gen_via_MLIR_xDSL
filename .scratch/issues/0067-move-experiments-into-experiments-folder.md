# Move past-issue experiments into experiments/issueNN/

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/129

**Parent:** #127, organize `tests/pure_xdsl_blis_rvv` by feature.
**Role / Label:** `needs-triage`. Spec, 2026-11-20.
**Status:** open. Blocked by #128 (baseline first).
**Branch:** `organize-test-folder-by-feature`.

---

## 1. Problem

Each experiment since #46 left a run script, configs, a report script and tests at the top level. 14 issues, 50 files.

## 2. What to build

`git mv` each issue's files into `experiments/issueNN/`:

| Issue | Files |
|-------|-------|
| 34, 35, 36, 37, 41, 43 | their `config_issueNN_*.json` only |
| 46 | `run_issue46.sh`, `issue46_report.py`, `config_issue46_ref_b3.json`, `test_run_issue46.py`, `test_issue46_report.py` |
| 48 | `run_issue48.sh`, `issue48_report.py`, `config_issue48_sweep*.json`, `test_run_issue48.py`, `test_issue48_report.py` |
| 49 | `run_issue49.sh`, `issue49_layout_table.py`, `config_issue49_sweep.json`, `test_run_issue49.py`, `test_issue49_layout_table.py` |
| 50 | `run_issue50.sh`, `config_issue50_sweep.json`, `test_run_issue50.py` |
| 51 | `run_issue51.sh`, `config_issue51_*.json` (5), `test_run_issue51.py` |
| 53 | `issue53_packer_compare.py`, `test_issue53_packer_compare.py` |
| 54 | `run_issue54_ref.sh` |
| 55 | `run_issue55.sh`, `issue55_report.py`, `config_issue55_tiles.json`, `test_run_issue55.py`, `test_issue55_report.py` |
| 63 | `run_issue63.sh`, `test_run_issue63.py` |
| 83 | `run_issue83.sh`, `run_issue83_pack_split.sh`, `config_issue83_pack_split.json`, `test_run_issue83.py` |
| 85 | `run_issue85.sh`, `config_issue85_*.json`, `test_run_issue85.py` |
| 94 | `run_issue94.sh`, `config_issue94_correctness.json`, `test_run_issue94.py` |
| 95 | `probe_issue95.sh`, `issue95/pack_order.c` |
| 112 | `run_issue112_handover.sh` (its config goes to `configs/correctness/`, #130) |

Then fix every path that named them: `HERE`/`ROOT` lines in the moved files, callers among the general scripts (`pack_dims_table.py` defaults to `config_issue48_sweep.json`; `run_issue83.sh` calls `run_issue63.sh`; ...), tests that name a moved file, and `docs/agents/execution.md`. Write `experiments/README.md`: one line per issue, old path -> new path, and how to run a script from its new place.

## 3. Out of scope

- Editing what a script does. Only paths change.
- Rewriting old paths inside `logs/` or `.scratch/issues/`.

## 4. Tests

- Each moved test passes from its new folder.
- Snapshot compare against the baseline: no difference.

## 5. Closure criteria

- [ ] No `*issue[0-9]*` file left at the top level.
- [ ] Snapshot compare clean; full host suite passes.
