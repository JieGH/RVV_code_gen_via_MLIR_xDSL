# Organize tests/pure_xdsl_blis_rvv by feature: past experiments, configs and tests out of the top level

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/127

**Role / Label:** `needs-triage`. Parent issue. Opened 2026-11-20 at the user's request.
**Status:** closed 2026-11-21. All four sub-issues done; behaviour unchanged (snapshot and test IDs); correctness pass on jbpi from the new layout all PASS.
**Branch:** `organize-test-folder-by-feature`, cut from `board-access-module` (#116 is closed but not yet merged into `pure-xdsl`, and it edits every run script; cutting from it avoids conflicts). Sub-issues are worked on this branch in order, one commit series each.
**Scope:** `tests/pure_xdsl_blis_rvv/` only, plus the docs that name its paths (`docs/agents/execution.md`, `docs/metrics.md`, `ref_baseline/README.md`).

---

## 1. Why

The project folder has 152 files at its top level (2026-11-20):

| Kind | Count | Of which tied to one past issue |
|------|------:|------:|
| `config*.json` | 45 | 22 named `config_issueNN_*`, more named by topic (`config_dim_sweep_*`, `config_b1b5_*`) |
| `test_*.py` | 51 | 16 (`test_run_issueNN.py`, `test_issueNN_*.py`) |
| other `.py` | 21 | 5 (`issueNN_report.py`, `issue53_packer_compare.py`, ...) |
| `.sh` | 27 | 14 (`run_issueNN.sh`, `probe_issue95.sh`, ...) |
| `.md` | 6 | 6 old notes, cited by no file |

The tools that make up the design (the Project Generator, the bench, the runners, the report readers) are hidden among one-off experiment files. A reader cannot tell which config checks which feature, or which test guards it.

## 2. What the design does (the features the layout and the baseline must cover)

| Feature | Entry point | Configs today | Host tests today |
|---------|-------------|---------------|------------------|
| **Kernel generation**: one GEMM project (packers, microkernel, fringe path, driver) per MRxNR tile, from xDSL | `generate_pure_blis_project.py`, `compile_macro_bench.sh` | `config.json` | `test_config.py`, `test_cache_config.py`, `test_pad_ldb.py`, `test_pad_zeroing.py`, `test_ref_protocol.py` |
| **Correctness pass**: scalar oracle against our C | `compile_macro_bench.sh --config ... --tmin 0.2` | `config_correctness.json`, `config_paper_correctness.json`, `config_yolo_correctness*.json`, `config_yolo53_correctness_*.json` | `test_correctness_pass.py`, `test_pack_correctness.py` (board) |
| **Tile Search** (sweep the MRxNR dimension, pick the best per shape) | `run_yolo_sweep.sh`, `run_paper_sweep.sh`, `search_report.py` | `config_yolo_search.json`, `config_paper_search.json`, `config_yolo53_search_*.json`, `config_dim_sweep_*.json` | `test_tile_search.py`, `test_run_yolo_sweep.py`, `test_run_paper_sweep.py` |
| **Data types**: fp32, f16f16f32 (mixed, scalar pack only, #112), int8 / i8i8i32 | datatype column of a config | `config_issue112_mixed_correctness.json` | `test_handover_ukernels.py` only; no host test generates an f16f16f32 or int8 project |
| **Packing methods**: RVV pack, scalar pack (`--scalar-pack`), B-pack variants (`--pack-b`), zeroed pack lines | `compile_macro_bench.sh` flags | `config_pack_swap*.json` | `test_pack_b_variants.py`, `test_pack_b_switch.py`, `test_pack_b_only.py`, `test_pack_dims_table.py`, `test_zero_pack_lines.py` |
| **Threads / YOLO** (8-thread GEMM, Héctor's YOLO layers) | `run_parallel_gemm.sh`, `run_hector_gemm.sh` | `config_parallel_gemm.json`, `config_yolo53_*.json`, `config_yolo_n1.json` | `test_parallel_gemm.py`, `test_parallel_gemm_report.py`, `test_run_hector_gemm.py`, `test_hector_gemm_report.py` |
| **Reference Design** comparison | `run_head_to_head.sh`, `ref_baseline/` | `ref_baseline/cases*.tsv` | `test_compare_ref_*.py`, `test_ref_pack_fix.py`, `test_ref_per_rep.py`, `test_run_ref_sweep.py`, `test_fringe_overhead.py` |
| **Measurement and house rules**: run protocol, per-rep logs, board access, provenance | `board_pin.sh`, `reps_report.py`, `rawlog.py` | none | `test_board_access.py`, `test_board_pin.py`, `test_house_rules.py`, `test_reps_report.py`, `test_rawlog.py`, `test_perf_counters.py`, `test_summarize_run.py` |

## 3. Target layout

```
tests/pure_xdsl_blis_rvv/
  README.md                     map: feature -> entry point -> configs -> tests (new)
  config.json                   default config (stays: every doc and command names it)
  generate_pure_blis_project.py, pack_builder.py, config.py, cache_config.py, ...   the generator (stays)
  compile_macro_bench.sh, board_pin.sh, run_*.sh (general runners), *_report.py (general readers)  (stay)
  configs/
    correctness/                one config per correctness pass
    tile_search/                multi-tile sweeps
    benchmark/                  fixed-tile timing configs
  experiments/
    issueNN/                    one folder per past issue: its run script, configs, report script, tests
  tests/                        every host test that is not tied to one experiment
  notes/                        the six old notes (.md)
  templates/ datasets/ ref_baseline/ modelLevel/ pack_only/ logs/ pj_gen/   (unchanged)
```

Rule for each file:

- Named for one issue (`run_issueNN*.sh`, `config_issueNN_*.json`, `issueNN_*.py`, `test_run_issueNN.py`, `test_issueNN_*.py`, `probe_issueNN.sh`, `issueNN/`) goes to `experiments/issueNN/`. One exception: `config_issue112_mixed_correctness.json` goes to `configs/correctness/`, because it is the only correctness check of the f16f16f32 data type.
- Any other config goes to `configs/<purpose>/`, `config.json` excepted.
- Any other `test_*.py` goes to `tests/`.
- Generator, bench, general runners and general report readers stay at the top level. They are the design; docs, memory and the user's commands name them.

## 4. Rules for every sub-issue

- **Behaviour stays the same.** Generated C, log names, log columns and `--dry-run` plans stay byte-identical. The only allowed change in a `--dry-run` is a config path in the overview's `Config` line (e.g. `configs/correctness/config_correctness.json`). Proved by the baseline snapshot of #128 taken before any move and compared after each sub-issue.
- **Same test results.** The full host suite passes the same test IDs as the baseline (test IDs change only by the file's new folder).
- **Closed issues keep their scripts runnable** from the new folder. Their old logs and issue files cite the old paths; those are not rewritten. `experiments/README.md` lists old path -> new path.
- **`git mv` only**, so `git log --follow` keeps each file's history.
- **Untouched:** `logs/` (run folders are cited by path; `test_house_rules.py` checks them), `pj_gen/`, `ref_baseline/`, `templates/`, `datasets/`, `modelLevel/`, `pack_only/`.
- **No tests during a board series.** Run `.venv/bin/python -m pytest` from `tests/pure_xdsl_blis_rvv` only when no series runs; save and restore `ref_baseline/generated/` around it.
- **No timing runs.** Nothing here changes the driver or the generated C.
- **Pre-commit:** `--no-verify`.

## 5. Sub-issues

| Order | Sub-issue | Depends on |
|-------|-----------|------------|
| 1 | Feature baseline: snapshot tool and feature smoke tests (#128) | none |
| 2 | Move past-issue experiments into `experiments/issueNN/` (#129) | 1 |
| 3 | Move the remaining configs into `configs/<purpose>/` (#130) | 1 |
| 4 | Move general tests into `tests/`, notes into `notes/`; README and docs (#131) | 2, 3 |

## 6. Out of scope

- `logs/` cleanup (835 MB, 1027 entries). Report and issue files cite run folders by path.
- Splitting the general runners and readers into `runners/` and `reports/`. Possible later; decide after this lands.
- Replacing per-issue run scripts with plan files: #119.

## 7. Closure

- [x] All four sub-issues closed.
- [x] Snapshot after the last sub-issue matches the baseline (§4).
- [x] Full host suite: same passes as the baseline.
- [x] One correctness pass on jbpi from the new layout (`configs/correctness/config_correctness.json`, busy board allowed), all rows PASS.
- [ ] Merged: branch -> `pure-xdsl` -> `main`.

## 8. Results (2026-11-21)

| Step | Result |
|------|--------|
| #128 baseline | snapshot of 45 configs (generated C hashes + `--dry-run` overview); 674 host tests passed on the unmoved tree; 7 feature smoke tests added |
| #129 experiments | 57 files into `experiments/issueNN/` (20 issues); compare clean; 681 passed |
| #130 configs | 22 configs into `configs/{correctness,tile_search,benchmark}/`; compare clean; 681 passed |
| #131 tests, notes, README | 35 tests into `tests/`, 6 notes into `notes/`, folder `README.md`; compare clean; 681 passed |

- 681 = the baseline's 674, same test IDs (folder part dropped), + the 7 smoke tests. Each full run about 16.6 min, `.venv` Python, `ref_baseline/generated/` saved and restored around it.
- Top level: 152 files -> 31 files + 11 folders.
- **Board check:** `compile_macro_bench.sh --jump ... --riscv-ip jbpi --no-nc-overrides --cpu 4 --config configs/correctness/config_correctness.json --tmin 0.2`, uninstrumented build, reference protocol, core 4, board idle (no other user's process, read just before). Run folder `tests/pure_xdsl_blis_rvv/logs/run_2026-11-21_021027/`: **9 of 9 rows PASS**. Correctness only, no timing number taken from it.
- Moves for #129-#131 were done by Sonnet subagents and each checked here by the snapshot compare and the full suite before its commit.
- Visible change for users: runners that print the config path they pass to the bench now print the new path (`configs/...`, `experiments/issueNN/...`). Old logs and issue files keep the old paths; `experiments/README.md` maps them.
