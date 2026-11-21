# Pure xDSL BLIS for RVV: the Project Generator and its bench

This folder generates GEMM projects for RISC-V Vector (packers, microkernel, fringe path, driver) from xDSL, runs them on the board, and reads the results. Terms (Microkernel, Candidate Tile, Tile Search, Reference Design, ...) are defined in the repo's `CONTEXT.md`. How to run on the board, and the house rules for a quoted number: `docs/agents/execution.md`.

## Layout

```
config.json                     default config of compile_macro_bench.sh
generate_pure_blis_project.py   the Project Generator (with pack_builder.py, config.py, cache_config.py)
compile_macro_bench.sh          generate, build and run one config on the board
board_pin.sh                    board access: ssh, pinning, other users, idle gate, series names
run_*.sh                        general runners (Tile Search, threads, packing, Reference Design)
*_report.py, reps_report.py,    readers of run folders and logs
  rawlog.py, summarize_run.py
configs/                        correctness/, tile_search/, benchmark/  (configs/README.md)
experiments/issueNN/            one folder per past issue: its run script, configs, report, tests
tests/                          host tests (pytest)
templates/  datasets/           driver, Makefile and headers copied into every project; GEMM shapes
ref_baseline/                   the Reference Design, staged and compared
logs/                           run folders, committed; cited by path from issues and reports
pj_gen/                         generator output, gitignored, rebuilt on every run
notes/                          old notes, kept for history
modelLevel/  pack_only/         analytical cache model; pack-only bench
```

## Features

| Feature | Entry point | Configs | Host tests (`tests/` unless noted) |
|---------|-------------|---------|------------------|
| **Kernel generation**: one project per MRxNR tile | `generate_pure_blis_project.py`, `compile_macro_bench.sh` | `config.json` | `test_features.py`, `test_config.py`, `test_cache_config.py`, `test_pad_ldb.py`, `test_pad_zeroing.py`, `test_ref_protocol.py` |
| **Correctness pass**: scalar oracle against our C | `compile_macro_bench.sh --config configs/correctness/<x>.json --tmin 0.2` | `configs/correctness/` | `test_correctness_pass.py`, `test_pack_correctness.py` (compiles on the board) |
| **Tile Search**: sweep MRxNR, best tile per shape | `run_yolo_sweep.sh`, `run_paper_sweep.sh`, `search_report.py` | `configs/tile_search/` | `test_tile_search.py`, `test_run_yolo_sweep.py`, `test_run_paper_sweep.py` |
| **Data types**: fp32, f16f16f32 (scalar pack only, #112), int8 / i8i8i32 | datatype column of a config row | `configs/correctness/config_issue112_mixed_correctness.json` | `test_features.py`, `test_handover_ukernels.py` |
| **Packing methods**: RVV pack, `--scalar-pack`, `--pack-b <variant>`, `--zero-pack-lines` | `compile_macro_bench.sh` flags, `run_pack_swap.sh` | `configs/benchmark/config_pack_swap*.json` | `test_pack_b_variants.py`, `test_pack_b_switch.py`, `test_pack_b_only.py`, `test_pack_dims_table.py`, `test_zero_pack_lines.py` |
| **Threads and YOLO**: N-thread GEMM, Héctor's YOLO layers | `run_parallel_gemm.sh`, `run_hector_gemm.sh` | `configs/benchmark/config_parallel_gemm.json`, `config_yolo53_*.json`, `config_yolo_n1.json` | `test_parallel_gemm.py`, `test_parallel_gemm_report.py`, `test_run_hector_gemm.py`, `test_hector_gemm_report.py` |
| **Reference Design** comparison | `run_head_to_head.sh`, `run_ref_sweep.sh`, `ref_baseline/` | `ref_baseline/cases*.tsv` | `test_compare_ref_baseline.py`, `test_compare_ref_sweep.py`, `test_ref_pack_fix.py`, `test_ref_per_rep.py`, `test_run_ref_sweep.py`, `test_fringe_overhead.py` |
| **Measurement and house rules**: run protocol, per-rep logs, board access, provenance | `board_pin.sh`, `reps_report.py`, `rawlog.py` | none | `test_board_access.py`, `test_board_pin.py`, `test_house_rules.py`, `test_reps_report.py`, `test_rawlog.py`, `test_perf_counters.py`, `test_summarize_run.py` |
| **Past experiments** | `experiments/issueNN/run_*.sh` | `experiments/issueNN/` | `experiments/issueNN/test_*.py` |

A config row is `[datatype, method, [MR, NR], vlen, arch, [datasets], skip_ref_run, iterations]`. The 8th value (`iterations`) also makes the row **pinned**: one project for the first MRxNR only. Leave it out for search mode (one project with every tile up to the largest). See `test_features.py`.

## Host tests

```bash
cd tests/pure_xdsl_blis_rvv && ../../.venv/bin/python -m pytest
```

Use the repo's `.venv` (the conda Python lacks `xdsl`). The full suite takes about 17 minutes (the idle-gate tests sleep). It rewrites `pj_gen/` and `ref_baseline/generated/`, and some tests start runners with `--dry-run`: never run it while a board series runs.
