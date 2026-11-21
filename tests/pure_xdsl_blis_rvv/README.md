# Pure xDSL BLIS for RVV: the Project Generator and its bench

This folder generates GEMM projects for RISC-V Vector (packers, microkernel, fringe path, driver) from xDSL, runs them on the board, and reads the results. How to do each operation, step by step: [`docs/user_guide.md`](../../docs/user_guide.md). Terms (Microkernel, Candidate Tile, Tile Search, Reference Design, ...) are defined in the repo's `CONTEXT.md`. How to run on the board, and the house rules for a quoted number: `docs/agents/execution.md`.

## Layout

```
generate_pure_blis_project.py   the Project Generator (with pack_builder.py, config.py, cache_config.py)
compile_macro_bench.sh          generate, build and run one config on the board: the main workflow
board_pin.sh                    board access: ssh, pinning, other users, idle gate, series names
config.json                     default config of compile_macro_bench.sh
runners/                        multi-run scripts: Tile Search, threads, pack swap, Reference Design sweeps,
                                  hand-over kernels, cron helpers (runners/README.md)
readers/                        read run folders and logs: per-run summary, per-rep stats, reports (readers/README.md)
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

## Quick start

Only the top level is needed to generate and run a kernel:

```bash
# from the repo root
(cd tests/pure_xdsl_blis_rvv && python3 generate_pure_blis_project.py --config config.json)   # generate only, host: writes pj_gen/
./tests/pure_xdsl_blis_rvv/compile_macro_bench.sh $CONN --no-nc-overrides --cpu 4 --tmin 10   # generate, build, run on the board
```

`$CONN` is the board connection (`--jump --riscv-ip jbpi ...`, set in the user guide §1).

The config row picks the data type, packing method and tile. Step-by-step: [`docs/user_guide.md`](../../docs/user_guide.md).

## Features

| Feature | Entry point | Configs | Host tests (`tests/` unless noted) |
|---------|-------------|---------|------------------|
| **Kernel generation**: one project per MRxNR tile | `generate_pure_blis_project.py`, `compile_macro_bench.sh` | `config.json` | `test_features.py`, `test_config.py`, `test_cache_config.py`, `test_pad_ldb.py`, `test_pad_zeroing.py`, `test_ref_protocol.py` |
| **Correctness pass**: scalar oracle against our C | `compile_macro_bench.sh --config configs/correctness/<x>.json --tmin 0.2` | `configs/correctness/` | `test_correctness_pass.py`, `test_pack_correctness.py` (compiles on the board) |
| **Tile Search**: sweep MRxNR, best tile per shape | `runners/run_yolo_sweep.sh`, `runners/run_paper_sweep.sh`, `readers/search_report.py` | `configs/tile_search/` | `test_tile_search.py`, `test_run_yolo_sweep.py`, `test_run_paper_sweep.py` |
| **Data types**: fp32, f16f16f32 (scalar pack only, #112), int8 / i8i8i32 | datatype column of a config row | `configs/correctness/config_issue112_mixed_correctness.json` | `test_features.py`, `test_handover_ukernels.py` |
| **Packing methods**: RVV pack, `--scalar-pack`, `--pack-b <variant>`, `--zero-pack-lines` | `compile_macro_bench.sh` flags, `runners/run_pack_swap.sh` | `configs/benchmark/config_pack_swap*.json` | `test_pack_b_variants.py`, `test_pack_b_switch.py`, `test_pack_b_only.py`, `test_pack_dims_table.py`, `test_zero_pack_lines.py` |
| **Threads and YOLO**: N-thread GEMM, Héctor's YOLO layers | `runners/run_parallel_gemm.sh`, `runners/run_hector_gemm.sh` | `configs/benchmark/config_parallel_gemm.json`, `config_yolo53_*.json`, `config_yolo_n1.json` | `test_parallel_gemm.py`, `test_parallel_gemm_report.py`, `test_run_hector_gemm.py`, `test_hector_gemm_report.py` |
| **Reference Design** comparison | `runners/run_head_to_head.sh`, `runners/run_ref_sweep.sh`, `ref_baseline/` | `ref_baseline/cases*.tsv` | `test_compare_ref_baseline.py`, `test_compare_ref_sweep.py`, `test_ref_pack_fix.py`, `test_ref_per_rep.py`, `test_run_ref_sweep.py`, `test_fringe_overhead.py` |
| **Measurement and house rules**: run protocol, per-rep logs, board access, provenance | `board_pin.sh`, `readers/reps_report.py`, `readers/rawlog.py` | none | `test_board_access.py`, `test_board_pin.py`, `test_house_rules.py`, `test_reps_report.py`, `test_rawlog.py`, `test_perf_counters.py`, `test_summarize_run.py`, `test_slow_state_survey.py`, `test_shape_breakdown.py` |
| **Past experiments** | `experiments/issueNN/run_*.sh` | `experiments/issueNN/` | `experiments/issueNN/test_*.py` |

A config row is `[datatype, method, [MR, NR], vlen, arch, [datasets], skip_ref_run, iterations]`. The 8th value (`iterations`) also makes the row **pinned**: one project for the first MRxNR only. Leave it out for search mode (one project with every tile up to the largest). See `test_features.py`.

## Test regime

Three tiers, cheapest first. Run tiers 1 and 2 after any change; tier 3 after any change to the generator, a template or a packer. Never run tiers 1 or 2 during a board series: they rewrite `pj_gen/` and `ref_baseline/generated/`.

**1. Host tests.** Seven `*_on_board` tests compile on the board through `fej` and skip when it is not reachable; `-k "not on_board"` runs host only.

```bash
cd tests/pure_xdsl_blis_rvv && ../../.venv/bin/python -m pytest                          # full suite, about 17 min
cd tests/pure_xdsl_blis_rvv && ../../.venv/bin/python -m pytest tests/test_layout_and_scripts.py   # layout and scripts, about 1 min
```

Use the repo's `.venv` (the conda Python lacks `xdsl`). The full suite covers the generator, every feature in the table above, every runner's plan (`--dry-run`, stubbed board) and every reader on made-up logs. `tests/test_layout_and_scripts.py` checks that the top level holds only the main workflow, that no path in code or docs points at a missing file, and that every runner and reader starts from its folder. Afterwards restore the tracked generated files: `git checkout -- ref_baseline/generated/`.

**2. Snapshot compare (no board).** Proves a refactor changed no output: generated C for every config, every `compile_macro_bench.sh --dry-run` plan, and the passing test IDs.

```bash
bash .scratch/reports/issue-128/snapshot.sh --pytest <out-dir>             # from the repo root
bash .scratch/reports/issue-128/snapshot.sh --compare <before-dir> <out-dir>   # prints every difference; exit 1 on any
```

**3. Feature baseline on the board.** Every user-facing option, one correctness run each against the scalar oracle: data types (fp32 AXPY, fp32 DOT, f16f16f32, int8), packers (RVV, `--scalar-pack`, every `--pack-b` variant, `--zero-pack-lines`, `--pad-ldb`), pinned tiles and search mode, `--threads 8`. About an hour.

```bash
./tests/pure_xdsl_blis_rvv/runners/run_feature_baseline.sh $CONN                    # all 15 cases
./tests/pure_xdsl_blis_rvv/runners/run_feature_baseline.sh $CONN --only fp32-axpy-rvv-pack   # the quick pass, a few minutes
```

Every case must end `PASS` (every row of its `all_raw_*.txt`); the table is in `logs/feature_baseline_<stamp>.tsv`. Check who else is on the board first (`docs/agents/execution.md`).
