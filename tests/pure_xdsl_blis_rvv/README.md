# Pure xDSL BLIS for RVV

The Project Generator and its bench. Pick a data type, packing method, tile and problem shapes in a config; the generator writes a C++ GEMM project; `compile_macro_bench.sh` builds it on the board, checks it and times it.

Setup, board connection (`$CONN`) and the full option list: the [repo README](../../README.md). Step by step: the [user guide](../../docs/user_guide.md).

## Quick start

From the repo root:

```bash
./tests/pure_xdsl_blis_rvv/compile_macro_bench.sh $CONN --no-nc-overrides --cpu 4 \
  --config configs/correctness/config_correctness.json --tmin 0.2        # correctness: every row PASS
./tests/pure_xdsl_blis_rvv/compile_macro_bench.sh $CONN --no-nc-overrides --cpu 4 --tmin 10   # benchmark config.json
```

Results: `logs/run_<stamp>/` (`comparison.md` is the summary). Without a board: `python3 generate_pure_blis_project.py --config config.json` in this folder writes `pj_gen/`.

## Layout

The top level holds only the main workflow.

```
generate_pure_blis_project.py   the Project Generator (with pack_builder.py, config.py, cache_config.py)
compile_macro_bench.sh          generate, build, check and time one config on the board
board_pin.sh                    board access, used by every board script
config.json                     the default config
configs/                        more configs: correctness/, tile_search/, benchmark/   (configs/README.md)
datasets/                       problem shapes, one per line: label M N K
runners/                        multi-run scripts: Tile Search, threads, packers, Reference Design, feature baseline   (runners/README.md)
readers/                        read run results: summaries, per-rep stats, reports   (readers/README.md)
ref_baseline/                   the Reference Design, for comparison   (ref_baseline/README.md)
tests/                          host tests (pytest)
templates/                      driver, Makefile and headers copied into every project
logs/                           run results, committed
pj_gen/                         generated projects, gitignored, rebuilt on every run
experiments/  notes/            past investigations and old notes, kept for history
modelLevel/  pack_only/         analytical cache model; pack-only bench
```

## Test regime

Cheapest first. Don't run tiers 1 or 2 while a board series runs: they rewrite `pj_gen/` and `ref_baseline/generated/`.

```bash
# 1. Host tests (about 17 min; the layout check alone: tests/test_layout_and_scripts.py, about 1 min)
cd tests/pure_xdsl_blis_rvv && ../../.venv/bin/python -m pytest
git checkout -- ref_baseline/generated/          # afterwards: restore the tracked files the tests rewrite

# 2. Snapshot compare (from the repo root): proves a refactor changed no generated code, plan or test
bash .scratch/reports/issue-128/snapshot.sh --pytest <out-dir>
bash .scratch/reports/issue-128/snapshot.sh --compare <before-dir> <out-dir>

# 3. Feature baseline on the board (from the repo root, about an hour): every data type, packer, tile mode and threads
./tests/pure_xdsl_blis_rvv/runners/run_feature_baseline.sh $CONN                      # --only NAME,... for some cases
```

Run 1 and 2 after any change; 3 after any change to the generator, a template or a packer. In tier 3 every case must say `PASS` (table in `logs/feature_baseline_<stamp>.tsv`). Use the repo's `.venv` Python; the conda one lacks `xdsl`.

## For developers: feature → configs → tests

| Feature | Entry point | Configs | Host tests (`tests/` unless noted) |
|---------|-------------|---------|------------------|
| Kernel generation | `generate_pure_blis_project.py`, `compile_macro_bench.sh` | `config.json` | `test_features.py`, `test_config.py`, `test_cache_config.py`, `test_pad_ldb.py`, `test_pad_zeroing.py`, `test_ref_protocol.py` |
| Correctness pass | `compile_macro_bench.sh --config configs/correctness/<x>.json --tmin 0.2` | `configs/correctness/` | `test_correctness_pass.py`, `test_pack_correctness.py` (board) |
| Feature baseline | `runners/run_feature_baseline.sh` | `configs/correctness/` | `test_run_feature_baseline.py` |
| Tile Search | `runners/run_yolo_sweep.sh`, `runners/run_paper_sweep.sh`, `readers/search_report.py` | `configs/tile_search/` | `test_tile_search.py`, `test_run_yolo_sweep.py`, `test_run_paper_sweep.py` |
| Data types: fp32, f16f16f32, int8 | data type column of a config row | `configs/correctness/` | `test_features.py`, `test_handover_ukernels.py` |
| Packing methods | `compile_macro_bench.sh` flags, `runners/run_pack_swap.sh` | `configs/benchmark/config_pack_swap*.json` | `test_pack_b_variants.py`, `test_pack_b_switch.py`, `test_pack_b_only.py`, `test_pack_dims_table.py`, `test_zero_pack_lines.py` |
| Threads and YOLO | `runners/run_parallel_gemm.sh`, `runners/run_hector_gemm.sh` | `configs/benchmark/` | `test_parallel_gemm.py`, `test_parallel_gemm_report.py`, `test_run_hector_gemm.py`, `test_hector_gemm_report.py` |
| Reference Design | `runners/run_head_to_head.sh`, `runners/run_ref_sweep.sh`, `ref_baseline/` | `ref_baseline/cases*.tsv` | `test_compare_ref_baseline.py`, `test_compare_ref_sweep.py`, `test_ref_pack_fix.py`, `test_ref_per_rep.py`, `test_run_ref_sweep.py`, `test_fringe_overhead.py` |
| Measurement and house rules | `board_pin.sh`, `readers/reps_report.py`, `readers/rawlog.py` | none | `test_board_access.py`, `test_board_pin.py`, `test_house_rules.py`, `test_reps_report.py`, `test_rawlog.py`, `test_perf_counters.py`, `test_summarize_run.py`, `test_slow_state_survey.py`, `test_shape_breakdown.py` |
| Layout | the folders above | none | `test_layout_and_scripts.py` |
| Past experiments | `experiments/issueNN/run_*.sh` | `experiments/issueNN/` | `experiments/issueNN/test_*.py` |
