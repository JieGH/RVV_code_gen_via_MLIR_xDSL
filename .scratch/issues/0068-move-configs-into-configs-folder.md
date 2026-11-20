# Move the remaining configs into configs/<purpose>/

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/130

**Parent:** #127, organize `tests/pure_xdsl_blis_rvv` by feature.
**Role / Label:** `needs-triage`. Spec, 2026-11-20.
**Status:** open. Blocked by #128.
**Branch:** `organize-test-folder-by-feature`.

---

## 1. Problem

After #129, 22 configs not named for one issue stay at the top level, with no grouping by what they are for.

## 2. What to build

`git mv` into three folders, by purpose (`config.json` stays: it is the default of `compile_macro_bench.sh` and every doc names it):

| Folder | Configs |
|--------|---------|
| `configs/correctness/` | `config_correctness.json`, `config_paper_correctness.json`, `config_yolo_correctness.json`, `config_yolo_correctness_8t.json`, `config_yolo53_correctness_24x8.json`, `config_yolo53_correctness_32x7.json`, `config_issue112_mixed_correctness.json` |
| `configs/tile_search/` | `config_yolo_search.json`, `config_paper_search.json`, `config_yolo53_search_1t.json`, `config_yolo53_search_8t.json`, `config_dim_sweep_b3.json`, `config_dim_sweep_b4.json`, `config_dim_sweep_16x16_24x10.json` |
| `configs/benchmark/` | `config_pack_swap.json`, `config_pack_swap_b2b3b4.json`, `config_parallel_gemm.json`, `config_yolo53_24x8.json`, `config_yolo53_32x7.json`, `config_yolo_n1.json`, `config_b1b5_nr6_first.json`, `config_b1b5_nr7_first.json` |

Then fix every caller: runner defaults (`run_yolo_sweep.sh`, `run_paper_sweep.sh`, `run_parallel_gemm.sh`, `run_pack_swap.sh`, experiment scripts), tests, `docs/agents/execution.md`, `docs/metrics.md`. `compile_macro_bench.sh` takes the config path relative to its own folder; configs name datasets by basename, so neither changes.

## 3. Out of scope

- Renaming configs. The basename stays, so log names and old citations still match.

## 4. Tests

- Snapshot compare against the baseline: no difference (the `Config` line is reduced to the basename).
- Full host suite passes.

## 5. Closure criteria

- [ ] Only `config.json` left at the top level.
- [ ] Snapshot compare clean; full host suite passes.
