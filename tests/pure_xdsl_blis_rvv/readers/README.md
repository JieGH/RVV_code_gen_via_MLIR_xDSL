# readers/: tools that read run logs

Each reader takes run folders under `../logs/` and prints or writes a table. None touches the board. Run them from the repo root with the repo's Python (`.venv/bin/python`, or `python3` if it has `matplotlib` for the plots); `--help` lists the options.

```bash
.venv/bin/python tests/pure_xdsl_blis_rvv/readers/reps_report.py tests/pure_xdsl_blis_rvv/logs/run_<stamp>
```

| Script | Reads | Gives |
|--------|-------|-------|
| `summarize_run.py <run dir>` | `all_raw_*`, `all_perf_*` | `comparison.md` in the run folder. `compile_macro_bench.sh` calls it after every run |
| `reps_report.py <run dir> [...]` | per-rep logs `reps_*.txt[.gz]` | per case: GOPs by mean time, CV, slow-rep share; average over launches. Warns under 3 launches |
| `search_report.py <run dir> [...]` | a Tile Search's run folders, one per launch | best tile per shape, GOPs grid, optional `--heatmap` |
| `compare_pack_swap.py --vector RUN... --scalar RUN...` | vector-pack and scalar-pack runs | e2e GOPs per side and the gap |
| `parallel_gemm_report.py <series>` | a `run_parallel_gemm.sh` series | 1- and 8-thread GOPs per YOLO layer, against Héctor's |
| `hector_gemm_report.py <series>` | a `run_hector_gemm.sh` series | Héctor's GOPs per YOLO layer |
| `report_scalar_sweep.py` | a scalar-pack sweep so far | per-layer report against the vector-pack sweep |
| `slow_state_survey.py` | every per-rep log under `../logs/` (on the dev machine only, #143) | the two-speed survey (#47). Slow: reads all of `../logs/` |
| `shape_breakdown.py <dataset> --mr M --nr N` | a dataset file | blocking, reuse and L1 set use per shape (#34) |
| `pack_dims_table.py` | `../experiments/issue48/` config | what each vector packer does per MR and NR (#48) |
| `rawlog.py` | (library) | how the driver names and lays out a run's raw CSV; the other readers import it |

`reps_report.py` and `rawlog.py` are also imported by `../ref_baseline/` and `../experiments/`. Host tests: `../tests/test_<name>.py`.
