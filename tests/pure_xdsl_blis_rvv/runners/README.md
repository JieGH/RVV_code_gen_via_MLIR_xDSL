# runners/: multi-run scripts

Each runner wraps `../compile_macro_bench.sh` (or the Reference Design's `../ref_baseline/run_ref_baseline.sh`) to run a series: several launches, an order, the idle-board wait, a report at the end. You do not need them to generate or run one kernel: use `../compile_macro_bench.sh` for that.

Run them from the repo root. Every runner takes `--help` (prints its header) and `--dry-run` (prints the plan, touches no board). Long series run detached:

```bash
setsid nohup ./tests/pure_xdsl_blis_rvv/runners/run_yolo_sweep.sh --tmin 2 --issue <N> > /tmp/sweep.log 2>&1 &
```

Every runner sources `../board_pin.sh` (board access, pinning, other-user checks) and writes its run folders under `../logs/`.

| Script | What it does | Report it writes (`../readers/`) |
|--------|--------------|----------------|
| `run_feature_baseline.sh` | **feature baseline**: one correctness run per user-facing option (data type, packer, `--pack-b` variant, pinned and search tiles, threads); every row must PASS. `--list`, `--only NAME,...` | `logs/feature_baseline_<stamp>.tsv` |
| `run_yolo_sweep.sh` | Tile Search on the YOLO26 shapes, one core (#61) or `--threads 8` (#63); rows rotated per launch | `search_report.py` |
| `run_paper_sweep.sh --issue N` | Tile Search on the paper's BERT B1-B5 and Square S1-S5 shapes, 1 and 8 threads alternating; calls `run_yolo_sweep.sh` | `search_report.py` |
| `run_parallel_gemm.sh` | our GEMM at 1 and 8 threads on the YOLO26 shapes (#56) | `parallel_gemm_report.py` |
| `run_hector_gemm.sh --src <clone>` | Héctor's GEMMLib_yolo (commit `b292c92`, unedited) at 1 and 8 threads (#62) | `hector_gemm_report.py` |
| `run_pack_swap.sh` | vector pack against scalar pack, order V S S V on one core (#26) | `compare_pack_swap.py` |
| `run_scalar_pack_sweep.sh` | unattended scalar-pack sweep, two boards taking turns (#26) | `compare_pack_swap.py` |
| `run_head_to_head.sh` | our run and the Reference Design back to back on one core (#21) | `../ref_baseline/compare_ref_baseline.py` |
| `run_ref_sweep.sh` | unattended sweep of the Reference Design alone (#29) | `../ref_baseline/compare_ref_sweep.py` |
| `gen_handover_ukernels.py --out DIR` | standalone fp32 and f16f16f32 microkernels in Héctor's argument order, with a naive reference and a test (#112). Host only | none |

Helpers, not run by hand:

| Script | Role |
|--------|------|
| `hector_yolo_dump.sh` | sourced by the YOLO runners: copies a series' report and logs to `.scratch/reports/issue-<N>/` |
| `cron_bench_run.sh <board>` | cron wrapper around `compile_macro_bench.sh`, one board at a time via a lock |
| `gather_sweep_report.sh [HH:MM]` | waits, then runs `report_scalar_sweep.py` on a scalar-pack sweep |

Host tests for each runner: `../tests/test_run_*.py`, `../tests/test_board_pin.py`, `../tests/test_handover_ukernels.py`.
