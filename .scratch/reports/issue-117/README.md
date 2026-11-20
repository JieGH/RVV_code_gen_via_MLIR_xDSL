# Issue #117: report output before and after the raw-log reader

"Before" = the tree as of the start of #117 (scripts copied before any edit). "After" = this branch.
Both run against the same committed logs under `tests/pure_xdsl_blis_rvv/logs/`.

| Report | Input | Result |
|---|---|---|
| `compare_pack_swap.py` | vector runs `run_2026-11-19_143324`, `_160242`; scalar runs `run_2026-11-19_144226`, `_155626` | identical (`pack_swap_before.txt` / `pack_swap_after.txt`) |
| `report_scalar_sweep.py` | `sweep_scalar_2026-09-20_132532` | identical except the output path and the "Generated" time (`sweep_before.txt` / `sweep_after.txt`) |
| `parallel_gemm_report.py` | `parallel_gemm_2026-11-19_102958`, `_103706`, `_104145` | identical |
| `hector_gemm_report.py` | `hector_gemm_2026-11-19_101535`, `_102246`, `_103438` | identical |
| `summarize_run.py` | `run_2026-11-19_160242` (copied; the script writes into the run dir) | identical |
| `compare_ref_baseline.py` | `refbase_2026-10-29_130709`, with and without `--raw-name ..._noinstr_refproto_no_nc_override.txt` | identical |

Not diffed on real data: `compare_ref_sweep.py` (its tests pass; it only imports `VECTOR_GLOB`), `issue46_report.py`, `issue48_report.py` (not switched to the reader).
