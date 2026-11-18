# Issue #94: YOLO GEMM at 1 and 8 threads, our vector packer, our scalar packer and Héctor's

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/94 · issue write-up: `.scratch/issues/0053-yolo-gemm-at-1-and-8-threads-our-vector-packer-our-scalar-packer-and-hectors.md`.

- **Report (PDF, A4, 6 pages):** [`report/issue94_report.pdf`](report/issue94_report.pdf) ([LaTeX](report/issue94_report.tex)). It has a glossary, setup, correctness, a one-pass table, a by-n table (GOPs by mean time only, since 2026-11-18), charts, a per-shape appendix and the log list.
- **Final report (PDF, A4, 23 pages):** [`final/issue94_final_report.pdf`](final/issue94_final_report.pdf) ([LaTeX](final/issue94_final_report.tex)). This is the #91 report with its §1.2 (ours against Héctor's) replaced by this rerun. The answer, glossary, Part 4, correctness, limits and runs list are updated to match. Since 2026-11-18 it reads Parts 1–3 and the appendix from `../yolo-multithread/report/` (GOPs by mean time only); `final/` keeps its own `r0_answer` and `r4_close`. The combined report in `../yolo-multithread/` supersedes it.
- **Data:** [`report/make_data.py`](report/make_data.py) → `report/data/`. `make_data.py --check` reproduces the #56 and #62 pass numbers quoted in #91, and writes them to `report/check/`, which is not kept.
- **Logs:** one folder per launch here (copies of `tests/pure_xdsl_blis_rvv/logs/`). The series plan and order are in `issue94_2026-11-18_115554/` (`plan.txt`, `steps.tsv`) and the console output is in `issue94_series_output.txt`. Run folders (`run_*`, per-rep times) stay in `tests/pure_xdsl_blis_rvv/logs/`. `issue94_stopped_1152_output.txt` is the first start, stopped during C1 before any timed launch.

Build: final report from `final/`, `tectonic -X compile issue94_final_report.tex` (after the report data below). Report: from `report/`, `python3 make_data.py ../../../../tests/pure_xdsl_blis_rvv/logs/issue94_2026-11-18_115554 --logs ../../../../tests/pure_xdsl_blis_rvv/logs --out data`, then `tectonic -X compile issue94_report.tex`.

## Setup

- Board: jbpi.
- Ours: tile 24x8, the 50 YOLO26 shapes, uninstrumented build (`INSTRUMENTED=0`), reference protocol, `--tmin 2`, `--no-nc-overrides`.
- Héctor's: `gemm.bin` at `b292c92`, unchanged; `TIMIN=2.0`, `TEST=T`, his tiles (16x8; 16x4 at n = 4).
- Cores: 1T on core 4. 8T on cores 0-7 with `OMP_PROC_BIND=close OMP_PLACES=cores` for all three setups.
- Order: 3 launches per setup per thread count, 18 launches in one series (2026-11-18, 12:00-13:09). Thread counts go A-B-B-A over the rounds; the setup order rotates inside each block.
- The board was idle before every launch: there are no side notes and no waits.

## Correctness

- C1 (1T) and C2 (8T): scalar packer, oracle on, 4 of 4 rows PASS each.
- Every timed launch of ours: 50 of 50 PASS.
- Every launch of Héctor's: 53 of 53 rows OK in his own check.

## Results (GOPs by mean time, mean of 3 launches)

| setup | 1T (core 4) ms | GOPs | 8T (cores 0-7) ms | GOPs | speedup |
|---|---|---|---|---|---|
| V: ours, vector packer | 697.2 | 4.27 | 155.2 | 19.17 | 4.49x |
| S: ours, scalar packer | 733.9 | 4.05 | 160.6 | 18.52 | 4.57x |
| H: Héctor's gemm.bin | 583.4 | 5.10 | 156.6 | 18.99 | 3.73x |

- Launch ranges:
  - 1T: V 4.25-4.29, S 4.05-4.05, H 5.08-5.12.
  - 8T: V 19.14-19.18, S 18.47-18.54, H 18.75-19.27.
- Without Y00 (as #91 §1.4 quotes ours against Héctor's): 8T V 19.59, S 18.96, H 22.98 GOPs (148.4, 153.3, 126.5 ms). Héctor's Y00 took 29.5, 30.2 and 30.4 ms in his three 8T launches; at 1T it took 15.3-15.6 ms, and at 8T in #62 (unbound) 13.7-15.3 ms. So the whole-pass tie at 8T comes from Y00.
- By n: report §3.

## Not the same on both sides

- Compiler: ours g++ 13.2 `-O2`, Héctor's gcc-14 `-O3`. We kept ours so the results stay comparable with #56. A matched-compiler run is offered as a separate run.
