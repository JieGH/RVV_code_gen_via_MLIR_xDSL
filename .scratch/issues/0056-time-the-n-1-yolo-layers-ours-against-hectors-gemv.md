# Time the n = 1 YOLO layers (Y06, Y21, Y35): ours against Héctor's GEMV

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/108

**Combined YOLO report (all findings, read first):** `.scratch/reports/yolo-multithread/report/yolo_report_v2.pdf`; index: `.scratch/reports/yolo-multithread/README.md`.

**Role / Label:** `needs-triage`. Opened 2026-11-18 at the user's request. Board runs on `jbpi`.
**Status:** stopped by the user 2026-11-18 23:36, superseded by #109 [[0057-yolo-gemm-rerun-best-single-tile-per-thread-count-against-hectors-unchanged-gemm]]. Series `issue94_2026-11-18_225810`: steps 1–14 of 18 ok (every row of ours PASS), step 15 (S 1T) cut off mid-build; board left clean. Its 8T rows ran ours at 24x8, the 1T best, which breaks the new rule (one best tile per thread count), so none of it goes into the report. Kept for #109: `yolo26_k1_n1.dat`, the `--config`/`--issue` runner flags and test, and 24x8 PASS on n = 1. Close as superseded once #109 is under way.
**Branch:** `time-the-n-1-yolo-layers-ours-against-hectors-gemv`, cut from `pure-xdsl`.
**Related:** #94 [[0053-yolo-gemm-at-1-and-8-threads-our-vector-packer-our-scalar-packer-and-hectors]] (the series this extends), #58 (parent), #59 (n = 1 left out).

---

## 1. Question

The combined report §1.2 (from #94) leaves Y06, Y21 and Y35 empty. These are the n = 1 layers:

| layer | m | n | k |
|---|---|---|---|
| Y06 | 4800 | 1 | 9 |
| Y21 | 1200 | 1 | 9 |
| Y35 | 300 | 1 | 9 |

Héctor runs them as a GEMV; ours never ran them. Time them for V, S and H at 1T and 8T and fill the empty rows.

## 2. Setup (same as #94)

- Ours: tile 24x8 (n = 1 runs as an n-edge tile), uninstrumented, reference protocol, `--tmin 2`, `--no-nc-overrides`, every row checked. New dataset `datasets/yolo26_k1_n1.dat` (the 3 rows), config `config_yolo_n1.json`.
- Héctor: `gemm.bin` at `b292c92`, unchanged, `TIMIN=2.0`, `TEST=T`. His program always runs all 53 rows of his `yolo.dat`; we read rows 6, 21, 35 (his GEMV path).
- 1T on core 4; 8T on cores 0-7, `OMP_PROC_BIND=close OMP_PLACES=cores` on both sides.
- One series, 18 launches, same order as #94 (`run_issue94.sh`).
- Correctness: C1/C2 of #94, plus every timed launch of ours checks every row against the oracle.

## 3. Code (smallest change)

- `run_parallel_gemm.sh --config FILE`: run another config; the shape count comes from the dataset it names.
- `run_issue94.sh --config FILE --issue N`: pass the config to ours, file logs under `issue-N`.

## 4. Output

- Logs in `.scratch/reports/issue-<N>/`.
- Combined report §1.2: per-layer charts get the Y06/Y21/Y35 bars. The one-pass tables stay over the 50 shapes (#94); the n = 1 rows get their own small table.
