# YOLO GEMM at 1 and 8 threads: our vector packer, our scalar packer and Héctor's

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/94

**Role / Label:** `needs-triage`. Opened 2026-11-18 at the user's request. Board runs on `jbpi`.
**Status:** open. Driver and host test done (637 passed on the full suite); board series next.
**Branch:** `yolo-gemm-at-1-and-8-threads-our-vector-packer-our-scalar-packer-and-hectors`, cut from `pure-xdsl`.
**Related:** #56 [[0040-parallel-gemm-design]] (ours, vector packer), #62 [[0045-read-hector-parallel-gemm]] (Héctor's `gemm.bin`), #83 [[0050-scalar-packing-rerun-of-yolo-and-paper-sweeps]] (scalar packer).

---

## 1. Question

Rerun the #56/#62 comparison with our scalar packer added. Three setups:

1. **V**: ours, vector packer (as #56).
2. **S**: ours, scalar packer (`--scalar-pack`, works at 8 threads since #83).
3. **H**: Héctor's `gemm.bin` at `b292c92`, unchanged (as #62).

Each at 1 thread (core 4) and 8 threads (cores 0-7).

Why: #83 compared the scalar packer to Héctor only at 8 threads, with the best tile per shape (#63 setup). It never covered the #56 setup: one fixed tile, one pass, 1 and 8 threads. And #56 compared to Héctor's #62 runs from another day, not mixed in one series.

## 2. Setup (same as #56)

- Tile 24x8, the 50 YOLO26 shapes of `datasets/yolo26_k1.dat` (no n = 1 rows), board `jbpi`.
- Ours: uninstrumented build (`INSTRUMENTED=0`), reference protocol, `--tmin 2`, `--no-nc-overrides`, every row checked (`config_parallel_gemm.json`).
- 8 threads: `OMP_PROC_BIND=close OMP_PLACES=cores`, cores 0-7 (core 0 by the user's 8-thread exception, #56 §5).
- Héctor: `TIMIN=2.0`, `TEST=T`, his tiles (16x8; 16x4 at n = 4), his `gemm_study/yolo.dat` (53 rows; the 3 n = 1 rows go to his GEMV and are left out of the comparison, as in #56).
- All three setups mixed in one series, 3 launches per setup per thread count, the order rotated between rounds.

## 3. Order

Three rounds. Each round runs one block per thread count; thread counts go 1 8 8 1 1 8 over the rounds (as #56). Inside a block the setup order rotates, so each setup takes each place once:

| round | first block | second block |
|---|---|---|
| 1 | 1T: V S H | 8T: V S H |
| 2 | 8T: S H V | 1T: S H V |
| 3 | 1T: H V S | 8T: H V S |

18 launches. Before each launch the board is read; while another user's job is busy, it waits (never kills).

## 4. Correctness first

On `jbpi`, before any timing: tile 24x8, scalar packer, scalar oracle on, on `datasets/yolo_correctness_small.dat` (as #83 C1/C2):

- **C1:** 1 thread, core 4.
- **C2:** 8 threads, cores 0-7.

Every row must PASS, or the series stops before timing. Their times are not quoted.

## 5. Plan for the code (smallest change)

- `run_parallel_gemm.sh`: add `--threads 1|8` (run only that thread count) and `--scalar-pack` (passed to `compile_macro_bench.sh`).
- New `run_issue94.sh`, like `run_issue83.sh` / `run_issue63.sh`: C1, C2, then the 18 launches by calling `run_parallel_gemm.sh --launches 1` and `run_hector_gemm.sh --launches 1`. Writes the plan and order into its log folder.
- New `config_issue94_correctness.json`: 24x8 on `yolo_correctness_small.dat`, oracle on.
- Host test `test_run_issue94.py` (`--dry-run`: order and refusals).
- The vector path and Héctor's code stay as they are.

## 6. Output

`.scratch/reports/issue-94/`: logs and a report (PDF + LaTeX) with the tables of #91 §1.2:

- whole pass ms and GOPs at 1T and 8T, speedup 8T/1T;
- the by-n table;
- a column group per setup (V, S, H);
- both GOPs by mean time and by median rep (Héctor: by mean time only, his driver gives no per-rep time);
- glossary first, pgfplots horizontal bars with values, one chart per figure.

Linked from this issue and #56, #62, #83.

## 7. Closure

- [x] Driver and host test, tests passing
- [ ] C1, C2 all PASS
- [ ] 18 launches done, logs committed
- [ ] Report written and linked
- [ ] Code review done, findings fixed
