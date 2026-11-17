# Scalar packing: rerun the YOLO and paper (BERT, Square) sweeps with the scalar packer, and check correctness

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/83
**Role / Label:** `needs-triage`. Board runs.
**Status:** opened 2026-11-17.
**Branch:** `scalar-packing-rerun-of-yolo-and-paper-sweeps`, cut from `pure-xdsl`.
**Reruns:** #61 (YOLO, 1 thread), #63 (YOLO, 8 threads, against Héctor's GEMMLib), #71 (paper shapes BERT B1-B5, Square S1-S5, 1 and 8 threads).

---

## 1. Question

Every sweep so far (#61, #63, #71) used the **vector packer**. How do the same benchmarks perform with the **scalar packer**? Does the best tile per shape change, and how far do the GOPs move?

**Same design, one change:** packing is scalar (`--scalar-pack`). Tiles, blocking, thread split, shapes, settings and run order all stay as in #61, #63 and #71. No new design.

## 2. What has to change first (found 2026-11-17)

- At 1 thread, `--scalar-pack` already works (`pack_RB_scalar` / `pack_CB_scalar`, `generate_pure_blis_project.py:188`).
- **At 8 threads it doesn't run:**
  - `compile_macro_bench.sh` refuses `--threads 8 --scalar-pack` (line 220).
  - The parallel loop nest (`render_parallel_nest`) only calls the vector packers.
- **Smallest fix:** in the parallel loop nest, call the same scalar packers on the same pieces each thread already packs:
  - B: one call per (panel, k-chunk) unit;
  - A: one call per thread's rows `[ir0, ir1)`.
- Lift the refusal, and pass `--scalar-pack` through `run_yolo_sweep.sh`, `run_paper_sweep.sh` and `run_issue63.sh`.
- Host test: 8 threads with scalar packing gives the same C and Bc/Ac as 1 thread, plus the existing refusal test updated.
- The vector path is unchanged.

## 3. Correctness (before any timing)

On the board, scalar packer, scalar oracle on (`skip_ref_run: false`), all 19 Candidate Tiles:

| dataset | 1 thread (core 4) | 8 threads (cores 0-7) |
|---|---|---|
| YOLO26 (`yolo26_k1.dat`, 50 shapes) | C1 | C2 |
| paper (`paper_bert_square.dat`, B1-B5, S1-S5) | C3 | C4 |

Pass: every row `PASS`. Correctness-only runs: their timings aren't quoted.

## 4. Timing runs (as the originals, plus `--scalar-pack`)

All runs: `jbpi`, uninstrumented, reference protocol, `--tmin 2`, `--no-nc-overrides`, 19 tiles rotated per launch, idle board.

| run | as | order |
|---|---|---|
| T1 YOLO 1T, core 4 | #61 | 3 launches |
| T2 YOLO 8T against Héctor's `gemm.x` (`b292c92`, unchanged: his own packing) | #63 | ours, his, his, ours, ours, his |
| T3 paper 1T / 8T | #71 | 1T, 8T, 8T, 1T, 1T, 8T |

## 5. Output

- Per shape: best tile and GOPs, scalar against vector, at 1 and 8 threads. Both metrics: by mean time and by median rep.
- The vector numbers come from the committed #61, #63 and #71 logs.
- Copies:
  - YOLO: `HectorYolo/issue-83/`, with a README and `report/`;
  - paper: `PaperSweep/issue-83/`.
- Brief: PDF + LaTeX, linked from this issue and #58, #61, #63 and #71.

## 6. Limits to state with the numbers

- The scalar and vector runs are from different days (vector: 15-17 Nov). Same board, build and protocol, but not interleaved in one launch, so small gaps (under about 3%) aren't called a difference.
- At 8 threads, core 0 is used (the user's 8-thread exception, #63 §2).

## 7. Closure

- [ ] §2 fix, with host tests passing
- [ ] C1-C4 all PASS
- [ ] T1-T3 done, logs committed
- [ ] Scalar-vs-vector tables, README and brief written and linked
