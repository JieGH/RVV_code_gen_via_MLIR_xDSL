# Scalar packing: rerun the YOLO and paper (BERT, Square) sweeps with the scalar packer, and check correctness

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/83
**Role / Label:** `needs-triage`. Board runs.
**Status:** ✅ runs and report done 2026-11-17. C1-C4 all PASS; T1-T4 done (idle boards, every launch ok). Brief [PDF](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/scalar-packing-rerun-of-yolo-and-paper-sweeps/.scratch/reports/issue-83/report/issue83_brief.pdf) / [LaTeX](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/scalar-packing-rerun-of-yolo-and-paper-sweeps/.scratch/reports/issue-83/report/issue83_brief.tex), report [`.scratch/reports/issue-83/README.md`](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/scalar-packing-rerun-of-yolo-and-paper-sweeps/.scratch/reports/issue-83/README.md). Open: close and merge, on the user's word.
**Follow-up #85 (when does vector packing pay off):** brief [PDF](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/when-does-vector-packing-pay-off/.scratch/reports/issue-85/report/issue85_brief.pdf) / [LaTeX](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/when-does-vector-packing-pay-off/.scratch/reports/issue-85/report/issue85_brief.tex), report [README](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/when-does-vector-packing-pay-off/.scratch/reports/issue-85/README.md).

**Result:** BERT/Square: no measurable difference (scalar/vector median 0.983 at 1T, 0.987 at 8T). YOLO: scalar slower, median 0.930 (1T) / 0.946 (8T), 0.59-0.83 on the n = 4 layers; best tile changes on 14 / 17 of 50 layers. T4: packing takes 49-83% of the time at n = 4-80 and 7-23% at n ≥ 256; scalar pack costs 1.07-1.94x the vector pack, and pack share x extra cost predicts the 1T ratio within 1% on 9 of 11 shapes.
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
- Host test: at 3 and 8 threads with scalar packing, C matches a double-precision reference on 6 split-edge cases x 2 betas x 20 reps (NaN-filled Ac/Bc); `--threads 8 --scalar-pack` is accepted by the script.
- The vector path is unchanged.

## 3. Correctness (before any timing)

On the board, scalar packer, scalar oracle on (`skip_ref_run: false`), all 19 Candidate Tiles:

| dataset | 1 thread (core 4) | 8 threads (cores 0-7) |
|---|---|---|
| YOLO-like (`config_yolo_correctness.json`: `yolo_correctness_small.dat`, m = 300, n = 4/16, k = 9/27) | C1 | C2 |
| paper-like (`config_paper_correctness.json`: `paper_correctness_small.dat`, B2, B3, 200^3) | C3 | C4 |

The full YOLO and paper sets are too big for the scalar oracle (O(m·n·k)): the house rule allows no shape above B2 (384x384x64) under it (`docs/agents/execution.md`, "Correctness pass"; `test_correctness_pass.py`). The small sets keep each set's fringe rows and columns and several K blocks; the host test covers the 8-thread split edges.

Pass: every row `PASS`. Correctness-only runs: their timings aren't quoted.

## 4. Timing runs (as the originals, plus `--scalar-pack`)

All runs: uninstrumented, reference protocol, `--tmin 2`, `--no-nc-overrides`, 19 tiles rotated per launch, idle board.

| run | as | order |
|---|---|---|
| T1 YOLO 1T, core 4 | #61 | 3 launches |
| T2 YOLO 8T against Héctor's `gemm.x` (`b292c92`, unchanged: his own packing) | #63 | ours, his, his, ours, ours, his |
| T3 paper 1T / 8T | #71 | 1T, 8T, 8T, 1T, 1T, 8T |

**Boards (user, 2026-11-17: use both boards, idle only):** `jbpi` runs the YOLO part (C1, C2, T2, T1), the board of #61 and #63. `jbpi2` runs the paper part (C3, C4, T3) at the same time, from its own checkout and board workspace. Driver: `run_issue83.sh --part yolo|paper`.

**T4, pack/compute split (user, 2026-11-17, added for Part 2 of the report):** the instrumented build (`--instr`, DEBUG, warm-up protocol, 200 reps) at 1 thread, core 4, on `jbpi2` after the paper part, from its own checkout and board workspace. 11 shapes spanning reuse n (Y14, Y30, Y47 n = 4; Y10, Y40 n = 16; Y29 n = 32; Y19 n = 80; Y42 n = 256; B1, B2 n = 384; S1 n = 1000) on each shape's 1-thread best tile (24x4, 32x4, 16x4, 24x8, 32x7, 16x15). 4 launches: vector, scalar, scalar, vector. Gives pack and compute time per shape for each packer: the share of time spent packing. Used for that share only; its GOPs are not quoted (timers cost 11-17%). The 8-thread path has no timers (#56). Driver: `run_issue83_pack_split.sh`, config `config_issue83_pack_split.json`, shapes `datasets/issue83_pack_split.dat`.

## 5. Output

**Subject (user, 2026-11-17):** the report is about the difference between vector and scalar packing on these tests, shape by shape. It has two parts:

- **Part 1, head to head:** vector against scalar packing, per shape, for YOLO (1T, 8T) and paper (BERT, Square; 1T, 8T): best tile, GOPs by mean time and by median rep, tile changes.
- **Part 2, case study, the broader picture:** why shape and reuse decide whether the packer matters, across models (BERT, Square, YOLO), thread counts (1T, 8T, from #61, #63, #71) and packers (#83).
  - Finding so far (partial logs, 17 Nov): on BERT and Square the packer choice makes no measurable difference (8T median scalar/vector 0.988); on YOLO at 8T scalar is slower on every layer (median 0.947), worst where n is smallest (n = 4: 0.67-0.82).
  - Reason to show in detail: a packed A element is used n times (packing per flop about 1/(2n)), so small-n layers spend a large share of their time packing; large-n shapes hide the packer. Also the role of small k and of thread count.
  - Note: this compares two packers; it does not test no packing at all.
- **Glossary (user, 2026-11-17):** the report starts with a glossary of every run-setup term it uses (launch, starting tile, A-B-B-A, tile MRxNR, best tile, GOPs by mean time / by median rep, launch spread, reference protocol, uninstrumented build, scalar / vector packer, 1T / 8T, core 4, reuse n). It says "launch 1/2/3", never "L1/L2/L3" (reads as a cache level).

- Per shape: best tile and GOPs, scalar against vector, at 1 and 8 threads. Both metrics: by mean time and by median rep.
- The vector numbers come from the committed #61, #63 and #71 logs.
- Copies:
  - all of #83 in one folder, `.scratch/reports/issue-83/`: `yolo/` (T1, T2 series), `paper/` (T3 series), `preliminary/` (17 Nov figures), then the README and `report/` (user, 2026-11-17: one place for all reports and logs; #56, #61, #62, #63, #71 and Héctor's email (`issue-58/`) moved there the same day). The links `HectorYolo/issue-83` and `PaperSweep/issue-83` used during the series are gone; the dump default (`hector_yolo_dump.sh`) is now `.scratch/reports/`.
- Brief: PDF + LaTeX, linked from this issue and #58, #61, #63 and #71.

## 6. Limits to state with the numbers

- The scalar and vector runs are from different days (vector: 15-17 Nov). Same board, build and protocol, but not interleaved in one launch, so small gaps (under about 3%) aren't called a difference.
- The paper part runs on `jbpi2`, #71 ran on `jbpi`: same SoC (SpacemiT X60), but a board-to-board gap adds to the day gap.
- At 8 threads, core 0 is used (the user's 8-thread exception, #63 §2).

## 7. Closure

- [x] §2 fix, with host tests passing
- [x] C1-C4 all PASS (76/76, 76/76, 57/57, 57/57)
- [x] T1-T4 done, logs committed
- [x] Scalar-vs-vector tables, README and brief written and linked (#58, #61, #63, #71)
