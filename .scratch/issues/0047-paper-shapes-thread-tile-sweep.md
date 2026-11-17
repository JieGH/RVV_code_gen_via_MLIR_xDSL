# Paper shapes: best tile at 1 thread and at 8 threads (BERT B1-B5, Square S1-S5)

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/71

**Role / Label:** `needs-triage`. Board run. Independent of the YOLO work (#58, #63).
**Status:** ✅ Closed 2026-11-17 (merged in PR #73, then #74 into main). Run done on jbpi (6 of 6 steps ok); results: `PaperSweep/issue-71/README.md`.
**Target:** `jbpi`, one board for the whole series.
**Shapes:** `datasets/paper_bert_square.dat`, the paper's Table 1 (arXiv 2603.17800v1).

---

## 1. Question

The paper (the user's published design) picks its tile from a 1-core microkernel heatmap (best on BPI: 16x15) and runs GEMM on one core (S1-S5 at 32x7; BERT at 32x6, 16x15, 32x7). Which tile is best per shape when the same GEMM runs on 8 threads, and does it differ from the 1-thread best?

## 2. Run

- `run_paper_sweep.sh --after-pid 3758978 --issue 71` on `jbpi`: 1T L1, 8T L1, 8T L2, 1T L2, 1T L3, 8T L3 (A-B-B-A).
- Each step: `run_yolo_sweep.sh --config config_paper_search.json --threads 1|8 --launch l`. The 19 Candidate Tiles of `config_yolo_search.json`, which include the paper's 16x15, 32x7 and 32x6. Rows rotated per launch.
- Uninstrumented, reference protocol, `--tmin 2`, `--no-nc-overrides`, vector packer. 1T on core 4. 8T on cores 0-7 (core 0: the user's 8-thread exception, #63 §2).
- The sweep waits while another user's job is busy on the board.
- New in the tooling: `run_yolo_sweep.sh --config FILE` (the dataset is read from the config's rows).

## 3. Output

- Series folders under `tests/pure_xdsl_blis_rvv/logs/yolo_sweep_<stamp>/`, each with `tile_search_report.md` and a heatmap, copied to `PaperSweep/issue-71/`.
- Then: one table per shape, best tile at 1T and at 8T, GOPs by mean time and by median rep, beside the paper's pick.

## 4. Limits to state with the numbers

- The 19 tiles are the YOLO candidate list, not the paper's full mr x nr heatmap.
- `--tmin 2` with S5 (5000^3, about 250 GFLOP per call) gives few reps per launch at 1 thread.
- Not the paper's protocol (200-run average); the numbers are this repo's reference protocol.

## 5. Closure criteria

- 3 launches per thread count finished, logs committed, the table written.

## 6. Result (2026-11-17)

Run on `jbpi`, 01:00 → 06:26, 1T 8T 8T 1T 1T 8T, all 6 steps ok, no other user's job. **This repo's current design** on the paper's shapes; the paper's GFLOPS (BPI-F3, its own xDSL code) are not compared. Results: [`PaperSweep/issue-71/README.md`](../../PaperSweep/issue-71/README.md).

- 1 thread (core 4): best single tile **32x7** (9.53 GOPs over the pass); 32x7 best on 9 shapes, 16x15 on B2. The paper's picks hold.
- 8 threads (cores 0-7): best single tile **16x8** (48.27 GOPs); 16x8 best on 8 shapes, 16x12 on B2, 8x24 on B3. The paper's 32x7 gives 44.07 GOPs over the pass, 16x15 44.96.
- Run folders: 1T `run_2026-11-17_020326`, `run_2026-11-17_043612`, `run_2026-11-17_053929`; 8T `run_2026-11-17_024808`, `run_2026-11-17_033254`, `run_2026-11-17_062411`.
