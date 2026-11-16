# S3: One-core sweep: which tile is fastest on each YOLO shape, on one X60 core

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/61

**Role / Label:** `needs-triage`. Sub-issue of #58 ([[0041-yolo-best-microkernel-size-parent]]). Board run.
**Status:** ✅ Series run 2026-11-15/16 on `jbpi`: 3 of 3 launches ok, idle board (§5a). Open question on Y14 (§5b).
**Blocked by:** #59 (S1, dataset and tiles), #60 (S2, config, report, correctness pass).
**Target:** `jbpi` or `jbpi2` (X60, same chip as the K1), core 4.
**Shapes:** `datasets/yolo26_k1.dat` (50 shapes).

---

## 1. Question

On one core, which MR x NR is fastest for each YOLO shape? How far is Héctor's 16x8 / 16x4 from the best?

This is the baseline for the 8-thread run (S6). It also answers part of Héctor's request before the parallel code exists. He says his single-thread GEMM is about level with OpenBLAS, so the one-core tile choice is worth knowing on its own.

## 2. Run

- `run_yolo_sweep.sh --tmin 2` (§2c): 3 launches of `config_yolo_search.json`, core 4, uninstrumented build, reference protocol, `--no-nc-overrides`.
- Idle board only: the series reads the board's processes before each launch and waits while another user's job is busy (house rules #39).
- No tests and no `--dry-run` while the series runs (they rewrite `pj_gen/`).

## 2b. `--tmin`: decided 2 (user, 2026-11-15)

`--tmin 2`, as Héctor uses (`TIMIN=2.0`). Board: whichever of `jbpi`/`jbpi2` is free and idle. Every result from this series names `--tmin 2`; it is a screening-level number next to our usual 10 s.
Board time measured on #60's correctness pass: about 3 min per tile to build, so about 90 min per launch, about 4.5 h for 3. Measured on the series: about 221 s per tile (100 s timed + 120 s build and copy), about 71 min per launch, 3 h 32 min for 3.

## 2c. Decisions (grilling, 2026-11-15; agent's recommended answers, accepted by the user's standing prompt)

1. **A runner script, `run_yolo_sweep.sh`**, since the series runs unattended for hours and must wait for an idle board.
2. **Tile order rotates between launches** (house rule 2, the #50 review lesson): each launch runs a copy of `config_yolo_search.json` with the rows rotated by 0, 7, 14 (of 19), written to the series folder. One `compile_macro_bench.sh` call per launch, no per-tile config files.
3. **One board for all 3 launches**: the first idle board of `jbpi,jbpi2` at the start, kept, so the launches compare like for like.
4. **Wait, never start under load**: another user's busy job (>= 20% CPU, `OTHER_USERS_AWK`) means wait 10 min and read again. A board that cannot be read counts as busy.
5. **Report**: `search_report.py` on the 3 run folders, `--compare 16x8 --compare n=4:16x4`.

## 3. Output

- `search_report.py` table per shape: best tile, GOPs by mean time and by median rep, launch spread, and Héctor's tile beside it.
- Summary: the best single tile for the whole network, and the best tile per n group (n = 4, 8, 16, 32, 64, 80, 128, 256).
- Logs committed, run folders cited.

## 4. Limits to state with the numbers

- Our GEMM packs A on every call. Héctor's phase-3 path gets A pre-packed from im2col.
- Our blocking comes from the cache model. His is fixed at MC=256, KC=256, NC=n.
- One core only. The best tile may change with 8 threads, because the threads share L2 and memory bandwidth (S6).

## 5a. Result (`jbpi`, uninstrumented, reference protocol, `--tmin 2`, core 4, `--no-nc-overrides`, 3 launches)

Series `logs/yolo_sweep_2026-11-15_205445/` (20:54-00:27); run folders `logs/run_2026-11-15_220501`, `logs/run_2026-11-15_231549`, `logs/run_2026-11-16_002638` (rotation starts 8x4, 16x8, 24x8). No other user's load before or after any launch. Full report and heatmap: `logs/yolo_sweep_2026-11-15_205445/tile_search_report.md`, `tile_search_heatmap.png` (copy in `HectorYolo/issue-61/sweep_final_2026-11-15/`).

GOPs over one pass of that n's shapes, whole call, packing included, as **by mean time / by median rep**:

| n | shapes | best tile | best GOPs | 16x8 GOPs | 16x4 GOPs |
|---|---|---|---|---|---|
| 4 | 3 | 24x4 | 2.50 / 2.54 | 1.66 / 1.67 | 2.50 / 2.52 |
| 8 | 1 | 24x9 | 1.12 / 1.12 | 0.88 / 0.88 | 0.86 / 0.86 |
| 16 | 10 | 24x9 | 2.12 / 2.12 | 1.67 / 1.67 | 1.63 / 1.63 |
| 32 | 6 | 24x9 | 3.21 / 3.21 | 2.81 / 2.81 | 2.59 / 2.59 |
| 64 | 10 | 24x8 | 4.97 / 4.98 | 4.51 / 4.52 | 4.01 / 4.01 |
| 80 | 6 | 32x7 | 5.90 / 5.88 | 5.29 / 5.27 | 4.55 / 4.56 |
| 128 | 9 | 32x7 | 6.61 / 6.62 | 6.01 / 6.01 | 5.11 / 5.11 |
| 256 | 5 | 24x8 | 7.60 / 7.56 | 7.25 / 7.28 | 6.17 / 6.18 |
| all | 50 | **24x8** | **4.26 / 4.25** | 3.79 / 3.79 | 3.44 / 3.44 |

Short answer: one tile for the whole network is **24x8** (698.6 ms per pass by mean time, against 785.3 ms for 16x8). Per n: 24x9 for n = 8-32, 24x8 for n = 64 and 256, 32x7 for n = 80 and 128. For n = 4, 24x4 and Héctor's 16x4 tie (2.50 by mean time). 16x8 is the best tile on 1 of 50 shapes (Y44). MR = 8 tiles are the slowest (2.97-3.43 over the dataset).

**Report:** [`.scratch/reports/issue-61-report.md`](../reports/issue-61-report.md) (engineering report: method, changes, verification, results, limitations; copy in `HectorYolo/sweep_final_2026-11-15/`).

## 5b. Open question: Y14 with MR = 32

Y14 (4800x4x16): 32x4 gives 0.69 / 0.69 / 0.68 GOPs by mean time in the 3 launches, 32x6 and 32x7 0.50-0.62, while 16x4 and 24x4 give 2.4-2.5. Y30 (1200x4x16) with 32x4 gives 2.7. Repeats in all 3 launches, so not noise. Cause not looked at (blocking or packing for m = 4800 with MR = 32 are the first suspects). It does not change any answer above: MR = 32 is not the pick for n = 4.

## 5. Closure criteria

- [x] `--tmin` decided (§2b): 2.
- [x] 3 launches, idle board, logs committed (§5a).
- [x] Report table committed and linked here and in the parent.
- [x] Short answer written here: best tile per n group, and the gap to 16x8/16x4 (§5a).
