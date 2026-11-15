# S3: One-core sweep: which tile is fastest on each YOLO shape, on one X60 core

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/61

**Role / Label:** `needs-triage`. Sub-issue of #58 ([[0041-yolo-best-microkernel-size-parent]]). Board run.
**Status:** 📝 Open. Not run.
**Blocked by:** #59 (S1, dataset and tiles), #60 (S2, config, report, correctness pass).
**Target:** `jbpi` or `jbpi2` (X60, same chip as the K1), core 4.
**Shapes:** `datasets/yolo26_k1.dat` (50 shapes).

---

## 1. Question

On one core, which MR x NR is fastest for each YOLO shape? How far is Héctor's 16x8 / 16x4 from the best?

This is the baseline for the 8-thread run (S6). It also answers part of Héctor's request before the parallel code exists. He says his single-thread GEMM is about level with OpenBLAS, so the one-core tile choice is worth knowing on its own.

## 2. Run

- **Our design's size search** (S2 #60): `compile_macro_bench.sh --config config_yolo_search.json --no-nc-overrides --cpu 4 --tmin <decision below>`
- uninstrumented build, reference protocol
- 3 launches; check for other users before each launch
- idle board for the numbers we quote (house rules #39; check other users)
- Wrap it in `run_yolo_sweep.sh` only if the series runs unattended (like `run_issue48.sh`: lock, idle check, `--start-at`, `--workspace`). Otherwise run it by hand.
- No tests and no `--dry-run` while the series runs (they rewrite `pj_gen/`).

## 2b. Open point: `--tmin` (moved here from #60 on 2026-11-15)

Héctor uses `TIMIN=2.0`. Our default for decisions is 10 s. With 50 shapes x 19 tiles, `--tmin 10` costs about 2.6 h per launch (about 8 h for 3 launches), against about 32 min per launch at `--tmin 2`. A third way: one pass at `--tmin 2`, then `--tmin 10` on the top few tiles per shape. Decide before the first launch.

## 3. Output

- `search_report.py` table per shape: best tile, GOPs by mean time and by median rep, launch spread, and Héctor's tile beside it.
- Summary: the best single tile for the whole network, and the best tile per n group (n = 4, 8, 16, 32, 64, 80, 128, 256).
- Logs committed, run folders cited.

## 4. Limits to state with the numbers

- Our GEMM packs A on every call. Héctor's phase-3 path gets A pre-packed from im2col.
- Our blocking comes from the cache model. His is fixed at MC=256, KC=256, NC=n.
- One core only. The best tile may change with 8 threads, because the threads share L2 and memory bandwidth (S6).

## 5. Closure criteria

- [ ] `--tmin` decided (§2b).
- [ ] 3 launches, idle board, logs committed.
- [ ] Report table committed and linked here and in the parent.
- [ ] Short answer written here: best tile per n group, and the gap to 16x8/16x4.
