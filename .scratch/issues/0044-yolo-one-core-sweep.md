# S3: One-core sweep: which tile is fastest on each YOLO shape, on one X60 core

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/61

**Role / Label:** `needs-triage`. Sub-issue of #58 ([[0041-yolo-best-microkernel-size-parent]]). Board run.
**Status:** 🔨 Runner built 2026-11-15 (§2c). Series not run yet: starts after #60's correctness pass.
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
Board time measured on #60's correctness pass: about 3 min per tile to build, so about 90 min per launch, about 4.5 h for 3.

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

## 5. Closure criteria

- [x] `--tmin` decided (§2b): 2.
- [ ] 3 launches, idle board, logs committed.
- [ ] Report table committed and linked here and in the parent.
- [ ] Short answer written here: best tile per n group, and the gap to 16x8/16x4.
