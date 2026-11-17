# Issue #79, Part B: are #63's numbers right?

Audit date 2026-11-17. Host only, committed logs only. I recomputed everything with a new script, `recompute_63.py`, written for this audit; #63's `search_report.py` and `tables.py` weren't used. Full output: [`B-recompute-output.md`](B-recompute-output.md).

**Inputs** (all committed):
- ours 8T, #63: `tests/pure_xdsl_blis_rvv/logs/run_2026-11-16_222551`, `run_2026-11-16_234245`, `run_2026-11-17_005337`;
- ours 1T, #61: `run_2026-11-15_220501`, `run_2026-11-15_231549`, `run_2026-11-16_002638`;
- his 8T: `.scratch/reports/issue-63/hector_gemm_2026-11-16_222943`, `_223301`, `hector_gemm_2026-11-17_005723` (`L1_T8_jbpi.csv`).

**Run conditions:**
- ours: `jbpi`, uninstrumented, reference protocol, `--tmin 2`, `--no-nc-overrides`, vector packer, 8 threads on cores 0-7 (`OMP_PROC_BIND=close`);
- his: `gemm.x` at `b292c92`, `TIMIN=2.0`, `taskset -c 0-7`, no binding;
- 1T column: core 4.

| item | verdict |
|---|---|
| B1 coverage, correctness | **FAIL → fixed**: no 8T tile-search row was ever checked for correctness |
| B2 headline numbers | PASS: they reproduce, but the headline needs a caveat |
| B3 best tile per shape | PASS: holds on held-out launches |
| B4 best tile per n | **GAP**: 5 of 8 n-groups change winner between launches |
| B5 ours vs his | **GAP**: "ours faster" rests on one unstable shape of his |
| B6 the two anomalies | explained in part, causes named, not proven |

---

## B1. Coverage and correctness: FAIL, fixed by Part C3

- All 3 x 950 rows of #63's 8-thread sweep have `Status = SKIP` ("correctness not checked"): `config_yolo_search.json` sets `skip_ref_run: true` on every row. The same holds for #61's 1-thread sweep.
- #61's tiles were checked at 1 thread by #60's correctness pass (76 of 76 PASS, `run_2026-11-15_203822`, 4 small shapes, m = 300).
- **At 8 threads only tile 24x8 was ever checked on the board** (#56, 50 of 50 PASS in 6 launches). The 18 other tiles were timed but never checked at 8 threads, and they include every tile #63 recommends (32x4, 32x7, 32x6, 16x15, 16x14, 16x12).
- The host test `test_parallel_gemm.py` checks the parallel loop nest at 3 and 8 threads, but with stub kernels at one tile (16x6).
- Fix: the board correctness pass in Part C3 (all 19 tiles x 50 shapes x 8 threads, oracle on).
- His launches: 53 of 53 rows `OK` in every launch (his `TEST=T`). Our 50 shapes are all in his CSV, row for row.

## B2. Headline numbers: PASS (reproduce within rounding)

One pass over the 50 shapes, ms (GOPs by mean time), launches L1 / L2 / L3:

| what | L1 | L2 | L3 | mean | GOPs | #63 says |
|---|---|---|---|---|---|---|
| ours 1T 24x8, core 4 (#61) | 698.9 | 701.2 | 696.6 | 698.9 | 4.25 | 698.6 · 4.26 |
| ours 8T best single tile 16x12 | 144.8 | 145.3 | 147.0 | 145.7 | 20.41 | 145.6 · 20.43 |
| ours 8T best tile per shape | 134.1 | 132.8 | 135.8 | 134.2 | 22.15 | 134.1 · 22.17 |
| **ours 8T at his tile 16x8** | 149.7 | 147.7 | 152.2 | 149.8 | 19.85 | not shown |
| his 8T | 136.8 | 184.7 | 229.7 | 183.7 | 16.18 | 183.7 · 16.18 |
| ours 8T best per shape, without Y00 | 127.9 | 126.6 | 129.5 | 128.0 | 22.72 | 127.9 · 22.74 |
| **ours 8T 16x8, without Y00** | 141.5 | 139.5 | 143.9 | 141.6 | 20.53 | not shown |
| his 8T, without Y00 | 123.9 | 122.1 | 124.2 | 123.4 | 23.56 | 123.4 · 23.56 |

The small differences (0.1 ms) come from averaging per launch here versus per shape in #63. The #63 single-tile ranking reproduces: the top 7 tiles sit within 145.7-149.8 ms of each other (2.8%).

## B3. Best tile per shape: PASS (not an artefact of picking on noise)

Picking the best of 19 tiles on the same launches it's scored on would favour lucky reps. So I picked on one launch and scored on the other two:

| picked on | per-shape picks, scored on the other launches | best single tile, same launches | gain |
|---|---|---|---|
| L1 | 137.8 | 146.1 (16x12) | 8.3 ms |
| L2 | 135.8 | 146.5 (32x7) | 10.7 ms |
| L3 | 135.5 | 145.7 (32x7) | 10.2 ms |

A different tile per shape gains about 6-7% over the best single tile, and the gain holds on launches it wasn't picked on.

## B4. Best tile per n: GAP (5 of 8 winners are inside the noise)

| n | shapes | #63 best | runner-up | gap, ms per pass | best's spread over launches, ms | winner per launch |
|---|---|---|---|---|---|---|
| 4 | 3 | 32x4 | 16x4 | 0.01 | 0.00 | same in all 3 |
| 8 | 1 | 32x4 | 16x10 | 0.14 | 0.06 | same in all 3 |
| 16 | 10 | 32x7 | 32x6 | 0.08 | 0.31 | 32x6, 32x6, 32x7 |
| 32 | 6 | 32x7 | 32x6 | 0.18 | 0.02 | same in all 3 |
| 64 | 10 | 32x6 | 32x7 | 0.14 | 0.21 | 32x6, 16x8, 32x6 |
| 80 | 6 | 16x15 | 16x14 | 0.01 | 0.42 | 16x15, 16x14, 24x8 |
| 128 | 9 | 16x15 | 16x12 | 0.22 | 1.92 | 16x15, 16x15, 16x12 |
| 256 | 5 | 16x14 | 16x12 | 0.03 | 1.02 | 16x14, 16x12, 16x14 |

- Only n = 4, 8 and 32 have a stable winner.
- For n = 16, 64, 80, 128 and 256 the gap to the runner-up is smaller than the winner's own launch spread, and the winner changes between launches.
- What the data supports is a **family**, not one tile: MR = 32 with NR = 4 for n ≤ 8, MR = 32 with NR = 6-7 for n = 16-64, MR = 16 with NR = 12-15 for n ≥ 80.
- #63's table reads as exact winners. The README now says this (fix A5).

## B5. Ours against his: GAP (the headline depends on one unstable shape of his)

- #63 headline: ours 22.17 GOPs, his 16.18. That holds only with his Y00. Without Y00, his GEMM is ahead of every configuration of ours: 123.4 ms against 128.0 ms (ours, best tile per shape) and 141.6 ms (ours at his own tile 16x8). #63 §1 says this, but the README headline table leads with the Y00-driven totals.
- **Tile for tile**, the like-for-like comparison of the two GEMMs (same 16x8 tile, without Y00), his engine is 15% faster: 123.4 against 141.6 ms. Our engine is ahead on the tile only.
- Per shape: ours beats his mean on 25 of 50 shapes with the best tile, and on 12 of 50 at 16x8.

## B6. The two anomalies

**His Y00 (76800x16x27).** Across 9 committed 8-thread launches (#62 and #63), his Y00 runs from 6.7 to 105.5 ms (15.8x). No other shape of his moves more than 1.56x. Two candidate causes, both from his code; neither is proven:
1. **Barrier count.** His `gemm_B3A2C0` ends every `ic` block with `#pragma omp barrier`, and `MC = 256`. Y00 has 300 `ic` blocks per call, so 300 barriers. The next-highest shape has 19. One late thread at a barrier holds up all 8, and his threads aren't bound to cores.
2. **First shape of the process.** Y00 is row 0 of `yolo.dat`, so it takes the first `gemm_rvv` call: a 512 MB `malloc` for `Ac`, the OpenMP thread pool start-up, and the first-touch page faults. With `TIMIN = 2 s` and 20-150 reps this should average out, which makes it less likely than cause 1.

A board test would tell them apart: run `gemm.x` with Y00 moved to a later row, or listed twice. Not run here (no new board runs on his side without the user).

**Our median rep far above our mean at 8T.** Per-rep logs, tile 16x8, launch 1 (B6 in the output file):
- On the affected shapes, 2-10% of reps are slow, and they take 30-46% of the time. Y24: 5.3% of reps take a median 9.87 ms against a 1.36 ms median.
- A slow rep costs a roughly fixed extra 8-13 ms, whatever the shape's own time. That fits one thread losing its core for a time slice while the other seven wait at a barrier.
- Per-core CPU time over the 57 timed runs of #63 (`cpu_times_before/after`): core 0 spent 13.1% in the kernel (`sys`), cores 1-7 spent 0.2-0.5%. Core 0 takes every device interrupt (house rule 1).
- With `OMP_PROC_BIND=close` our thread on core 0 can't move away; his unbound threads can.
- So the likely cause is the core-0 exception that every 8-thread run uses (#56 §5, #63 Q1). Not proven. A run with 7 threads on cores 1-7 (house rule 1, no exception needed) or with binding off would test it.

Effect on the tile answer:
- The per-shape winner is chosen by mean time, which takes in these slow reps. Where slow reps are 30-46% of the time, the winner partly reflects which tile was unlucky.
- Ranking by median rep would remove that, but #63 reports both metrics and ranks by mean (the house metric). This audit keeps that rule and records the effect.

## B7. Fixes made on this branch

- `.scratch/reports/issue-63/README.md`: the headline table gains the "ours at 16x8" rows and a note on Y00, and the per-n tile list is marked as a family with the noise stated (see A5).
- Board correctness pass at 8 threads for all 19 tiles: Part C3.

## B8. Left open (needs the user; each is a board run)

1. His Y00: cause 1 or 2 (B6), one short run of `gemm.x` with Y00 moved.
2. Our slow reps: 8 threads on 0-7 against 7 threads on 1-7, or binding off.
3. Part D: his GEMM rebuilt with his own 32x8 / 16x16 kernels (Part A6).
