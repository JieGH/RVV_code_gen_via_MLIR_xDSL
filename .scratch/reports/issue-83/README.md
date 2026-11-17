# Issue #83: vector packing against scalar packing

The YOLO, BERT and Square sweeps (#61, #63, #71) were run again with one change: the **scalar packer** instead of the vector packer. Tiles, blocking, thread split, shapes, settings and run order stay as they were.

- **Brief (PDF):** [`report/issue83_brief.pdf`](report/issue83_brief.pdf) ([LaTeX](report/issue83_brief.tex))
- **All tables:** [`report/tables.md`](report/tables.md)
- **1T against 8T per shape (added 2026-11-17):** brief Part 1b, one table each for YOLO and BERT/Square: vector and scalar GOPs (by mean time / by median rep) at 1T and 8T, faster packer in bold, 8T/1T per packer.
- **Script:** [`report/make_tables_and_figures.py`](report/make_tables_and_figures.py), run from `tests/pure_xdsl_blis_rvv/logs`. It writes `tables.md` and the chart data (`report/data/*.csv`); the brief draws the charts in LaTeX.

## Answer

- **BERT and Square:** the packer makes no measurable difference. At 8 threads, scalar reaches 0.987 of vector GOPs (median over 10 shapes). No shape moves by more than 3.2%, and every shape keeps its best tile.
- **YOLO:** scalar is slower on almost every layer. It reaches 0.930 of vector at 1 thread and 0.946 at 8 threads (median over 50 layers).
  - **Worst where nc is small:** at nc = 4 it reaches 0.59-0.71 at 1 thread and 0.67-0.83 at 8 threads.
  - **Best tile:** it changes on 14 layers at 1 thread and 17 at 8 threads.
- **Why:** each packed element of A is used n times, so the pack cost is spread over n uses.
  - **Small n:** packing takes 50-83% of the time (T4, below), and the scalar packer's extra pack time shows directly.
  - **Large n (BERT, Square):** packing takes 7-19%, and scalar packing costs only 7-13% more there, so the total moves 1-2%.
- **Check:** T4's pack/compute split predicts the measured 1-thread GOPs ratio within 1% on 9 of 11 shapes (§ Part 2).
- **Limit:** this compares two packers. It does not test running with no packing at all.

## Glossary

| term | meaning |
|---|---|
| **packer** | Code that copies a block of A or B into a contiguous buffer (Ac, Bc) before the microkernel reads it. |
| **vector packer** | Copies with RVV vector loads and stores: one instruction moves a whole register (8 fp32 at VLEN 256). Used in every sweep before #83. |
| **scalar packer** | Copies one float per load/store pair (`--scalar-pack`). Same layout, same buffers. |
| **tile MRxNR** | The microkernel's block of C: MR rows by NR columns. 19 Candidate Tiles are tried per shape. |
| **best tile** | For one shape, the tile with the highest GOPs by mean time, averaged over the launches. |
| **m x n x k** | GEMM size: C (m x n) += A (m x k) · B (k x n). |
| **mc, nc, kc** | Cache blocking from the analytical model (`get_optim_mc_nc_kc`, per shape and tile): A is packed in mc x kc blocks, B in kc x nc blocks. nc = min(n, model value). n itself is the problem size, not a cache parameter. |
| **reuse nc** | How many times each packed element of A is used: the packed A block meets all nc columns of the current B block, then A is packed again for the next one. Packing A costs about 1/(2·nc) of the flops. nc = n for every YOLO and BERT shape; Square has nc = 1000-1536, below its n = 1000-5000. |
| **GOPs by mean time** | 2·m·n·k / mean time per call. The main metric. |
| **GOPs by median rep** | The median over all reps of each rep's GOPs. Less sensitive to a few slow reps. |
| **launch** | One full run of the test: every tile on every shape. A series has 3 launches ("launch 1/2/3"). They are the same test; only the starting tile of the rotation and the time slot differ. |
| **starting tile / rotation** | The tile list is rotated per launch (launch 1 starts at 8x4, launch 2 at 16x8, launch 3 at 24x8), so no tile always runs first or last. |
| **launch spread** | (max - min) / mean of a shape's best-tile GOPs over its launches. Shows run-to-run noise. |
| **A-B-B-A** | Run order that interleaves two sides (e.g. vector, scalar, scalar, vector), so slow drift hits both equally. |
| **1T / 8T** | 1 thread pinned to core 4 / 8 threads on cores 0-7 (core 0 by the user's 8-thread exception, #63 §2). |
| **uninstrumented build** | `INSTRUMENTED=0`: no timers inside the GEMM. All GOPs in this report come from it, except T4. |
| **instrumented build** | `--instr`, DEBUG: timers around pack and compute. Timers cost 11-17%, so only its time shares are used. |
| **reference protocol** | No warm-up; reps repeat until `--tmin` (2 s) has passed. Same as the reference design. |
| **warm-up protocol** | Warm-up first, then a fixed rep count (200 here). Used by the instrumented build. |
| **oracle** | The naive scalar GEMM that checks C (correctness passes only). |

## Runs

All on the SpacemiT X60 boards (VLEN 256). Uninstrumented build, reference protocol, `--tmin 2`, `--no-nc-overrides`, 19 tiles rotated per launch, idle board (other users checked before every launch).

| run | what | board | logs (`tests/pure_xdsl_blis_rvv/logs/`) |
|---|---|---|---|
| C1 / C2 | YOLO-like correctness, scalar packer, 1T / 8T, oracle on | jbpi | `run_2026-11-17_115404` (76/76 PASS), `run_2026-11-17_122902` (76/76 PASS) |
| C3 / C4 | paper-like correctness, 1T / 8T | jbpi2 | `run_2026-11-17_115336` (57/57 PASS), `run_2026-11-17_122805` (57/57 PASS) |
| T1 | YOLO 1T, scalar, 3 launches | jbpi | `run_2026-11-17_171901`, `_182951`, `_194103` |
| T2 | YOLO 8T, scalar, against Héctor's `gemm.x` (his side unchanged) | jbpi | ours `run_2026-11-17_133645`, `_145239`, `_160305`; his `hector_gemm_2026-11-17_133951`, `_134308`, `_160610` |
| T3 | BERT/Square 1T and 8T, scalar | jbpi2 | 1T `run_2026-11-17_133114`, `_160334`, `_170643`; 8T `_141548`, `_150024`, `_175117` |
| T4 | pack/compute split, instrumented, 1T core 4, 4 launches vector, scalar, scalar, vector | jbpi2 | `run_2026-11-17_182435`, `_185726`, `_193019`, `_200306` |
| vector side | the committed #61, #63, #71 runs | jbpi | YOLO 1T `run_2026-11-15_220501`, `_231549`, `run_2026-11-16_002638`; YOLO 8T `run_2026-11-16_222551`, `_234245`, `run_2026-11-17_005337`; paper 1T `run_2026-11-17_020326`, `_043612`, `_053929`; paper 8T `_024808`, `_033254`, `_062411` |

Series folders (reports, heatmaps, per-launch logs): [`yolo/`](yolo/) (T1, T2), [`paper/`](paper/) (T3). Driver logs (the C1-C4 PASS lines, launch times): [`driver_logs/`](driver_logs/). The early figures from 17 Nov (partial logs) are in [`preliminary/`](preliminary/).

## Part 1: head to head

GOPs by mean time / by median rep, best tile, mean of 3 launches per side. Full per-shape tables, with tiles and launch spread: [`report/tables.md`](report/tables.md).

| test | shapes | scalar / vector, by mean time: median (min-max) | by median rep: median (min-max) | best tile changed | median launch spread, vector / scalar |
|---|---|---|---|---|---|
| YOLO 1T | 50 | 0.930 (0.589-1.022) | 0.932 (0.589-1.043) | 14 | 1.2% / 1.0% |
| YOLO 8T | 50 | 0.946 (0.671-0.985) | 0.946 (0.674-0.986) | 17 | 2.0% / 1.6% |
| BERT/Square 1T | 10 | 0.983 (0.961-1.194) | 0.982 (0.961-1.223) | 1 | 1.0% / 0.6% |
| BERT/Square 8T | 10 | 0.987 (0.968-1.012) | 0.993 (0.952-1.012) | 0 | 1.8% / 3.5% |

BERT and Square, GOPs by mean time / by median rep:

| shape | m x n x k | 1T vector | 1T scalar | 8T vector | 8T scalar |
|---|---|---|---|---|---|
| B1 | 1024x384x1024 | 8.06 / 8.06 | 7.92 / 7.92 | 47.52 / 47.81 | 47.68 / 47.96 |
| B2 | 384x384x64 | 12.58 / 12.61 | 12.10 / 12.13 | 85.06 / 86.31 | 82.35 / 83.60 |
| B3 | 64x384x384 | 7.24 / 7.23 | 8.64 / 8.84 | 28.14 / 28.08 | 27.67 / 27.65 |
| B4 | 4096x384x1024 | 8.12 / 8.12 | 7.89 / 7.89 | 49.35 / 49.57 | 48.91 / 49.19 |
| B5 | 1024x384x4096 | 7.80 / 7.80 | 7.66 / 7.66 | 43.65 / 43.90 | 42.29 / 43.93 |
| S1 | 1000^3 | 8.84 / 8.84 | 8.79 / 8.79 | 47.92 / 49.80 | 46.53 / 49.50 |
| S2 | 2000^3 | 9.28 / 9.28 | 9.06 / 9.06 | 48.12 / 50.27 | 47.90 / 47.88 |
| S3 | 3000^3 | 9.54 / 9.54 | 9.44 / 9.44 | 49.85 / 51.16 | 49.53 / 51.44 |
| S4 | 4000^3 | 9.40 / 9.40 | 9.21 / 9.21 | 50.99 / 50.99 | 49.55 / 49.55 |
| S5 | 5000^3 | 9.66 / 9.66 | 9.54 / 9.54 | 46.83 / 46.83 | 47.39 / 47.39 |

B3 at 1T is noise: its vector launches gave 8.38, 8.05 and 5.29 GOPs (spread 43%).

YOLO, the layers with the largest gap (GOPs by mean time; all 50 in [`report/tables.md`](report/tables.md)):

| layer | m x n x k | 1T vector | 1T scalar | 8T vector | 8T scalar |
|---|---|---|---|---|---|
| Y14 | 4800x4x16 | 2.48 | 1.49 | 13.45 | 9.41 |
| Y30 | 1200x4x16 | 2.69 | 1.92 | 13.85 | 9.29 |
| Y47 | 300x4x16 | 2.96 | 1.74 | 5.37 | 4.44 |
| Y40 | 300x16x144 | 7.08 | 5.55 | 24.62 | 19.90 |
| Y10 | 4800x16x32 | 1.83 | 1.65 | 23.64 | 19.71 |
| Y29 | 1200x32x64 | 4.25 | 3.97 | 44.56 | 37.84 |
| Y32 | 1200x64x64 | 5.97 | 5.49 | 57.97 | 50.40 |

Y10 at 8T: the vector launches spread 20%, so its gap is uncertain.

Charts (vector graphics, drawn by LaTeX/pgfplots from [`report/data/`](report/data/)), in the [brief](report/issue83_brief.pdf), Part 1: one bar chart per test, on wide landscape pages. Each shape is one row, with GOPs along x. The bar is the mean of 3 launches, the whisker shows launch min-max, and the value is printed past the whisker. YOLO takes 4 pages per thread count (13 layers each), with the same GOPs scale on every page. Per-launch values: `report/data/launches_*.csv`.

## Part 2: why the shape and the reuse decide

### The argument

A GEMM packs a block of A (mc x kc), then multiplies each packed element by the nc columns of the current B block. Packing costs one copy per element; compute costs 2·nc flops per element. So the pack cost, relative to the work, falls as 1/nc.

- **What "n" means here:** the reuse is **nc**, the cache-blocking width of B (nc = min(n, model)), not the problem size n. For every YOLO and BERT shape, nc = n. For Square, nc = 1000-1536 while n = 1000-5000. B is packed once per kc x nc block and used by all m rows, so its weight falls as 1/m, which is small here (m ≥ 64).
- **BERT (nc = 384) and Square (nc = 1000-1536):** each packed element is used hundreds of times. The packer's speed hardly reaches the total.
- **YOLO:** most layers have nc = 16-128, and three have nc = 4 (Y14, Y30, Y47). Each packed element is used only a few times, so packing is a large share of the time, and a slower packer shows almost one to one.
- **Small k makes it worse:** with k = 16 (the nc = 4 layers), each call is short (microseconds), so fixed pack overheads weigh more.

### Measured: scalar/vector by nc (median per nc group, shapes in brackets)

| test | nc=4 | 8 | 16 | 32 | 64 | 80 | 128 | 256 | 384 | 1000-1536 |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLO 1T | 0.601 (3) | 0.894 (1) | 0.906 (10) | 0.928 (6) | 0.941 (10) | 0.933 (6) | 0.945 (9) | 0.968 (5) | | |
| YOLO 8T | 0.699 (3) | 0.939 (1) | 0.939 (10) | 0.947 (6) | 0.954 (10) | 0.925 (6) | 0.951 (9) | 0.966 (5) | | |
| BERT/Square 1T | | | | | | | | | 0.982 (5) | 0.976-0.995 (5) |
| BERT/Square 8T | | | | | | | | | 0.983 (5) | 0.971-1.012 (5) |

The ratio climbs steadily with nc. YOLO at nc = 256 (0.97) reaches what BERT shows at nc = 384 (0.98). At 8 threads, the nc = 4 gap is smaller (0.70 against 0.60). Each thread packs only its own rows of A, so the pack work per thread is smaller.

Chart: scalar/vector against n, 1T and 8T, in the [brief](report/issue83_brief.pdf), Part 2.

### Which packer gaps are real

For each shape: the packer gap (|vector - scalar| mean GOPs) against the launch jitter (the wider of the two sides' max - min over 3 launches). A gap larger than its jitter means the two sides' launches don't overlap. Chart: "Packer gap against launch jitter" in the brief (log-log, one dot per shape).

| test | gap larger than jitter | of which scalar faster |
|---|---|---|
| YOLO 1T | 47 of 50 | 0 |
| YOLO 8T | 42 of 50 | 0 |
| BERT/Square 1T | 7 of 10 (gaps 1-2.4%, 0.05-0.23 GOPs) | 0 |
| BERT/Square 8T | 1 of 10 | 0 |

- **YOLO:** the scalar loss is real on almost every layer.
- **BERT/Square at 8T:** the gap is lost in the jitter.
- **BERT/Square at 1T:** a steady 1-2% scalar loss sits above the launch jitter. But the vector side ran on `jbpi` two days before the scalar side ran on `jbpi2`, and that day/board gap isn't measured inside a launch. So 1-2% is not called a packer difference (see Limits).

### T4: how much of the time is packing

Instrumented build, warm-up protocol, 200 reps, 1T core 4, jbpi2, 4 launches (vector, scalar, scalar, vector). Each shape runs on its 1T best tile. The pack time covers A and B packing in the timed loop. Times are means over the two launches of each packer.

| shape | m x n x k | tile | vector: pack ms / compute ms (pack share) | scalar: pack ms / compute ms (pack share) | scalar pack time / vector pack time | predicted scalar/vector | measured scalar/vector (1T, uninstrumented) |
|---|---|---|---|---|---|---|---|
| Y14 | 4800x4x16 | 24x4 | 0.404 / 0.136 (75%) | 0.642 / 0.133 (83%) | 1.59 | 0.697 | 0.601 |
| Y30 | 1200x4x16 | 32x4 | 0.040 / 0.022 (65%) | 0.062 / 0.022 (74%) | 1.58 | 0.738 | 0.712 |
| Y47 | 300x4x16 | 16x4 | 0.008 / 0.008 (51%) | 0.016 / 0.008 (66%) | 1.94 | 0.667 | 0.589 |
| Y10 | 4800x16x32 | 24x8 | 2.114 / 0.633 (77%) | 2.431 / 0.626 (80%) | 1.15 | 0.899 | 0.900 |
| Y40 | 300x16x144 | 24x8 | 0.072 / 0.132 (36%) | 0.128 / 0.129 (50%) | 1.76 | 0.794 | 0.784 |
| Y29 | 1200x32x64 | 32x4 | 0.726 / 0.582 (56%) | 0.845 / 0.568 (60%) | 1.16 | 0.926 | 0.934 |
| Y19 | 4800x80x64 | 32x7 | 4.483 / 4.610 (49%) | 5.051 / 4.590 (52%) | 1.13 | 0.943 | 0.945 |
| Y42 | 300x256x1152 | 24x8 | 5.492 / 18.419 (23%) | 6.125 / 18.489 (25%) | 1.12 | 0.971 | 0.968 |
| B1 | 1024x384x1024 | 32x7 | 18.213 / 89.690 (17%) | 20.571 / 88.815 (19%) | 1.13 | 0.986 | 0.983 |
| B2 | 384x384x64 | 16x15 | 0.104 / 1.491 (7%) | 0.162 / 1.482 (10%) | 1.55 | 0.970 | 0.961 |
| S1 | 1000^3 | 32x7 | 21.125 / 224.134 (9%) | 22.704 / 223.544 (9%) | 1.07 | 0.996 | 0.995 |

"Predicted" = (vector pack + compute) / (scalar pack + compute): the slowdown if only the pack time changes. Compute time is the same for both packers (within 2.5%).

What T4 shows:

- **Pack share follows nc.** Packing takes 49-83% of the time at nc = 4-80 (vector packer) and 7-23% at nc = 256-1000. The share is far above the 1/(2·nc) flop ratio because packing is memory-bound and the microkernel is not. The order still follows n.
- **The scalar packer costs 1.07-1.16x the vector pack time** on large, memory-bound packs (Y10, Y19, Y29, Y42, B1, S1). It costs **1.55-1.94x** on small packs that stay in cache (Y14, Y30, Y47, Y40, B2). There, the copy instructions are the limit, and the vector packer moves 8 floats per instruction.
- **Total effect = pack share x extra pack cost.** BERT/Square: a 7-19% share times 1.07-1.13 gives about 1-2%. That is inside the noise, which is why the packer "doesn't matter" there. YOLO nc = 4: a 51-75% share times 1.6-1.9 gives 26-40%.
- **The prediction holds within 1% on 9 of 11 shapes.** The two misses are nc = 4 shapes (Y14, Y47), where the measured loss is larger than predicted. Their calls take 16-540 µs, so the timers (11-17%) distort the split most, and at 1T the best tile changes for both.
- **This explains** why YOLO at nc = 128-256 still loses 3-5% while BERT loses about 1-2%. Y42 (nc = 256) spends 23% of its time packing, against 9-17% for S1 and B1.

Charts in the [brief](report/issue83_brief.pdf), Part 2:
- **"Where the time goes":** pack + compute time per shape, both packers, as % of the vector packer's total. Only the pack segment grows.
- **"Does the pack share explain the slowdown?":** predicted against measured scalar/vector, with a diagonal line.

### Threads

At 8 threads the YOLO gap is a little smaller (median 0.946 against 0.930). The nc = 4 layers move most: Y14 goes from 0.60 to 0.70, Y47 from 0.59 to 0.83. Each thread packs its own part of A in parallel, so pack time per thread falls about 8x, while the B pack is split by panels. The 8-thread path has no timers (#56), so this part is inferred, not measured.

## Limits

- **Different days.** The vector runs are from 15-17 Nov, the scalar runs from 17 Nov: same build and protocol, but not interleaved in one launch. Gaps under about 3% are not called a difference. Day drift check: Héctor's unchanged `gemm.x` ran on jbpi on 16 Nov (#63) and 17 Nov (T2). Per layer, 17 Nov over 16 Nov has a median of 0.991 (sum of the 50 layers: 1223.5 → 1195.9 GFLOPS, mean of 3 launches each).
- **Different boards for the paper part.** Vector ran on jbpi (#71), scalar on jbpi2. Same SoC, but a board-to-board gap adds to the day gap. YOLO ran on jbpi both times.
- **T4 uses the instrumented build.** Only its time shares and pack-time ratios are used, not its GOPs. It ran on jbpi2, while YOLO 1T ran on jbpi.
- **Two packers, not packing against no packing.** These runs don't measure what skipping packing would gain or lose.
- **Correctness passes (C1-C4)** ran on small shape sets under the oracle limit (B2 and smaller), at 1 and 8 threads. All passed.
