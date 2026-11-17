# Issue #63: best tile per YOLO shape at 8 threads, and our GEMM against Héctor's GEMMLib (2026-11-16/17)

**Board:** `jbpi`, one board for the whole series. No other user's job before or after any launch.
**Ours:** uninstrumented, reference protocol, `--tmin 2`, `--no-nc-overrides`, vector packer, `--threads 8` on **cores 0-7** (`OMP_PROC_BIND=close OMP_PLACES=cores`). 19 Candidate Tiles x 50 shapes (`datasets/yolo26_k1.dat`), rows rotated per launch.
**His:** GEMMLib_yolo at `b292c92`, unedited, `gemm.bin` command line, `TIMIN=2.0`, 8 threads on **cores 0-7**, no `OMP_PROC_BIND` (as he runs it). His tile: 16x8, 16x4 at n = 4.
**Core 0 is used on both sides:** the user's exception for 8-thread runs (#63 §2).
**Order:** ours L1, his, his, ours L2, ours L3, his (A-B-B-A), 21:18 → 01:00.
**1-thread column:** #61's committed series (core 4, same settings, 3 launches on jbpi, 2026-11-15/16).

Run folders: ours `logs/run_2026-11-16_222551`, `run_2026-11-16_234245`, `run_2026-11-17_005337`; his `logs/hector_gemm_2026-11-16_222943`, `_223301`, `hector_gemm_2026-11-17_005723`.

## 1. One pass over the 50 shapes (n = 1 left out, each shape once)

| what | ms per pass | GOPs |
|---|---|---|
| ours 1T, best single tile 24x8 (#61) | 698.6 | 4.26 |
| ours 8T, best single tile **16x12** | 145.6 | 20.43 |
| ours 8T, best tile per shape | 134.1 | 22.17 |
| his 8T (avg of 3 launches) | 183.7 | 16.18 |
| ours 8T best per shape, **without Y00** | 127.9 | 22.74 |
| his 8T, **without Y00** | 123.4 | 23.56 |

GOPs by mean time. The 8T single-tile ranking is tight: 16x12, 32x7, 16x14, 16x10, 16x15, 32x6 and 16x8 are all within 145.6-149.6 ms.

**Y00 (76800x16x27) decides the totals.** His Y00 took 12.9, 62.6 and 105.5 ms in his three launches (5.14, 1.06, 0.63 GOPs). Without Y00, his three passes are steady (122.1-124.2 ms) and **his GEMM is faster than ours (123.4 against 127.9 ms)**. Our Y00 is steady at 10.6 GOPs (32x4). In #62 his Y00 on jbpi was 4.61 GOPs at 8T; these runs do not show why it swings.

## 2. Best tile per n at 8 threads, beside 1 thread

| n | shapes | 1T best (#61) | 1T GOPs | 8T best | 8T GOPs |
|---|---|---|---|---|---|
| 4 | 3 | 24x4 | 2.50 | 32x4 | 12.24 |
| 8 | 1 | 24x9 | 1.12 | 32x4 | 6.04 |
| 16 | 10 | 24x9 | 2.12 | 32x7 | 10.09 |
| 32 | 6 | 24x9 | 3.21 | 32x7 | 17.68 |
| 64 | 10 | 24x8 | 4.97 | 32x6 | 25.24 |
| 80 | 6 | 32x7 | 5.90 | 16x15 | 29.88 |
| 128 | 9 | 32x7 | 6.61 | 16x15 | 30.60 |
| 256 | 5 | 24x8 | 7.60 | 16x14 | 29.34 |

The best tile changes with threads: MR = 24 wins at 1 thread; at 8 threads MR = 32 wins for n ≤ 64 and MR = 16 with wide NR (14-15) for n ≥ 80.

## 3. Per shape

GOPs: "by mean time" is avg over launches (lowest-highest launch); "by median rep" is the median rep of each launch, averaged. His driver gives the mean only.

Ours beats his on 25 of 50 shapes by mean time, and on 39 of 50 by median rep.

See [`issue-63-figs/tables.md`](issue-63-figs/tables.md) for all 50 rows. Rows worth a look:

| shape | m x n x k | ours 1T best | ours 8T best, mean | ours 8T best, median rep | his 8T, mean |
|---|---|---|---|---|---|
| Y00 | 76800x16x27 | 32x4 2.00 | 32x4 10.64 | 32x4 10.74 | 2.28 (0.63-5.14) |
| Y04 | 19200x64x48 | 24x9 4.98 | 32x7 28.35 | 32x7 28.52 | 25.17 |
| Y07 | 4800x128x96 | 16x15 7.15 | 16x14 44.41 | 16x14 44.92 | 36.34 |
| Y32 | 1200x64x64 | 32x6 5.96 | 16x14 38.19 | 16x14 61.09 | 45.56 |
| Y34 | 1200x80x80 | 32x7 6.05 | 16x15 37.47 | 16x10 60.14 | 42.31 |

On Y32 and Y34 his mean beats our mean, but our median rep beats his mean (§4).

## 4. Mean time against median rep at 8 threads

On many mid-size shapes our median rep is far above our mean (Y32: 61.09 against 38.19 GOPs; Y34: 60.14 against 37.47). Some reps are much slower than the typical one, and the mean pays for them. At 1 thread (#61) the two metrics agree within about 0.1%. The slow reps were not studied here.

## 5. Limits

- One board, jbpi. 8-thread numbers include core 0 (interrupts).
- His 8T launches ran without `OMP_PROC_BIND`; ours with `close`/`cores`. Both as each side runs normally.
- Ours: whole GEMM call, packing included. His: his driver's mean time per call.
- Each shape is counted once; how often YOLO runs each shape per image is not known.

## 6. Files

- [`tile_search_8t_3launches.md`](tile_search_8t_3launches.md): `search_report.py` over our 3 launches (per shape, per n, the 19-tile grid). Heatmap: [`issue-63-figs/heatmap_8t.png`](issue-63-figs/heatmap_8t.png).
- [`tile_search_1t_issue61_3launches.md`](tile_search_1t_issue61_3launches.md): the same over #61's 3 launches.
- [`hector_8t_3launches.md`](hector_8t_3launches.md): `hector_gemm_report.py` over his 3 launches.
- `../yolo_sweep_*` and `../hector_gemm_*`: the per-launch series folders.
- Tables built by `.scratch/reports/issue-63-figs/tables.py`.
