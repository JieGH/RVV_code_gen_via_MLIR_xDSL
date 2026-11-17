# Issue #71: best tile at 1 and 8 threads on the paper's shapes (2026-11-17)

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/71 · issue write-up: `.scratch/issues/0047-paper-shapes-thread-tile-sweep.md`.

**What ran:** **this repo's current design** (not the paper's xDSL code), on the paper's Table 1 shapes (arXiv 2603.17800v1: BERT B1-B5, Square S1-S5), 19 Candidate Tiles, on `jbpi`. 1 thread on core 4, 8 threads on cores 0-7 (core 0: the user's 8-thread exception). Uninstrumented, reference protocol, `--tmin 2`, `--no-nc-overrides`, vector packer. Order 1T, 8T, 8T, 1T, 1T, 8T, 3 launches each, 01:00 → 06:26. No other user's job on the board.

**Not comparable to the paper's GFLOPS:** the paper measured on a BPI-F3 with its own xDSL code and protocol. Only the tile choices are compared here.

## Answer

| | 1 thread (core 4) | 8 threads (cores 0-7) |
|---|---|---|
| best single tile over the 10 shapes | **32x7**, 48.0 s per pass, 9.53 GOPs | **16x8**, 9.47 s per pass, 48.27 GOPs |
| best per shape | 32x7 on 9 shapes, 16x15 on B2 | 16x8 on 8 shapes, 16x12 on B2, 8x24 on B3 |
| the paper's 32x7 over the pass | 9.53 GOPs | 44.07 GOPs (4th) |
| the paper's 16x15 over the pass | 7.47 GOPs | 44.96 GOPs (2nd) |

- **At 1 thread the paper's picks hold:** 32x7 for the squares and most BERT shapes, 16x15 for B2.
- **At 8 threads they don't:** 16x8 wins every shape except B2 (16x12 81.99, 16x15 81.12 GOPs) and B3 (8x24). Over the pass: 16x8 48.27 GOPs, 32x7 44.07 GOPs.
- Best tile over the pass: 9.53 GOPs at 1 thread, 48.27 GOPs at 8 threads.

## Per shape, GOPs by mean time (avg over 3 launches)

| shape | 1T best | 1T 32x7 | 1T 32x6 | 1T 16x15 | 8T best | 8T 16x8 | 8T 32x7 | 8T 32x6 | 8T 16x15 |
|---|---|---|---|---|---|---|---|---|---|
| B1 | 32x7 8.06 | 8.06 | 7.92 | 6.30 | 16x8 47.44 | 47.44 | 42.31 | 42.30 | 38.77 |
| B2 | 16x15 12.58 | 11.76 | 10.60 | 12.58 | 16x12 81.99 | 76.65 | 59.13 | 63.07 | 81.12 |
| B3 | 32x7 7.01 | 7.01 | 6.63 | 4.98 | 8x24 27.78 | 23.30 | 16.59 | 16.36 | 24.80 |
| B4 | 32x7 8.12 | 8.12 | 7.81 | 6.56 | 16x8 49.33 | 49.33 | 42.84 | 42.32 | 43.45 |
| B5 | 32x7 7.80 | 7.80 | 7.68 | 6.03 | 16x8 43.64 | 43.64 | 40.28 | 39.97 | 36.93 |
| S1 | 32x7 8.84 | 8.84 | 8.30 | 7.24 | 16x8 44.70 | 44.70 | 44.21 | 43.34 | 42.33 |
| S2 | 32x7 9.28 | 9.28 | 8.63 | 7.78 | 16x8 47.33 | 47.33 | 43.38 | 40.82 | 45.23 |
| S3 | 32x7 9.54 | 9.54 | 8.81 | 7.42 | 16x8 49.76 | 49.76 | 44.79 | 42.86 | 44.36 |
| S4 | 32x7 9.40 | 9.40 | 8.74 | 7.74 | 16x8 50.99 | 50.99 | 43.50 | 41.22 | 45.26 |
| S5 | 32x7 9.66 | 9.66 | 8.93 | 7.36 | 16x8 46.83 | 46.83 | 44.35 | 43.29 | 45.12 |

By median rep the bests are the same except: 1T B3 → 16x12 (7.23, launches 4.15-8.87); 8T B2 → 16x15 (86.31). At 8T, S1 and S2 by median rep are 49.80 and 50.15 GOPs (mean 44.70, 47.33). The full reports, with the median-rep columns and the 19-tile grids, are in `report/`.

## Limits

- The 19 tiles are the YOLO candidate list (#59), not the paper's full mr x nr heatmap.
- S5 (5000^3) gets few reps per launch at 1 thread with `--tmin 2`.
- 8-thread numbers include core 0, which takes the board's interrupts.

## Folders

| folder | what |
|---|---|
| `report/` | `tile_search_1t_3launches.md`, `tile_search_8t_3launches.md` (`search_report.py` over the 3 launches of each), `heatmap_1t.png`, `heatmap_8t.png` |
| `yolo_sweep_2026-11-17_*` | one series folder per launch (plan, config, launch log, one-launch report) |

Run folders (`tests/pure_xdsl_blis_rvv/logs/`): 1T `run_2026-11-17_020326`, `run_2026-11-17_043612`, `run_2026-11-17_053929`; 8T `run_2026-11-17_024808`, `run_2026-11-17_033254`, `run_2026-11-17_062411`.
