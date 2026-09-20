# Scalar-pack sweep, report so far

Generated Sun 20 Sep 2026 16:30. Sweep dir `tests/pure_xdsl_blis_rvv/logs/sweep_scalar_2026-09-20_132532`.

**Scalar-pack runs in: 6 of 16 (partial, the sweep is still running).** Runs still to come, and a run in progress, are not in the numbers.

| | runs | boards | started |
|---|---|---|---|
| scalar pack (this sweep) | 6 | jbpi 3, jbpi2 3 | 13:32 to 16:02 |
| vector pack (overnight 19-20 Sep) | 16 | jbpi 8, jbpi2 8 | 23:06 to 06:36 |

Both sets: uninstrumented build, reference protocol, `tmin` 10 s, `--no-nc-overrides`, core 0, the full `config.json`. e2e GOPs = 2*M*N*K / mean time per whole GEMM call, packing included.

**What is not the same between the two sets:** the vector runs were made overnight (23:00-06:30) and the scalar runs in the daytime, so the time of day, the room, and the board's temperature differ. A gap of a few percent can come from that. A partial sweep may also have run on one board only (see the boards column).

## Per layer

| Layer | Shape | Cfg | vector mean | vector range | vector runs | scalar mean | scalar range | scalar runs | scalar vs vector | verdict | published | scalar vs published | scalar reps per run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 1000x1000x1000 | 32x7 | 8.49 | 8.43-8.55 | 16 | 8.47 | 8.43-8.52 | 6 | -0.2% | within noise | 8.5 | -0.3% | 43 |
| S2 | 2000x2000x2000 | 32x7 | 8.75 | 8.50-8.86 | 16 | 8.62 | 8.51-8.77 | 6 | -1.4% | within noise | 8.7 | -0.9% | 6 |
| S3 | 3000x3000x3000 | 32x7 | 9.05 | 8.92-9.11 | 16 | 8.92 | 8.74-9.09 | 6 | -1.4% | within noise | 8.9 | +0.3% | 2 |
| S4 | 4000x4000x4000 | 32x7 | 8.83 | 8.68-8.96 | 16 | 8.78 | 8.64-8.92 | 6 | -0.6% | within noise | 9.0 | -2.5% | 1 |
| S5 | 5000x5000x5000 | 32x7 | 9.16 | 9.05-9.26 | 16 | 9.08 | 8.97-9.20 | 6 | -0.9% | within noise | 9.0 | +0.9% | 1 |
| B1 | 1024x384x1024 | 32x6 | 8.04 | 7.86-8.15 | 16 | 7.91 | 7.85-7.98 | 6 | -1.7% | within noise | 8.0 | -1.1% | 99 |
| B2 | 384x384x64 | 16x15 | 12.72 | 12.38-12.94 | 16 | 12.24 | 12.11-12.30 | 6 | -3.8% | within noise | 12.5 | -2.1% | 6487 |
| B3 | 64x384x384 | 16x15 | 5.30 | 5.00-5.59 | 16 | 8.22 | 8.09-8.36 | 6 | +55.1% | scalar faster | 8.8 | -6.6% | 4356 |
| B4 | 4096x384x1024 | 32x7 | 7.80 | 7.66-7.90 | 16 | 7.60 | 7.45-7.72 | 6 | -2.6% | within noise | 8.5 | -10.6% | 24 |
| B5 | 1024x384x4096 | 32x6 | 7.82 | 7.60-7.97 | 16 | 7.77 | 7.69-7.86 | 6 | -0.7% | within noise | 7.5 | +3.6% | 24 |

Layers with 1-2 reps per run (S3, S4, S5) have no spread inside a run; their noise is the spread between runs only.

## Per board

Mean e2e GOPs per board and packer. A board with no scalar run yet reads `-`.

| Layer | vector jbpi | vector jbpi2 | scalar jbpi | scalar jbpi2 |
|---|---|---|---|---|
| S1 | 8.51 (n=8) | 8.47 (n=8) | 8.49 (n=3) | 8.45 (n=3) |
| S2 | 8.77 (n=8) | 8.72 (n=8) | 8.63 (n=3) | 8.62 (n=3) |
| S3 | 9.07 (n=8) | 9.03 (n=8) | 8.90 (n=3) | 8.95 (n=3) |
| S4 | 8.84 (n=8) | 8.83 (n=8) | 8.74 (n=3) | 8.82 (n=3) |
| S5 | 9.18 (n=8) | 9.15 (n=8) | 9.07 (n=3) | 9.09 (n=3) |
| B1 | 8.09 (n=8) | 8.00 (n=8) | 7.97 (n=3) | 7.85 (n=3) |
| B2 | 12.72 (n=8) | 12.72 (n=8) | 12.25 (n=3) | 12.24 (n=3) |
| B3 | 5.34 (n=8) | 5.26 (n=8) | 8.19 (n=3) | 8.25 (n=3) |
| B4 | 7.83 (n=8) | 7.78 (n=8) | 7.55 (n=3) | 7.64 (n=3) |
| B5 | 7.86 (n=8) | 7.78 (n=8) | 7.83 (n=3) | 7.71 (n=3) |

## B3, run by run

B3 runs at two speeds (about 4.2 and about 9 GOPs); quote it by mean, the median moves between identical runs.

| packer | run | board | mean | median rep | 5th pct | 95th pct | fastest | reps |
|---|---|---|---|---|---|---|---|---|
| scalar | 13:32 | jbpi | 8.32 | 8.91 | 5.82 | 9.19 | 9.35 | 4406 |
| scalar | 14:02 | jbpi2 | 8.09 | 8.79 | 5.78 | 9.07 | 9.23 | 4285 |
| scalar | 14:32 | jbpi | 8.10 | 8.89 | 5.84 | 9.18 | 9.34 | 4291 |
| scalar | 15:02 | jbpi2 | 8.36 | 8.80 | 5.94 | 9.11 | 9.28 | 4428 |
| scalar | 15:32 | jbpi | 8.16 | 8.84 | 5.87 | 9.17 | 9.35 | 4326 |
| scalar | 16:02 | jbpi2 | 8.30 | 8.85 | 5.98 | 9.12 | 9.28 | 4400 |
| vector | 23:06 | jbpi | 5.34 | 4.48 | 4.26 | 9.63 | 10.05 | 2832 |
| vector | 23:37 | jbpi2 | 5.34 | 4.40 | 4.15 | 9.59 | 10.02 | 2830 |
| vector | 00:06 | jbpi | 5.51 | 4.55 | 4.31 | 9.56 | 9.98 | 2919 |
| vector | 00:36 | jbpi2 | 5.22 | 4.36 | 4.17 | 9.35 | 9.91 | 2767 |
| vector | 01:06 | jbpi | 5.28 | 4.37 | 4.14 | 9.59 | 10.00 | 2798 |
| vector | 01:36 | jbpi2 | 5.17 | 4.43 | 4.23 | 9.33 | 9.90 | 2742 |
| vector | 02:06 | jbpi | 5.50 | 4.57 | 4.30 | 9.53 | 10.02 | 2914 |
| vector | 02:36 | jbpi2 | 5.33 | 4.42 | 4.18 | 9.43 | 9.93 | 2826 |
| vector | 03:06 | jbpi | 5.27 | 4.43 | 4.23 | 9.61 | 10.02 | 2791 |
| vector | 03:36 | jbpi2 | 5.01 | 4.29 | 4.11 | 9.46 | 9.98 | 2654 |
| vector | 04:06 | jbpi | 5.22 | 4.44 | 4.25 | 9.50 | 9.98 | 2765 |
| vector | 04:36 | jbpi2 | 5.00 | 4.28 | 4.09 | 9.43 | 9.95 | 2650 |
| vector | 05:06 | jbpi | 5.22 | 4.43 | 4.23 | 9.37 | 10.00 | 2764 |
| vector | 05:36 | jbpi2 | 5.59 | 5.86 | 4.19 | 9.54 | 10.04 | 2964 |
| vector | 06:06 | jbpi | 5.35 | 4.46 | 4.23 | 9.53 | 9.98 | 2834 |
| vector | 06:36 | jbpi2 | 5.44 | 5.59 | 4.04 | 9.34 | 9.98 | 2880 |
