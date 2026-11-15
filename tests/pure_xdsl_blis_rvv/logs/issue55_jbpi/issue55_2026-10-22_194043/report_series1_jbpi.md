# Issues #54/#55 series issue55_2026-10-22_194043

Build: uninstrumented (INSTRUMENTED=0); protocol: reference, --tmin 10; core 4; --no-nc-overrides (each case's nc is the cache model's)

## Run conditions

- run_2026-10-22_194942: jlei@jbpi, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 1%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-22_195846: jlei@jbpi, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 1% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-22_200751: jlei@jbpi, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 1% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-22_201655: jlei@jbpi, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 1% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-22_202559: jlei@jbpi, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 1% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-22_203503: jlei@jbpi, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 1% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-22_210654: jlei@jbpi, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 1% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-22_211557: jlei@jbpi, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 1% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-22_213911: jlei@jbpi, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 1% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 1% cpu1 0% cpu2 1% cpu3 0% cpu4 100% cpu5 0% cpu6 1% cpu7 0%
- run_2026-10-22_214815: jlei@jbpi, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 2% cpu1 1% cpu2 1% cpu3 72% cpu4 99% cpu5 1% cpu6 18% cpu7 1%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 1% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-22_215718: jlei@jbpi, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 1% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-22_222911: jlei@jbpi, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 1% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-22_223923: jlei@jbpi, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 1% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-22_224826: jlei@jbpi, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 1% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-22_225730: jlei@jbpi, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 1% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-22_230634: jlei@jbpi, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 1% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-22_231537: jlei@jbpi, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 1%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-22_232441: jlei@jbpi, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 1%

Runs left out (status is not ok): jbpi L2 col-gall (discarded-load)

## 1. GEMM arms per tile and shape

### jbpi, 16x6, B3

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 256 | 3 | 6.73 (6.70-6.75) | 6.74 | 0 | 10.00 | 1.4 | RED NOTE: 1 sample(s) under another user's load |
| row-intr | 256 | 3 | 6.27 (6.25-6.29) | 6.28 | 0 | 10.00 | 1.4 | RED NOTE: 1 sample(s) under another user's load |
| scalar | 256 | 3 | 6.15 (6.10-6.18) | 6.18 | 0 | 10.00 | 1.8 | RED NOTE: 1 sample(s) under another user's load |
| col-g1 | 256 | 3 | 6.88 (6.86-6.93) | 6.94 | 0 | 10.00 | 2.7 | RED NOTE: 1 sample(s) under another user's load |
| col-gall | 256 | 3 | 3.91 (3.81-3.98) | 3.75 | 81 | 0.08 | 27.0 | 1 run(s) of this arm discarded; RED NOTE: 1 sample(s) under another user's load |
| col-g1-m4 | 256 | 3 | 6.86 (6.84-6.88) | 6.90 | 0 | 10.00 | 2.5 | RED NOTE: 1 sample(s) under another user's load |

### jbpi, 16x6, B3_K256

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 256 | 3 | 4.69 (4.60-4.77) | 4.91 | 53 | 0.30 | 35.2 | RED NOTE: 1 sample(s) under another user's load |
| row-intr | 256 | 3 | 3.90 (3.84-3.98) | 3.37 | 74 | 0.14 | 34.9 | RED NOTE: 1 sample(s) under another user's load |
| scalar | 256 | 3 | 5.73 (5.38-5.92) | 6.55 | 30 | 0.71 | 21.3 | RED NOTE: 1 sample(s) under another user's load |
| col-g1 | 256 | 3 | 7.62 (7.59-7.64) | 7.58 | 0 | 10.00 | 3.1 | RED NOTE: 1 sample(s) under another user's load |
| col-gall | 256 | 3 | 4.25 (4.10-4.36) | 4.56 | 52 | 0.22 | 30.8 | 1 run(s) of this arm discarded; RED NOTE: 1 sample(s) under another user's load |
| col-g1-m4 | 256 | 3 | 7.66 (7.56-7.79) | 7.62 | 0 | 10.00 | 3.4 | RED NOTE: 1 sample(s) under another user's load |

### jbpi, 16x6, B2

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 64 | 3 | 10.24 (10.18-10.30) | 10.25 | 0 | 10.00 | 0.6 | RED NOTE: 1 sample(s) under another user's load |
| row-intr | 64 | 3 | 10.08 (10.04-10.10) | 10.09 | 0 | 10.00 | 0.6 | RED NOTE: 1 sample(s) under another user's load |
| scalar | 64 | 3 | 9.88 (9.83-9.91) | 9.89 | 0 | 10.00 | 0.6 | RED NOTE: 1 sample(s) under another user's load |
| col-g1 | 64 | 3 | 10.17 (10.08-10.24) | 10.17 | 0 | 10.00 | 0.6 | RED NOTE: 1 sample(s) under another user's load |
| col-gall | 64 | 3 | 10.13 (10.05-10.19) | 10.14 | 0 | 10.00 | 0.6 | 1 run(s) of this arm discarded; RED NOTE: 1 sample(s) under another user's load |
| col-g1-m4 | 64 | 3 | 10.23 (10.18-10.29) | 10.24 | 0 | 10.00 | 0.6 | RED NOTE: 1 sample(s) under another user's load |

### jbpi, 16x6, R24

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 256 | 3 | 9.17 (9.14-9.21) | 9.18 | 0 | 10.01 | 0.5 | RED NOTE: 1 sample(s) under another user's load |
| row-intr | 256 | 3 | 8.92 (8.90-8.93) | 8.92 | 0 | 10.00 | 0.4 | RED NOTE: 1 sample(s) under another user's load |
| scalar | 256 | 3 | 8.87 (8.81-8.92) | 8.87 | 0 | 10.01 | 0.3 | RED NOTE: 1 sample(s) under another user's load |
| col-g1 | 256 | 3 | 9.27 (9.23-9.32) | 9.29 | 0 | 10.01 | 0.8 | RED NOTE: 1 sample(s) under another user's load |
| col-gall | 256 | 3 | 9.04 (8.91-9.14) | 9.11 | 0 | 10.00 | 2.1 | 1 run(s) of this arm discarded; RED NOTE: 1 sample(s) under another user's load |
| col-g1-m4 | 256 | 3 | 9.30 (9.27-9.35) | 9.31 | 0 | 10.01 | 0.7 | RED NOTE: 1 sample(s) under another user's load |

### jbpi, 16x8, B3

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 256 | 3 | 3.55 (3.51-3.58) | 3.33 | 94 | 0.03 | 24.5 | RED NOTE: 1 sample(s) under another user's load |
| row-intr | 256 | 3 | 3.50 (3.49-3.50) | 3.30 | 94 | 0.02 | 24.1 | RED NOTE: 1 sample(s) under another user's load |
| scalar | 256 | 3 | 7.16 (6.91-7.41) | 7.49 | 6 | 5.63 | 9.1 | RED NOTE: 1 sample(s) under another user's load |
| col-g1 | 256 | 3 | 7.07 (7.02-7.11) | 7.11 | 0 | 10.00 | 2.1 | RED NOTE: 1 sample(s) under another user's load |
| col-gall | 256 | 3 | 7.10 (7.04-7.13) | 7.10 | 0 | 10.00 | 1.4 | 1 run(s) of this arm discarded; RED NOTE: 1 sample(s) under another user's load |
| col-g1-m4 | 256 | 3 | 7.08 (7.07-7.09) | 7.11 | 0 | 10.00 | 1.7 | RED NOTE: 1 sample(s) under another user's load |

### jbpi, 16x8, B3_K256

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 256 | 3 | 3.63 (3.62-3.63) | 3.53 | 96 | 0.01 | 20.4 | RED NOTE: 1 sample(s) under another user's load |
| row-intr | 256 | 3 | 3.65 (3.64-3.65) | 3.60 | 97 | 0.01 | 13.7 | RED NOTE: 1 sample(s) under another user's load |
| scalar | 256 | 3 | 6.99 (6.93-7.04) | 7.79 | 21 | 0.99 | 15.8 | RED NOTE: 1 sample(s) under another user's load |
| col-g1 | 256 | 3 | 7.86 (7.86-7.87) | 7.89 | 0 | 10.00 | 1.1 | RED NOTE: 1 sample(s) under another user's load |
| col-gall | 256 | 3 | 6.61 (6.60-6.62) | 6.62 | 0 | 10.00 | 1.5 | 1 run(s) of this arm discarded; RED NOTE: 1 sample(s) under another user's load |
| col-g1-m4 | 256 | 3 | 7.92 (7.86-8.01) | 7.94 | 0 | 10.00 | 1.5 | RED NOTE: 1 sample(s) under another user's load |

### jbpi, 16x8, B2

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 64 | 3 | 10.30 (10.27-10.34) | 10.31 | 0 | 10.00 | 0.6 | RED NOTE: 1 sample(s) under another user's load |
| row-intr | 64 | 3 | 10.11 (10.10-10.13) | 10.12 | 0 | 10.00 | 0.6 | RED NOTE: 1 sample(s) under another user's load |
| scalar | 64 | 3 | 9.86 (9.85-9.88) | 9.87 | 0 | 10.00 | 0.6 | RED NOTE: 1 sample(s) under another user's load |
| col-g1 | 64 | 3 | 10.18 (10.11-10.24) | 10.19 | 0 | 10.00 | 1.0 | RED NOTE: 1 sample(s) under another user's load |
| col-gall | 64 | 3 | 10.16 (10.09-10.23) | 10.16 | 0 | 10.00 | 0.7 | 1 run(s) of this arm discarded; RED NOTE: 1 sample(s) under another user's load |
| col-g1-m4 | 64 | 3 | 10.24 (10.22-10.25) | 10.24 | 0 | 10.00 | 0.6 | RED NOTE: 1 sample(s) under another user's load |

### jbpi, 16x8, R24

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 256 | 3 | 8.81 (8.66-8.94) | 8.92 | 0 | 10.00 | 2.7 | RED NOTE: 1 sample(s) under another user's load |
| row-intr | 256 | 3 | 7.45 (7.05-8.05) | 7.67 | 36 | 3.49 | 13.1 | RED NOTE: 1 sample(s) under another user's load |
| scalar | 256 | 3 | 8.93 (8.88-8.97) | 9.09 | 0 | 10.01 | 3.7 | RED NOTE: 1 sample(s) under another user's load |
| col-g1 | 256 | 3 | 9.16 (9.13-9.17) | 9.15 | 0 | 10.01 | 0.4 | RED NOTE: 1 sample(s) under another user's load |
| col-gall | 256 | 3 | 8.98 (8.96-8.99) | 8.98 | 0 | 10.01 | 0.5 | 1 run(s) of this arm discarded; RED NOTE: 1 sample(s) under another user's load |
| col-g1-m4 | 256 | 3 | 9.16 (9.16-9.17) | 9.16 | 0 | 10.00 | 0.4 | RED NOTE: 1 sample(s) under another user's load |

### jbpi, 16x15, B3

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 128 | 3 | 4.62 (4.35-5.07) | 4.29 | 87 | 0.03 | 22.7 | RED NOTE: 1 sample(s) under another user's load |
| row-intr | 128 | 3 | 4.40 (4.35-4.46) | 4.21 | 86 | 0.06 | 20.3 | RED NOTE: 1 sample(s) under another user's load |
| scalar | 128 | 3 | 6.92 (6.55-7.30) | 7.54 | 44 | 0.30 | 21.4 | RED NOTE: 1 sample(s) under another user's load |
| col-g1 | 128 | 3 | 6.90 (6.89-6.93) | 6.89 | 0 | 10.00 | 0.8 | RED NOTE: 1 sample(s) under another user's load |
| col-gall | 128 | 3 | 6.75 (6.74-6.78) | 6.75 | 0 | 10.00 | 1.0 | 1 run(s) of this arm discarded; RED NOTE: 1 sample(s) under another user's load |
| col-g1-m4 | 128 | 3 | 6.96 (6.87-7.09) | 6.96 | 0 | 6.79 | 1.3 | RED NOTE: 1 sample(s) under another user's load |

### jbpi, 16x15, B3_K256

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 128 | 3 | 4.77 (4.66-4.93) | 4.53 | 91 | 0.02 | 24.6 | RED NOTE: 1 sample(s) under another user's load |
| row-intr | 128 | 3 | 4.59 (4.58-4.59) | 4.45 | 94 | 0.03 | 20.4 | RED NOTE: 1 sample(s) under another user's load |
| scalar | 128 | 3 | 4.88 (4.67-5.18) | 4.46 | 83 | 0.05 | 27.3 | RED NOTE: 1 sample(s) under another user's load |
| col-g1 | 128 | 3 | 8.01 (7.96-8.06) | 8.07 | 0 | 10.00 | 1.4 | RED NOTE: 1 sample(s) under another user's load |
| col-gall | 128 | 3 | 7.47 (7.43-7.49) | 7.48 | 0 | 10.00 | 1.7 | 1 run(s) of this arm discarded; RED NOTE: 1 sample(s) under another user's load |
| col-g1-m4 | 128 | 3 | 7.85 (7.70-7.96) | 7.90 | 0 | 10.00 | 1.9 | RED NOTE: 1 sample(s) under another user's load |

### jbpi, 16x15, B2

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 64 | 3 | 12.59 (12.55-12.64) | 12.61 | 0 | 10.00 | 0.8 | RED NOTE: 1 sample(s) under another user's load |
| row-intr | 64 | 3 | 12.33 (12.31-12.35) | 12.35 | 0 | 10.00 | 0.9 | RED NOTE: 1 sample(s) under another user's load |
| scalar | 64 | 3 | 12.11 (12.06-12.17) | 12.14 | 0 | 8.16 | 1.1 | RED NOTE: 1 sample(s) under another user's load |
| col-g1 | 64 | 3 | 12.44 (12.41-12.47) | 12.46 | 0 | 10.00 | 0.9 | RED NOTE: 1 sample(s) under another user's load |
| col-gall | 64 | 3 | 12.49 (12.34-12.57) | 12.51 | 0 | 10.00 | 0.9 | 1 run(s) of this arm discarded; RED NOTE: 1 sample(s) under another user's load |
| col-g1-m4 | 64 | 3 | 12.44 (12.35-12.48) | 12.47 | 0 | 10.00 | 0.9 | RED NOTE: 1 sample(s) under another user's load |

### jbpi, 16x15, R24

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 128 | 3 | 8.40 (8.21-8.69) | 9.00 | 28 | 1.24 | 11.6 | RED NOTE: 1 sample(s) under another user's load |
| row-intr | 128 | 3 | 7.43 (7.36-7.49) | 6.83 | 58 | 0.79 | 13.1 | RED NOTE: 1 sample(s) under another user's load |
| scalar | 128 | 3 | 9.29 (9.28-9.31) | 9.31 | 0 | 10.00 | 0.7 | RED NOTE: 1 sample(s) under another user's load |
| col-g1 | 128 | 3 | 8.87 (8.87-8.88) | 8.87 | 0 | 10.01 | 0.5 | RED NOTE: 1 sample(s) under another user's load |
| col-gall | 128 | 3 | 8.74 (8.73-8.75) | 8.74 | 0 | 10.01 | 0.6 | 1 run(s) of this arm discarded; RED NOTE: 1 sample(s) under another user's load |
| col-g1-m4 | 128 | 3 | 8.78 (8.57-8.88) | 8.82 | 0 | 10.01 | 1.8 | RED NOTE: 1 sample(s) under another user's load |

## 2. Pack-only loop (no compute; 1 clock read per rep; rep 0 left out of the split)

Pack-only runs left out: jbpi L2 pack-only scalar B3_K256 16x15: discarded-load

### jbpi, 16x15, B3

| variant | kc | launches | median pack (ms) | fast reps: mean (ms) | slow reps: mean (ms) | slow reps % | flags |
|---|---|---|---|---|---|---|---|
| row-intr | 128 | 3 | 1.1511 | 0.6506 | 1.1677 | 97 |  |
| scalar | 128 | 3 | 1.7153 | 0.7110 | 1.6595 | 82 |  |
| col-g1 | 128 | 3 | 0.9004 | 0.9172 | no slow reps | 0 |  |
| col-gall | 128 | 3 | 1.3827 | 1.4032 | no slow reps | 0 |  |
| col-g1-m4 | 128 | 3 | 0.8670 | 0.8821 | no slow reps | 0 |  |

### jbpi, 16x15, B3_K256

| variant | kc | launches | median pack (ms) | fast reps: mean (ms) | slow reps: mean (ms) | slow reps % | flags |
|---|---|---|---|---|---|---|---|
| row-intr | 128 | 3 | 0.5486 | 0.4426 | 0.7712 | 31 |  |
| scalar | 128 | 3 | 0.9492 | 0.4873 | 0.9521 | 91 |  |
| col-g1 | 128 | 3 | 0.4380 | 0.4507 | no slow reps | 0 |  |
| col-gall | 128 | 3 | 0.7395 | 0.7578 | no slow reps | 0 |  |
| col-g1-m4 | 128 | 3 | 0.3774 | 0.3882 | no slow reps | 0 |  |

### jbpi, 16x8, B3

| variant | kc | launches | median pack (ms) | fast reps: mean (ms) | slow reps: mean (ms) | slow reps % | flags |
|---|---|---|---|---|---|---|---|
| row-intr | 256 | 3 | 1.2520 | 0.8531 | 1.3797 | 81 |  |
| scalar | 256 | 3 | 0.7986 | 0.7178 | 1.3140 | 32 |  |
| col-g1 | 256 | 3 | 0.6304 | 0.6362 | 0.9411 | 5 |  |
| col-gall | 256 | 3 | 1.1100 | 1.1134 | no slow reps | 0 |  |
| col-g1-m4 | 256 | 3 | 0.5884 | 0.5956 | 0.8578 | 6 |  |

### jbpi, 16x8, B3_K256

| variant | kc | launches | median pack (ms) | fast reps: mean (ms) | slow reps: mean (ms) | slow reps % | flags |
|---|---|---|---|---|---|---|---|
| row-intr | 256 | 3 | 0.2171 | 0.2179 | 0.3191 | 16 |  |
| scalar | 256 | 3 | 0.3797 | 0.3862 | no slow reps | 0 |  |
| col-g1 | 256 | 3 | 0.3013 | 0.3042 | 0.4851 | 5 |  |
| col-gall | 256 | 3 | 0.4410 | 0.4462 | 1.3382 | 0 |  |
| col-g1-m4 | 256 | 3 | 0.2651 | 0.2683 | 0.4704 | 5 |  |

RED NOTE: 1 run(s) were discarded because the board was busy after the run; they are left out above.
