# Issues #54/#55 series issue55_2026-10-23_100122

Build: uninstrumented (INSTRUMENTED=0); protocol: reference, --tmin 10; core 4; --no-nc-overrides (each case's nc is the cache model's)

## Run conditions

- run_2026-10-23_101023: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 1% cpu1 2% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-23_101926: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 1% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-23_102829: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 1% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-23_103733: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 1% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-23_104636: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 1%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 1%
- run_2026-10-23_111354: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 1% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-23_112257: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 1% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-23_113200: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 1% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-23_114104: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 1% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-23_115008: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 1% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-23_121725: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 1%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-23_122629: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 1% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-23_123532: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 1% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-23_124436: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 1% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-23_125339: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 1% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 1% cpu6 0% cpu7 0%

## 1. GEMM arms per tile and shape

### jbpi2, 16x6, B3

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 256 | 3 | 6.70 (6.68-6.72) | 6.71 | 0 | 10.00 | 1.7 |  |
| col-g2 | 256 | 3 | 6.36 (6.35-6.37) | 6.39 | 0 | 10.00 | 2.0 |  |
| col-g4 | 256 | 3 | 6.48 (6.46-6.51) | 6.53 | 0 | 10.00 | 2.7 |  |
| col-gall | 256 | 3 | 3.88 (3.69-4.09) | 3.77 | 84 | 0.10 | 26.2 |  |
| col-g1-m4 | 256 | 3 | 6.88 (6.86-6.91) | 6.93 | 0 | 10.00 | 2.6 |  |

### jbpi2, 16x6, B3_K256

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 256 | 3 | 5.49 (4.67-5.92) | 6.74 | 34 | 0.33 | 29.2 |  |
| col-g2 | 256 | 3 | 3.48 (3.47-3.48) | 3.47 | 0 | 10.00 | 7.3 |  |
| col-g4 | 256 | 3 | 3.32 (3.10-3.70) | 3.01 | 82 | 0.03 | 30.3 |  |
| col-gall | 256 | 3 | 4.24 (4.05-4.36) | 4.47 | 54 | 0.48 | 30.9 |  |
| col-g1-m4 | 256 | 3 | 7.70 (7.56-7.83) | 7.65 | 0 | 10.00 | 3.8 |  |

### jbpi2, 16x6, B2

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 64 | 3 | 10.26 (10.22-10.31) | 10.27 | 0 | 10.00 | 0.6 |  |
| col-g2 | 64 | 3 | 10.17 (10.09-10.21) | 10.18 | 0 | 10.00 | 0.8 |  |
| col-g4 | 64 | 3 | 10.24 (10.22-10.29) | 10.25 | 0 | 10.00 | 0.6 |  |
| col-gall | 64 | 3 | 10.22 (10.20-10.24) | 10.23 | 0 | 7.43 | 0.7 |  |
| col-g1-m4 | 64 | 3 | 10.24 (10.19-10.26) | 10.24 | 0 | 10.00 | 0.6 |  |

### jbpi2, 16x6, R24

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 256 | 3 | 9.21 (9.19-9.22) | 9.21 | 0 | 10.01 | 0.5 |  |
| col-g2 | 256 | 3 | 9.17 (9.15-9.17) | 9.18 | 0 | 10.01 | 0.6 |  |
| col-g4 | 256 | 3 | 9.16 (9.11-9.21) | 9.18 | 0 | 10.01 | 0.7 |  |
| col-gall | 256 | 3 | 9.08 (9.04-9.12) | 9.15 | 0 | 10.01 | 2.1 |  |
| col-g1-m4 | 256 | 3 | 9.27 (9.26-9.28) | 9.28 | 0 | 10.01 | 0.7 |  |

### jbpi2, 16x8, B3

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 256 | 3 | 3.61 (3.60-3.62) | 3.34 | 91 | 0.05 | 25.9 |  |
| col-g2 | 256 | 3 | 6.51 (6.48-6.55) | 6.55 | 0 | 10.00 | 1.9 |  |
| col-g4 | 256 | 3 | 7.11 (7.10-7.11) | 7.13 | 0 | 10.00 | 1.7 |  |
| col-gall | 256 | 3 | 7.10 (7.08-7.14) | 7.11 | 0 | 10.00 | 1.6 |  |
| col-g1-m4 | 256 | 3 | 7.08 (7.06-7.09) | 7.11 | 0 | 10.00 | 2.1 |  |

### jbpi2, 16x8, B3_K256

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 256 | 3 | 3.65 (3.59-3.70) | 3.56 | 96 | 0.03 | 19.6 |  |
| col-g2 | 256 | 3 | 3.89 (3.88-3.90) | 3.89 | 0 | 10.00 | 5.7 |  |
| col-g4 | 256 | 3 | 8.18 (8.15-8.21) | 8.21 | 0 | 10.00 | 1.3 |  |
| col-gall | 256 | 3 | 6.65 (6.61-6.68) | 6.66 | 0 | 10.00 | 1.5 |  |
| col-g1-m4 | 256 | 3 | 7.92 (7.86-7.97) | 7.94 | 0 | 10.00 | 1.5 |  |

### jbpi2, 16x8, B2

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 64 | 3 | 10.24 (10.19-10.28) | 10.25 | 0 | 10.00 | 0.6 |  |
| col-g2 | 64 | 3 | 10.13 (10.11-10.15) | 10.13 | 0 | 10.00 | 0.7 |  |
| col-g4 | 64 | 3 | 10.15 (10.11-10.18) | 10.15 | 0 | 10.00 | 0.6 |  |
| col-gall | 64 | 3 | 10.21 (10.18-10.24) | 10.22 | 0 | 10.00 | 0.7 |  |
| col-g1-m4 | 64 | 3 | 10.24 (10.23-10.26) | 10.24 | 0 | 10.00 | 0.6 |  |

### jbpi2, 16x8, R24

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 256 | 3 | 8.96 (8.95-8.97) | 8.96 | 0 | 10.01 | 0.7 |  |
| col-g2 | 256 | 3 | 9.09 (9.07-9.11) | 9.09 | 0 | 10.01 | 0.3 |  |
| col-g4 | 256 | 3 | 9.13 (9.08-9.16) | 9.18 | 0 | 10.01 | 1.8 |  |
| col-gall | 256 | 3 | 8.99 (8.97-9.00) | 8.99 | 0 | 10.01 | 0.4 |  |
| col-g1-m4 | 256 | 3 | 9.15 (9.14-9.18) | 9.15 | 0 | 10.00 | 0.4 |  |

### jbpi2, 16x15, B3

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 128 | 3 | 4.53 (4.44-4.63) | 4.28 | 89 | 0.05 | 23.2 |  |
| col-g2 | 128 | 3 | 6.48 (6.37-6.55) | 6.48 | 0 | 10.00 | 0.9 |  |
| col-g4 | 128 | 3 | 7.17 (7.08-7.31) | 7.17 | 0 | 10.00 | 1.0 |  |
| col-gall | 128 | 3 | 6.67 (6.62-6.76) | 6.67 | 0 | 10.00 | 1.0 |  |
| col-g1-m4 | 128 | 3 | 7.01 (6.95-7.09) | 7.00 | 0 | 10.00 | 0.9 |  |

### jbpi2, 16x15, B3_K256

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 128 | 3 | 4.94 (4.78-5.19) | 4.67 | 89 | 0.02 | 24.7 |  |
| col-g2 | 128 | 3 | 7.86 (7.83-7.93) | 7.91 | 0 | 10.00 | 1.4 |  |
| col-g4 | 128 | 3 | 7.57 (7.55-7.58) | 7.59 | 0 | 10.00 | 1.7 |  |
| col-gall | 128 | 3 | 7.57 (7.52-7.61) | 7.58 | 0 | 10.00 | 1.6 |  |
| col-g1-m4 | 128 | 3 | 7.88 (7.85-7.91) | 7.88 | 0 | 10.00 | 1.6 |  |

### jbpi2, 16x15, B2

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 64 | 3 | 12.47 (12.21-12.70) | 12.50 | 0 | 10.00 | 0.8 |  |
| col-g2 | 64 | 3 | 12.49 (12.42-12.58) | 12.51 | 0 | 10.00 | 0.8 |  |
| col-g4 | 64 | 3 | 12.40 (12.22-12.53) | 12.42 | 0 | 10.00 | 0.8 |  |
| col-gall | 64 | 3 | 12.43 (12.35-12.46) | 12.44 | 0 | 10.00 | 0.8 |  |
| col-g1-m4 | 64 | 3 | 12.46 (12.40-12.53) | 12.49 | 0 | 10.00 | 0.8 |  |

### jbpi2, 16x15, R24

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 128 | 3 | 8.68 (8.51-9.00) | 9.14 | 14 | 4.89 | 9.3 |  |
| col-g2 | 128 | 3 | 8.71 (8.71-8.72) | 8.71 | 0 | 10.00 | 0.5 |  |
| col-g4 | 128 | 3 | 8.93 (8.91-8.94) | 8.93 | 0 | 10.01 | 0.4 |  |
| col-gall | 128 | 3 | 8.73 (8.71-8.74) | 8.73 | 0 | 10.01 | 0.6 |  |
| col-g1-m4 | 128 | 3 | 8.88 (8.88-8.90) | 8.89 | 0 | 10.01 | 0.4 |  |

## 2. Pack-only loop (no compute; 1 clock read per rep; rep 0 left out of the split)

### jbpi2, 16x15, B3

| variant | kc | launches | median pack (ms) | fast reps: mean (ms) | slow reps: mean (ms) | slow reps % | flags |
|---|---|---|---|---|---|---|---|
| col-g2 | 128 | 3 | 1.1368 | 1.1542 | no slow reps | 0 |  |
| col-g4 | 128 | 3 | 1.0359 | 1.0525 | no slow reps | 0 |  |
| col-gall | 128 | 3 | 1.3723 | 1.3852 | no slow reps | 0 |  |
| col-g1-m4 | 128 | 3 | 0.8392 | 0.8525 | no slow reps | 0 |  |

### jbpi2, 16x15, B3_K256

| variant | kc | launches | median pack (ms) | fast reps: mean (ms) | slow reps: mean (ms) | slow reps % | flags |
|---|---|---|---|---|---|---|---|
| col-g2 | 128 | 3 | 0.4751 | 0.4821 | no slow reps | 0 |  |
| col-g4 | 128 | 3 | 0.5014 | 0.5111 | no slow reps | 0 |  |
| col-gall | 128 | 3 | 0.6729 | 0.6840 | no slow reps | 0 |  |
| col-g1-m4 | 128 | 3 | 0.3630 | 0.3711 | no slow reps | 0 |  |

### jbpi2, 16x8, B3

| variant | kc | launches | median pack (ms) | fast reps: mean (ms) | slow reps: mean (ms) | slow reps % | flags |
|---|---|---|---|---|---|---|---|
| col-g2 | 256 | 3 | 0.8741 | 0.8949 | no slow reps | 0 |  |
| col-g4 | 256 | 3 | 0.7902 | 0.8039 | no slow reps | 0 |  |
| col-gall | 256 | 3 | 1.0882 | 1.0925 | no slow reps | 0 |  |
| col-g1-m4 | 256 | 3 | 0.5557 | 0.5599 | 0.7802 | 6 |  |

### jbpi2, 16x8, B3_K256

| variant | kc | launches | median pack (ms) | fast reps: mean (ms) | slow reps: mean (ms) | slow reps % | flags |
|---|---|---|---|---|---|---|---|
| col-g2 | 256 | 3 | 0.6448 | 0.6583 | no slow reps | 0 |  |
| col-g4 | 256 | 3 | 0.3424 | 0.3488 | no slow reps | 0 |  |
| col-gall | 256 | 3 | 0.3209 | 0.3265 | no slow reps | 0 |  |
| col-g1-m4 | 256 | 3 | 0.2500 | 0.2518 | 0.4159 | 4 |  |
