# Issues #54/#55 series issue55_2026-10-22_194043

Build: uninstrumented (INSTRUMENTED=0); protocol: reference, --tmin 10; core 4; --no-nc-overrides (each case's nc is the cache model's)

## Run conditions

- run_2026-10-22_194942: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 1% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-22_195846: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 1% cpu7 0%
- run_2026-10-22_200751: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 1% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-22_201655: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 1% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-22_202600: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 1% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-22_203504: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 1% cpu7 0%
- run_2026-10-22_210650: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 1% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-22_211553: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 1% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-22_212456: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 1% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-22_213359: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 1% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 1% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-22_214302: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 1% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-22_215206: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 1% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-22_222854: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 1% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-22_223758: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 1% cpu6 1% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 1% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-22_224701: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 1% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-22_225605: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 1% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 1% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-22_230508: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 1% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%
- run_2026-10-22_231411: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%; fp32_AXPY_FP32_Pack_vlen256_MR16NR6: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 1%; fp32_AXPY_FP32_Pack_vlen256_MR16NR8: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%

## 1. GEMM arms per tile and shape

### jbpi2, 16x6, B3

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 256 | 3 | 6.72 (6.69-6.77) | 6.73 | 0 | 10.00 | 1.6 |  |
| row-intr | 256 | 3 | 6.30 (6.28-6.32) | 6.33 | 0 | 10.00 | 1.7 |  |
| scalar | 256 | 3 | 6.19 (6.18-6.21) | 6.19 | 0 | 10.00 | 1.0 |  |
| col-g1 | 256 | 3 | 6.90 (6.88-6.93) | 6.95 | 0 | 10.00 | 2.6 |  |
| col-gall | 256 | 3 | 4.22 (3.69-4.80) | 4.50 | 69 | 0.32 | 27.0 |  |
| col-g1-m4 | 256 | 3 | 6.89 (6.87-6.90) | 6.94 | 0 | 10.00 | 2.6 |  |

### jbpi2, 16x6, B3_K256

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 256 | 3 | 4.81 (4.63-5.14) | 5.33 | 51 | 0.35 | 33.4 |  |
| row-intr | 256 | 3 | 4.33 (3.80-5.22) | 4.42 | 63 | 0.12 | 32.5 |  |
| scalar | 256 | 3 | 5.87 (5.75-5.93) | 6.47 | 41 | 0.52 | 19.4 |  |
| col-g1 | 256 | 3 | 7.61 (7.59-7.65) | 7.58 | 0 | 10.00 | 3.4 |  |
| col-gall | 256 | 3 | 4.38 (3.97-5.05) | 4.59 | 48 | 0.33 | 31.1 |  |
| col-g1-m4 | 256 | 3 | 7.72 (7.66-7.76) | 7.66 | 0 | 10.00 | 4.0 |  |

### jbpi2, 16x6, B2

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 64 | 3 | 10.24 (10.19-10.27) | 10.25 | 0 | 10.00 | 0.6 |  |
| row-intr | 64 | 3 | 10.05 (10.01-10.09) | 10.06 | 0 | 10.00 | 0.6 |  |
| scalar | 64 | 3 | 9.87 (9.84-9.89) | 9.87 | 0 | 10.00 | 0.6 |  |
| col-g1 | 64 | 3 | 10.13 (10.11-10.17) | 10.14 | 0 | 10.00 | 0.6 |  |
| col-gall | 64 | 3 | 10.19 (10.08-10.26) | 10.19 | 0 | 10.00 | 0.6 |  |
| col-g1-m4 | 64 | 3 | 10.15 (10.05-10.21) | 10.16 | 0 | 10.00 | 0.6 |  |

### jbpi2, 16x6, R24

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 256 | 3 | 9.21 (9.20-9.23) | 9.22 | 0 | 10.01 | 0.5 |  |
| row-intr | 256 | 3 | 8.89 (8.84-8.91) | 8.89 | 0 | 10.01 | 0.4 |  |
| scalar | 256 | 3 | 8.84 (8.83-8.87) | 8.85 | 0 | 10.01 | 0.3 |  |
| col-g1 | 256 | 3 | 9.28 (9.22-9.33) | 9.29 | 0 | 10.01 | 0.7 |  |
| col-gall | 256 | 3 | 9.10 (9.08-9.13) | 9.16 | 0 | 10.01 | 1.6 |  |
| col-g1-m4 | 256 | 3 | 9.29 (9.28-9.29) | 9.30 | 0 | 10.01 | 0.7 |  |

### jbpi2, 16x8, B3

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 256 | 3 | 3.83 (3.52-4.46) | 3.65 | 84 | 0.05 | 28.5 |  |
| row-intr | 256 | 3 | 3.50 (3.48-3.53) | 3.30 | 93 | 0.05 | 24.7 |  |
| scalar | 256 | 3 | 6.89 (6.32-7.24) | 7.45 | 9 | 4.84 | 12.7 |  |
| col-g1 | 256 | 3 | 7.05 (7.03-7.07) | 7.09 | 0 | 10.00 | 2.1 |  |
| col-gall | 256 | 3 | 7.11 (7.08-7.13) | 7.12 | 0 | 10.00 | 1.6 |  |
| col-g1-m4 | 256 | 3 | 7.07 (7.06-7.09) | 7.11 | 0 | 10.00 | 2.0 |  |

### jbpi2, 16x8, B3_K256

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 256 | 3 | 3.65 (3.61-3.68) | 3.55 | 96 | 0.01 | 19.0 |  |
| row-intr | 256 | 3 | 3.76 (3.66-3.94) | 3.63 | 93 | 0.01 | 19.2 |  |
| scalar | 256 | 3 | 7.39 (7.32-7.48) | 7.91 | 14 | 1.03 | 12.1 |  |
| col-g1 | 256 | 3 | 7.88 (7.86-7.92) | 7.90 | 0 | 10.00 | 1.3 |  |
| col-gall | 256 | 3 | 6.69 (6.68-6.69) | 6.69 | 0 | 10.00 | 1.5 |  |
| col-g1-m4 | 256 | 3 | 7.87 (7.86-7.88) | 7.90 | 0 | 10.00 | 1.3 |  |

### jbpi2, 16x8, B2

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 64 | 3 | 10.26 (10.24-10.31) | 10.27 | 0 | 10.00 | 0.6 |  |
| row-intr | 64 | 3 | 10.18 (10.16-10.19) | 10.19 | 0 | 10.00 | 0.6 |  |
| scalar | 64 | 3 | 9.87 (9.83-9.89) | 9.88 | 0 | 10.00 | 0.6 |  |
| col-g1 | 64 | 3 | 10.19 (10.17-10.21) | 10.19 | 0 | 10.00 | 0.6 |  |
| col-gall | 64 | 3 | 10.17 (10.11-10.24) | 10.17 | 0 | 10.00 | 0.6 |  |
| col-g1-m4 | 64 | 3 | 10.17 (10.13-10.21) | 10.17 | 0 | 10.00 | 0.6 |  |

### jbpi2, 16x8, R24

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 256 | 3 | 8.95 (8.88-8.99) | 8.97 | 0 | 10.01 | 1.0 |  |
| row-intr | 256 | 3 | 7.07 (6.99-7.13) | 7.08 | 73 | 0.44 | 15.0 |  |
| scalar | 256 | 3 | 9.03 (8.96-9.11) | 9.11 | 0 | 10.01 | 2.2 |  |
| col-g1 | 256 | 3 | 9.17 (9.14-9.21) | 9.16 | 0 | 10.01 | 0.4 |  |
| col-gall | 256 | 3 | 8.99 (8.98-9.01) | 8.99 | 0 | 10.01 | 0.4 |  |
| col-g1-m4 | 256 | 3 | 9.15 (9.14-9.17) | 9.15 | 0 | 10.01 | 0.4 |  |

### jbpi2, 16x15, B3

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 128 | 3 | 4.63 (4.31-5.17) | 4.28 | 86 | 0.09 | 23.0 |  |
| row-intr | 128 | 3 | 4.37 (4.27-4.52) | 4.18 | 86 | 0.05 | 18.9 |  |
| scalar | 128 | 3 | 6.98 (6.59-7.62) | 7.81 | 38 | 0.36 | 21.1 |  |
| col-g1 | 128 | 3 | 7.05 (7.02-7.08) | 7.05 | 0 | 10.00 | 0.9 |  |
| col-gall | 128 | 3 | 6.72 (6.61-6.79) | 6.72 | 0 | 10.00 | 1.0 |  |
| col-g1-m4 | 128 | 3 | 7.03 (6.95-7.09) | 7.02 | 0 | 10.00 | 0.9 |  |

### jbpi2, 16x15, B3_K256

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 128 | 3 | 4.72 (4.67-4.76) | 4.56 | 94 | 0.02 | 22.1 |  |
| row-intr | 128 | 3 | 4.60 (4.55-4.65) | 4.46 | 94 | 0.02 | 20.5 |  |
| scalar | 128 | 3 | 4.64 (4.60-4.66) | 4.42 | 91 | 0.04 | 23.8 |  |
| col-g1 | 128 | 3 | 8.01 (7.96-8.03) | 8.02 | 0 | 10.00 | 1.5 |  |
| col-gall | 128 | 3 | 7.59 (7.54-7.67) | 7.60 | 0 | 10.00 | 1.7 |  |
| col-g1-m4 | 128 | 3 | 7.96 (7.85-8.08) | 8.01 | 0 | 10.00 | 1.4 |  |

### jbpi2, 16x15, B2

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 64 | 3 | 12.64 (12.60-12.71) | 12.68 | 0 | 10.00 | 0.9 |  |
| row-intr | 64 | 3 | 12.33 (12.25-12.38) | 12.35 | 0 | 10.00 | 0.8 |  |
| scalar | 64 | 3 | 12.13 (12.10-12.16) | 12.17 | 0 | 10.00 | 0.7 |  |
| col-g1 | 64 | 3 | 12.43 (12.33-12.52) | 12.45 | 0 | 10.00 | 0.7 |  |
| col-gall | 64 | 3 | 12.38 (12.37-12.41) | 12.40 | 0 | 7.11 | 0.8 |  |
| col-g1-m4 | 64 | 3 | 12.57 (12.55-12.59) | 12.60 | 0 | 10.00 | 0.8 |  |

### jbpi2, 16x15, R24

| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |
|---|---|---|---|---|---|---|---|---|
| row | 128 | 3 | 8.48 (8.31-8.72) | 9.01 | 17 | 4.10 | 10.9 |  |
| row-intr | 128 | 3 | 7.55 (7.40-7.68) | 7.26 | 53 | 1.43 | 13.4 |  |
| scalar | 128 | 3 | 9.25 (9.22-9.26) | 9.27 | 0 | 10.01 | 1.1 |  |
| col-g1 | 128 | 3 | 8.86 (8.85-8.87) | 8.86 | 0 | 10.01 | 0.4 |  |
| col-gall | 128 | 3 | 8.74 (8.74-8.75) | 8.74 | 0 | 10.01 | 0.5 |  |
| col-g1-m4 | 128 | 3 | 8.87 (8.86-8.88) | 8.87 | 0 | 10.01 | 0.4 |  |

## 2. Pack-only loop (no compute; 1 clock read per rep; rep 0 left out of the split)

### jbpi2, 16x15, B3

| variant | kc | launches | median pack (ms) | fast reps: mean (ms) | slow reps: mean (ms) | slow reps % | flags |
|---|---|---|---|---|---|---|---|
| row-intr | 128 | 3 | 1.1904 | 0.6443 | 1.1994 | 91 |  |
| scalar | 128 | 3 | 1.6721 | 0.6571 | 1.5974 | 84 |  |
| col-g1 | 128 | 3 | 0.8734 | 0.8907 | no slow reps | 0 |  |
| col-gall | 128 | 3 | 1.3892 | 1.4031 | no slow reps | 0 |  |
| col-g1-m4 | 128 | 3 | 0.8070 | 0.8218 | no slow reps | 0 |  |

### jbpi2, 16x15, B3_K256

| variant | kc | launches | median pack (ms) | fast reps: mean (ms) | slow reps: mean (ms) | slow reps % | flags |
|---|---|---|---|---|---|---|---|
| row-intr | 128 | 3 | 0.4113 | 0.3779 | 0.4361 | 33 |  |
| scalar | 128 | 3 | 0.7790 | 0.4179 | 0.7830 | 95 |  |
| col-g1 | 128 | 3 | 0.4045 | 0.4121 | no slow reps | 0 |  |
| col-gall | 128 | 3 | 0.6442 | 0.6554 | no slow reps | 0 |  |
| col-g1-m4 | 128 | 3 | 0.3563 | 0.3638 | no slow reps | 0 |  |

### jbpi2, 16x8, B3

| variant | kc | launches | median pack (ms) | fast reps: mean (ms) | slow reps: mean (ms) | slow reps % | flags |
|---|---|---|---|---|---|---|---|
| row-intr | 256 | 3 | 1.1612 | 0.8762 | 1.1740 | 91 |  |
| scalar | 256 | 3 | 0.6329 | 0.6393 | 0.9121 | 5 |  |
| col-g1 | 256 | 3 | 0.6037 | 0.6086 | 0.8361 | 6 |  |
| col-gall | 256 | 3 | 1.0725 | 1.0775 | no slow reps | 0 |  |
| col-g1-m4 | 256 | 3 | 0.5553 | 0.5598 | 0.7851 | 6 |  |

### jbpi2, 16x8, B3_K256

| variant | kc | launches | median pack (ms) | fast reps: mean (ms) | slow reps: mean (ms) | slow reps % | flags |
|---|---|---|---|---|---|---|---|
| row-intr | 256 | 3 | 0.2162 | 0.2185 | no slow reps | 0 |  |
| scalar | 256 | 3 | 0.3755 | 0.3776 | no slow reps | 0 |  |
| col-g1 | 256 | 3 | 0.2867 | 0.2892 | 0.4577 | 7 |  |
| col-gall | 256 | 3 | 0.3249 | 0.3307 | no slow reps | 0 |  |
| col-g1-m4 | 256 | 3 | 0.2442 | 0.2459 | 0.4092 | 5 |  |
