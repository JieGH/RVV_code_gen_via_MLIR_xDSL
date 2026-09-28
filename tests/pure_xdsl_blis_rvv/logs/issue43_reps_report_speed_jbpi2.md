- run_2026-09-27_162639: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-09-27_162910: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 1% cpu7 0%
- run_2026-09-27_163142: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 1% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-09-27_163413: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-09-27_163646: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-09-27_163917: jlei@jbpi2, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 1% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%

| packer | layer | nc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | CV % | best rep | slow reps % | clumping | fast start (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| vector | B3 | 384 | 3 | 4.45 (4.38-4.51) | 4.30 | 18.6 | 9.81 | 89 | 0.04 | 0.11 |
| vector | B3_nc128 | 128 | 3 | 5.49 (5.41-5.54) | 4.98 | 23.9 | 9.32 | 77 | 0.03 | 0.07 |
| vector | B3_K256 | 384 | 3 | 4.73 (4.60-4.86) | 4.64 | 17.5 | 10.02 | 96 | 0.06 | 0.03 |
| vector | B2 | 384 | 3 | 12.36 (12.32-12.40) | 12.48 | 2.8 | 12.73 | 0 | nan | 10.00 |
| vector | R24 | 384 | 3 | 7.85 (7.73-7.98) | 7.81 | 13.3 | 9.63 | 46 | 0.03 | 0.87 |
| vector+zeroline | B3 | 384 | 3 | 4.85 (4.30-5.75) | 4.77 | 24.7 | 9.85 | 78 | 0.04 | 0.16 |
| vector+zeroline | B3_nc128 | 128 | 3 | 5.99 (5.76-6.16) | 5.79 | 25.1 | 9.16 | 60 | 0.01 | 0.14 |
| vector+zeroline | B3_K256 | 384 | 3 | 4.68 (4.60-4.84) | 4.55 | 19.8 | 9.81 | 95 | 0.08 | 0.02 |
| vector+zeroline | B2 | 384 | 3 | 12.35 (12.30-12.38) | 12.40 | 1.7 | 12.72 | 0 | nan | 10.00 |
| vector+zeroline | R24 | 384 | 3 | 8.26 (8.15-8.49) | 8.76 | 12.0 | 9.56 | 21 | 0.03 | 4.26 |
