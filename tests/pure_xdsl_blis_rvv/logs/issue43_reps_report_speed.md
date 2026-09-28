- run_2026-09-27_153743: jlei@jbpi, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 1% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-09-27_154015: jlei@jbpi, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 1% cpu7 0%
- run_2026-09-27_154247: jlei@jbpi, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-09-27_154519: jlei@jbpi, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-09-27_154752: jlei@jbpi, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 1% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-09-27_155023: jlei@jbpi, UNINSTRUMENTED (INSTRUMENTED=0), REFERENCE (tmin=10 s), core 4, raw events none, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 1% cpu7 0%

| packer | layer | nc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | CV % | best rep | slow reps % | clumping | fast start (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| vector | B3 | 384 | 3 | 4.50 (4.41-4.65) | 4.31 | 20.8 | 9.94 | 90 | 0.05 | 0.12 |
| vector | B3_nc128 | 128 | 3 | 5.52 (5.37-5.69) | 4.95 | 25.8 | 9.35 | 79 | 0.02 | 0.14 |
| vector | B3_K256 | 384 | 3 | 4.61 (4.52-4.66) | 4.52 | 17.5 | 10.06 | 97 | 0.08 | 0.03 |
| vector | B2 | 384 | 3 | 12.33 (12.26-12.38) | 12.46 | 3.0 | 12.75 | 0 | nan | 10.00 |
| vector | R24 | 384 | 3 | 8.37 (7.95-8.58) | 9.01 | 12.6 | 9.62 | 31 | 0.06 | 2.32 |
| vector+zeroline | B3 | 384 | 3 | 4.44 (4.34-4.60) | 4.23 | 20.9 | 9.66 | 86 | 0.04 | 0.04 |
| vector+zeroline | B3_nc128 | 128 | 3 | 6.79 (6.14-7.36) | 7.73 | 22.2 | 9.21 | 42 | 0.03 | 0.81 |
| vector+zeroline | B3_K256 | 384 | 3 | 4.64 (4.58-4.68) | 4.55 | 16.5 | 9.77 | 97 | 0.08 | 0.02 |
| vector+zeroline | B2 | 384 | 3 | 12.21 (11.91-12.47) | 12.40 | 3.2 | 12.69 | 0 | nan | 10.00 |
| vector+zeroline | R24 | 384 | 3 | 8.15 (8.01-8.37) | 8.58 | 12.9 | 9.56 | 23 | 0.02 | 4.35 |
