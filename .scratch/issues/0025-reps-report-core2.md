# Issue #41/#42, 2026-09-27: reps_report.py over the three core-2 launches (vector packer), interleaved with the core-4 ones.
# INSTRUMENTED build, warm-up protocol, 1000 reps, --no-nc-overrides, --raw-events x60-l2, jbpi.

- run_2026-09-27_131815: jlei@jbpi, INSTRUMENTED (INSTRUMENTED=1), WARM-UP, core 2, raw events x60-l2, busy cpu0 1% cpu1 0% cpu2 29% cpu3 7% cpu4 1% cpu5 14% cpu6 0% cpu7 50%
- run_2026-09-27_132052: jlei@jbpi, INSTRUMENTED (INSTRUMENTED=1), WARM-UP, core 2, raw events x60-l2, busy cpu0 9% cpu1 8% cpu2 30% cpu3 0% cpu4 13% cpu5 42% cpu6 0% cpu7 0%
- run_2026-09-27_133354: jlei@jbpi, INSTRUMENTED (INSTRUMENTED=1), WARM-UP, core 2, raw events x60-l2, busy cpu0 0% cpu1 0% cpu2 29% cpu3 0% cpu4 41% cpu5 15% cpu6 14% cpu7 1%

| packer | layer | nc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | CV % | best rep | slow reps % | clumping | fast start (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| vector | B3 | 384 | 3 | 5.82 (5.35-6.53) | 5.47 | 27.5 | 9.80 | 64 | 0.08 | 0.08 |
| vector | B3_K256 | 384 | 3 | 5.86 (5.69-5.97) | 5.20 | 28.5 | 9.91 | 68 | 0.08 | 0.04 |
| vector | B2 | 384 | 3 | 12.13 (12.09-12.16) | 12.16 | 1.3 | 12.36 | 0 | nan | 1.56 |
| vector | R24 | 384 | 3 | 8.04 (7.98-8.13) | 8.41 | 9.9 | 9.19 | 0 | nan | 14.09 |

| packer | layer | nc | launches | counter | fast reps: mean per rep (n) | slow reps: mean per rep (n) | slow/fast |
|---|---|---|---|---|---|---|---|
| vector | B3 | 384 | 3 | Pack_Cycles | 1.189e+06 (1084) | 2.254e+06 (1913) | 1.89 |
| vector | B3 | 384 | 3 | Compute_Cycles | 2.77e+06 (1084) | 4.06e+06 (1913) | 1.47 |
| vector | B3 | 384 | 3 | Pack_l2_load_access | 1.363e+04 (1084) | 1.37e+04 (1913) | 1.01 |
| vector | B3 | 384 | 3 | Compute_l2_load_access | 6.192e+04 (1084) | 5.908e+04 (1913) | 0.95 |
| vector | B3 | 384 | 3 | Pack_l2_load_miss | 3116 (1084) | 7009 (1913) | 2.25 |
| vector | B3 | 384 | 3 | Compute_l2_load_miss | 1730 (1084) | 5231 (1913) | 3.02 |
| vector | B3 | 384 | 3 | Pack_l2_store_access | 2.546e+04 (1084) | 2.557e+04 (1913) | 1.00 |
| vector | B3 | 384 | 3 | Compute_l2_store_access | 6.333e+04 (1084) | 6.047e+04 (1913) | 0.95 |
| vector | B3 | 384 | 3 | Pack_l2_ar_channel_stall_cycle | 2.075e+05 (1084) | 1.111e+05 (1913) | 0.54 |
| vector | B3 | 384 | 3 | Compute_l2_ar_channel_stall_cycle | 4.448e+04 (1084) | 2.571e+04 (1913) | 0.58 |
| vector | B3 | 384 | 3 | Pack_l2_aw_channel_stall_cycle | 44.49 (1084) | 33.89 (1913) | 0.76 |
| vector | B3 | 384 | 3 | Compute_l2_aw_channel_stall_cycle | 9.314 (1084) | 4.297 (1913) | 0.46 |
| vector | B3 | 384 | 3 | Pack_l1d_miss | 5162 (1084) | 8268 (1913) | 1.60 |
| vector | B3 | 384 | 3 | Compute_l1d_miss | 8512 (1084) | 2.367e+04 (1913) | 2.78 |
| vector | B3 | 384 | 3 | Pack_l1d_prefetch_refill | 5525 (1084) | 3004 (1913) | 0.54 |
| vector | B3 | 384 | 3 | Compute_l1d_prefetch_refill | 5.459e+04 (1084) | 3.495e+04 (1913) | 0.64 |
| vector | B3 | 384 | 3 | Pack_l1d_prefetch_hit | 8.382e+04 (1084) | 4.175e+04 (1913) | 0.50 |
| vector | B3 | 384 | 3 | Compute_l1d_prefetch_hit | 6.529e+05 (1084) | 3.684e+05 (1913) | 0.56 |
| vector | B3_K256 | 384 | 3 | Pack_Cycles | 7.271e+05 (963) | 1.428e+06 (2034) | 1.96 |
| vector | B3_K256 | 384 | 3 | Compute_Cycles | 1.785e+06 (963) | 2.682e+06 (2034) | 1.50 |
| vector | B3_K256 | 384 | 3 | Pack_l2_load_access | 9154 (963) | 9604 (2034) | 1.05 |
| vector | B3_K256 | 384 | 3 | Compute_l2_load_access | 4.025e+04 (963) | 3.803e+04 (2034) | 0.94 |
| vector | B3_K256 | 384 | 3 | Pack_l2_load_miss | 1387 (963) | 4001 (2034) | 2.89 |
| vector | B3_K256 | 384 | 3 | Compute_l2_load_miss | 997.5 (963) | 3311 (2034) | 3.32 |
| vector | B3_K256 | 384 | 3 | Pack_l2_store_access | 1.704e+04 (963) | 1.764e+04 (2034) | 1.04 |
| vector | B3_K256 | 384 | 3 | Compute_l2_store_access | 4.179e+04 (963) | 3.952e+04 (2034) | 0.95 |
| vector | B3_K256 | 384 | 3 | Pack_l2_ar_channel_stall_cycle | 1.399e+05 (963) | 7.368e+04 (2034) | 0.53 |
| vector | B3_K256 | 384 | 3 | Compute_l2_ar_channel_stall_cycle | 2.562e+04 (963) | 1.1e+04 (2034) | 0.43 |
| vector | B3_K256 | 384 | 3 | Pack_l2_aw_channel_stall_cycle | 1.277 (963) | 1.796 (2034) | 1.41 |
| vector | B3_K256 | 384 | 3 | Compute_l2_aw_channel_stall_cycle | 0.2451 (963) | 0.1391 (2034) | 0.57 |
| vector | B3_K256 | 384 | 3 | Pack_l1d_miss | 3198 (963) | 5896 (2034) | 1.84 |
| vector | B3_K256 | 384 | 3 | Compute_l1d_miss | 4988 (963) | 1.65e+04 (2034) | 3.31 |
| vector | B3_K256 | 384 | 3 | Pack_l1d_prefetch_refill | 4784 (963) | 2554 (2034) | 0.53 |
| vector | B3_K256 | 384 | 3 | Compute_l1d_prefetch_refill | 3.587e+04 (963) | 2.12e+04 (2034) | 0.59 |
| vector | B3_K256 | 384 | 3 | Pack_l1d_prefetch_hit | 6.082e+04 (963) | 3.04e+04 (2034) | 0.50 |
| vector | B3_K256 | 384 | 3 | Compute_l1d_prefetch_hit | 4.35e+05 (963) | 2.17e+05 (2034) | 0.50 |
| vector | B2 | 384 | 3 | Pack_Cycles | 1.864e+05 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_Cycles | 2.463e+06 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Pack_l2_load_access | 3977 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_l2_load_access | 5.05e+04 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Pack_l2_load_miss | 184.1 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_l2_load_miss | 233.5 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Pack_l2_store_access | 7058 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_l2_store_access | 6.015e+04 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Pack_l2_ar_channel_stall_cycle | 7458 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_l2_ar_channel_stall_cycle | 143.6 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Pack_l2_aw_channel_stall_cycle | 3.028 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_l2_aw_channel_stall_cycle | 0 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Pack_l1d_miss | 857.2 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_l1d_miss | 3454 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Pack_l1d_prefetch_refill | 2136 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_l1d_prefetch_refill | 4.858e+04 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Pack_l1d_prefetch_hit | 2.474e+04 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_l1d_prefetch_hit | 6.221e+05 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Pack_Cycles | 2.653e+06 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Compute_Cycles | 2.05e+07 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Pack_l2_load_access | 2.202e+04 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Compute_l2_load_access | 3.403e+05 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Pack_l2_load_miss | 7023 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Compute_l2_load_miss | 2.136e+04 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Pack_l2_store_access | 4.138e+04 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Compute_l2_store_access | 3.495e+05 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Pack_l2_ar_channel_stall_cycle | 4.037e+05 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Compute_l2_ar_channel_stall_cycle | 9.539e+04 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Pack_l2_aw_channel_stall_cycle | 217.9 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Compute_l2_aw_channel_stall_cycle | 1.006 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Pack_l1d_miss | 8555 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Compute_l1d_miss | 8.309e+04 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Pack_l1d_prefetch_refill | 9696 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Compute_l1d_prefetch_refill | 2.66e+05 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Pack_l1d_prefetch_hit | 8.887e+04 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Compute_l1d_prefetch_hit | 2.19e+06 (2997) | one speed, no slow reps | - |
