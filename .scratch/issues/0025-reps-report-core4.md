# Issue #41/#42, 2026-09-27: reps_report.py over the six core-4 launches (vector and scalar packers, A-B-B-A-A-B).
# INSTRUMENTED build, warm-up protocol, 1000 reps, --no-nc-overrides, --raw-events x60-l2, jbpi.
# Busy share in these six folders was read over the whole launch (board build included), before the per-run fix.

- run_2026-09-27_131302: jlei@jbpi, INSTRUMENTED (INSTRUMENTED=1), WARM-UP, core 4, raw events x60-l2, busy cpu0 0% cpu1 55% cpu2 15% cpu3 1% cpu4 29% cpu5 1% cpu6 1% cpu7 0%
- run_2026-09-27_132606: jlei@jbpi, INSTRUMENTED (INSTRUMENTED=1), WARM-UP, core 4, raw events x60-l2, busy cpu0 0% cpu1 40% cpu2 9% cpu3 2% cpu4 30% cpu5 7% cpu6 13% cpu7 0%
- run_2026-09-27_132843: jlei@jbpi, INSTRUMENTED (INSTRUMENTED=1), WARM-UP, core 4, raw events x60-l2, busy cpu0 9% cpu1 1% cpu2 0% cpu3 0% cpu4 77% cpu5 0% cpu6 0% cpu7 13%
- run_2026-09-27_131537: jlei@jbpi, INSTRUMENTED (INSTRUMENTED=1), WARM-UP, core 4, raw events x60-l2, busy cpu0 0% cpu1 2% cpu2 36% cpu3 12% cpu4 28% cpu5 0% cpu6 0% cpu7 22%
- run_2026-09-27_132328: jlei@jbpi, INSTRUMENTED (INSTRUMENTED=1), WARM-UP, core 4, raw events x60-l2, busy cpu0 0% cpu1 1% cpu2 0% cpu3 1% cpu4 28% cpu5 0% cpu6 58% cpu7 13%
- run_2026-09-27_133117: jlei@jbpi, INSTRUMENTED (INSTRUMENTED=1), WARM-UP, core 4, raw events x60-l2, busy cpu0 0% cpu1 1% cpu2 14% cpu3 0% cpu4 41% cpu5 0% cpu6 42% cpu7 0%

| packer | layer | nc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | CV % | best rep | slow reps % | clumping | fast start (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| vector | B3 | 384 | 3 | 5.92 (5.47-6.57) | 5.55 | 28.4 | 9.57 | 51 | 0.07 | 0.06 |
| vector | B3_K256 | 384 | 3 | 6.35 (5.86-7.34) | 6.18 | 26.6 | 9.79 | 60 | 0.11 | 0.05 |
| vector | B2 | 384 | 3 | 11.94 (11.91-11.96) | 11.97 | 1.2 | 12.20 | 0 | nan | 1.58 |
| vector | R24 | 384 | 3 | 7.95 (7.72-8.36) | 8.41 | 10.6 | 9.11 | 7 | 0.04 | 9.88 |
| scalar | B3 | 384 | 3 | 8.48 (8.46-8.50) | 8.50 | 2.3 | 8.97 | 0 | nan | 2.23 |
| scalar | B3_K256 | 384 | 3 | 6.49 (6.22-6.75) | 7.74 | 25.3 | 9.02 | 36 | 0.07 | 0.11 |
| scalar | B2 | 384 | 3 | 11.51 (11.46-11.53) | 11.54 | 1.1 | 11.72 | 0 | nan | 1.64 |
| scalar | R24 | 384 | 3 | 8.71 (8.69-8.73) | 8.72 | 0.6 | 8.89 | 0 | nan | 13.00 |

| packer | layer | nc | launches | counter | fast reps: mean per rep (n) | slow reps: mean per rep (n) | slow/fast |
|---|---|---|---|---|---|---|---|
| vector | B3 | 384 | 3 | Pack_Cycles | 1.188e+06 (1480) | 2.481e+06 (1517) | 2.09 |
| vector | B3 | 384 | 3 | Compute_Cycles | 2.806e+06 (1480) | 4.232e+06 (1517) | 1.51 |
| vector | B3 | 384 | 3 | Pack_l2_load_access | 1.354e+04 (1480) | 1.364e+04 (1517) | 1.01 |
| vector | B3 | 384 | 3 | Compute_l2_load_access | 6.196e+04 (1480) | 5.862e+04 (1517) | 0.95 |
| vector | B3 | 384 | 3 | Pack_l2_load_miss | 3091 (1480) | 7799 (1517) | 2.52 |
| vector | B3 | 384 | 3 | Compute_l2_load_miss | 1715 (1480) | 5367 (1517) | 3.13 |
| vector | B3 | 384 | 3 | Pack_l2_store_access | 2.537e+04 (1480) | 2.552e+04 (1517) | 1.01 |
| vector | B3 | 384 | 3 | Compute_l2_store_access | 6.332e+04 (1480) | 5.996e+04 (1517) | 0.95 |
| vector | B3 | 384 | 3 | Pack_l2_ar_channel_stall_cycle | 2.014e+05 (1480) | 8.74e+04 (1517) | 0.43 |
| vector | B3 | 384 | 3 | Compute_l2_ar_channel_stall_cycle | 4.348e+04 (1480) | 2.077e+04 (1517) | 0.48 |
| vector | B3 | 384 | 3 | Pack_l2_aw_channel_stall_cycle | 41.16 (1480) | 32.76 (1517) | 0.80 |
| vector | B3 | 384 | 3 | Compute_l2_aw_channel_stall_cycle | 2.829 (1480) | 1.95 (1517) | 0.69 |
| vector | B3 | 384 | 3 | Pack_l1d_miss | 5152 (1480) | 8847 (1517) | 1.72 |
| vector | B3 | 384 | 3 | Compute_l1d_miss | 8561 (1480) | 2.621e+04 (1517) | 3.06 |
| vector | B3 | 384 | 3 | Pack_l1d_prefetch_refill | 5524 (1480) | 2527 (1517) | 0.46 |
| vector | B3 | 384 | 3 | Compute_l1d_prefetch_refill | 5.449e+04 (1480) | 3.172e+04 (1517) | 0.58 |
| vector | B3 | 384 | 3 | Pack_l1d_prefetch_hit | 8.415e+04 (1480) | 3.356e+04 (1517) | 0.40 |
| vector | B3 | 384 | 3 | Compute_l1d_prefetch_hit | 6.509e+05 (1480) | 3.298e+05 (1517) | 0.51 |
| vector | B3_K256 | 384 | 3 | Pack_Cycles | 6.779e+05 (1186) | 1.335e+06 (1811) | 1.97 |
| vector | B3_K256 | 384 | 3 | Compute_Cycles | 1.761e+06 (1186) | 2.64e+06 (1811) | 1.50 |
| vector | B3_K256 | 384 | 3 | Pack_l2_load_access | 9026 (1186) | 9480 (1811) | 1.05 |
| vector | B3_K256 | 384 | 3 | Compute_l2_load_access | 4.03e+04 (1186) | 3.826e+04 (1811) | 0.95 |
| vector | B3_K256 | 384 | 3 | Pack_l2_load_miss | 1132 (1186) | 3644 (1811) | 3.22 |
| vector | B3_K256 | 384 | 3 | Compute_l2_load_miss | 838.4 (1186) | 3128 (1811) | 3.73 |
| vector | B3_K256 | 384 | 3 | Pack_l2_store_access | 1.688e+04 (1186) | 1.75e+04 (1811) | 1.04 |
| vector | B3_K256 | 384 | 3 | Compute_l2_store_access | 4.182e+04 (1186) | 3.973e+04 (1811) | 0.95 |
| vector | B3_K256 | 384 | 3 | Pack_l2_ar_channel_stall_cycle | 1.35e+05 (1186) | 7.836e+04 (1811) | 0.58 |
| vector | B3_K256 | 384 | 3 | Compute_l2_ar_channel_stall_cycle | 2.491e+04 (1186) | 1.283e+04 (1811) | 0.51 |
| vector | B3_K256 | 384 | 3 | Pack_l2_aw_channel_stall_cycle | 1.246 (1186) | 2.895 (1811) | 2.32 |
| vector | B3_K256 | 384 | 3 | Compute_l2_aw_channel_stall_cycle | 0.3263 (1186) | 0.2568 (1811) | 0.79 |
| vector | B3_K256 | 384 | 3 | Pack_l1d_miss | 2932 (1186) | 5628 (1811) | 1.92 |
| vector | B3_K256 | 384 | 3 | Compute_l1d_miss | 4351 (1186) | 1.526e+04 (1811) | 3.51 |
| vector | B3_K256 | 384 | 3 | Pack_l1d_prefetch_refill | 4984 (1186) | 2769 (1811) | 0.56 |
| vector | B3_K256 | 384 | 3 | Compute_l1d_prefetch_refill | 3.665e+04 (1186) | 2.276e+04 (1811) | 0.62 |
| vector | B3_K256 | 384 | 3 | Pack_l1d_prefetch_hit | 6.433e+04 (1186) | 3.329e+04 (1811) | 0.52 |
| vector | B3_K256 | 384 | 3 | Compute_l1d_prefetch_hit | 4.552e+05 (1186) | 2.351e+05 (1811) | 0.52 |
| vector | B2 | 384 | 3 | Pack_Cycles | 1.848e+05 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_Cycles | 2.506e+06 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Pack_l2_load_access | 3941 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_l2_load_access | 5.062e+04 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Pack_l2_load_miss | 169.8 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_l2_load_miss | 275.3 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Pack_l2_store_access | 7016 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_l2_store_access | 6.023e+04 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Pack_l2_ar_channel_stall_cycle | 6262 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_l2_ar_channel_stall_cycle | 146 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Pack_l2_aw_channel_stall_cycle | 2.528 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_l2_aw_channel_stall_cycle | 0.01401 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Pack_l1d_miss | 859.5 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_l1d_miss | 3422 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Pack_l1d_prefetch_refill | 2142 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_l1d_prefetch_refill | 4.861e+04 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Pack_l1d_prefetch_hit | 2.475e+04 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_l1d_prefetch_hit | 6.261e+05 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Pack_Cycles | 2.575e+06 (2792) | 3.712e+06 (205) | 1.44 |
| vector | R24 | 384 | 3 | Compute_Cycles | 2.065e+07 (2792) | 2.289e+07 (205) | 1.11 |
| vector | R24 | 384 | 3 | Pack_l2_load_access | 2.193e+04 (2792) | 2.208e+04 (205) | 1.01 |
| vector | R24 | 384 | 3 | Compute_l2_load_access | 3.408e+05 (2792) | 3.39e+05 (205) | 0.99 |
| vector | R24 | 384 | 3 | Pack_l2_load_miss | 6676 (2792) | 1.081e+04 (205) | 1.62 |
| vector | R24 | 384 | 3 | Compute_l2_load_miss | 2.116e+04 (2792) | 2.606e+04 (205) | 1.23 |
| vector | R24 | 384 | 3 | Pack_l2_store_access | 4.129e+04 (2792) | 4.151e+04 (205) | 1.01 |
| vector | R24 | 384 | 3 | Compute_l2_store_access | 3.498e+05 (2792) | 3.48e+05 (205) | 0.99 |
| vector | R24 | 384 | 3 | Pack_l2_ar_channel_stall_cycle | 4.106e+05 (2792) | 3.091e+05 (205) | 0.75 |
| vector | R24 | 384 | 3 | Compute_l2_ar_channel_stall_cycle | 9.913e+04 (2792) | 8.657e+04 (205) | 0.87 |
| vector | R24 | 384 | 3 | Pack_l2_aw_channel_stall_cycle | 209 (2792) | 314.9 (205) | 1.51 |
| vector | R24 | 384 | 3 | Compute_l2_aw_channel_stall_cycle | 1.213 (2792) | 1.961 (205) | 1.62 |
| vector | R24 | 384 | 3 | Pack_l1d_miss | 8249 (2792) | 1.12e+04 (205) | 1.36 |
| vector | R24 | 384 | 3 | Compute_l1d_miss | 8.097e+04 (2792) | 1.143e+05 (205) | 1.41 |
| vector | R24 | 384 | 3 | Pack_l1d_prefetch_refill | 9928 (2792) | 7517 (205) | 0.76 |
| vector | R24 | 384 | 3 | Compute_l1d_prefetch_refill | 2.696e+05 (2792) | 2.204e+05 (205) | 0.82 |
| vector | R24 | 384 | 3 | Pack_l1d_prefetch_hit | 9.207e+04 (2792) | 5.74e+04 (205) | 0.62 |
| vector | R24 | 384 | 3 | Compute_l1d_prefetch_hit | 2.282e+06 (2792) | 1.092e+06 (205) | 0.48 |
| scalar | B3 | 384 | 3 | Pack_Cycles | 1.247e+06 (2997) | one speed, no slow reps | - |
| scalar | B3 | 384 | 3 | Compute_Cycles | 2.553e+06 (2997) | one speed, no slow reps | - |
| scalar | B3 | 384 | 3 | Pack_l2_load_access | 1.691e+04 (2997) | one speed, no slow reps | - |
| scalar | B3 | 384 | 3 | Compute_l2_load_access | 6.226e+04 (2997) | one speed, no slow reps | - |
| scalar | B3 | 384 | 3 | Pack_l2_load_miss | 2081 (2997) | one speed, no slow reps | - |
| scalar | B3 | 384 | 3 | Compute_l2_load_miss | 936.2 (2997) | one speed, no slow reps | - |
| scalar | B3 | 384 | 3 | Pack_l2_store_access | 2.515e+04 (2997) | one speed, no slow reps | - |
| scalar | B3 | 384 | 3 | Compute_l2_store_access | 6.368e+04 (2997) | one speed, no slow reps | - |
| scalar | B3 | 384 | 3 | Pack_l2_ar_channel_stall_cycle | 1.881e+05 (2997) | one speed, no slow reps | - |
| scalar | B3 | 384 | 3 | Compute_l2_ar_channel_stall_cycle | 4.543e+04 (2997) | one speed, no slow reps | - |
| scalar | B3 | 384 | 3 | Pack_l2_aw_channel_stall_cycle | 17.39 (2997) | one speed, no slow reps | - |
| scalar | B3 | 384 | 3 | Compute_l2_aw_channel_stall_cycle | 1.974 (2997) | one speed, no slow reps | - |
| scalar | B3 | 384 | 3 | Pack_l1d_miss | 7619 (2997) | one speed, no slow reps | - |
| scalar | B3 | 384 | 3 | Compute_l1d_miss | 5483 (2997) | one speed, no slow reps | - |
| scalar | B3 | 384 | 3 | Pack_l1d_prefetch_refill | 6494 (2997) | one speed, no slow reps | - |
| scalar | B3 | 384 | 3 | Compute_l1d_prefetch_refill | 5.845e+04 (2997) | one speed, no slow reps | - |
| scalar | B3 | 384 | 3 | Pack_l1d_prefetch_hit | 1.151e+05 (2997) | one speed, no slow reps | - |
| scalar | B3 | 384 | 3 | Compute_l1d_prefetch_hit | 7.273e+05 (2997) | one speed, no slow reps | - |
| scalar | B3_K256 | 384 | 3 | Pack_Cycles | 8.951e+05 (1926) | 1.879e+06 (1071) | 2.10 |
| scalar | B3_K256 | 384 | 3 | Compute_Cycles | 1.743e+06 (1926) | 2.539e+06 (1071) | 1.46 |
| scalar | B3_K256 | 384 | 3 | Pack_l2_load_access | 1.214e+04 (1926) | 1.545e+04 (1071) | 1.27 |
| scalar | B3_K256 | 384 | 3 | Compute_l2_load_access | 4.057e+04 (1926) | 3.779e+04 (1071) | 0.93 |
| scalar | B3_K256 | 384 | 3 | Pack_l2_load_miss | 1288 (1926) | 5421 (1071) | 4.21 |
| scalar | B3_K256 | 384 | 3 | Compute_l2_load_miss | 746.3 (1926) | 2531 (1071) | 3.39 |
| scalar | B3_K256 | 384 | 3 | Pack_l2_store_access | 1.74e+04 (1926) | 1.729e+04 (1071) | 0.99 |
| scalar | B3_K256 | 384 | 3 | Compute_l2_store_access | 4.208e+04 (1926) | 3.926e+04 (1071) | 0.93 |
| scalar | B3_K256 | 384 | 3 | Pack_l2_ar_channel_stall_cycle | 1.238e+05 (1926) | 5.373e+04 (1071) | 0.43 |
| scalar | B3_K256 | 384 | 3 | Compute_l2_ar_channel_stall_cycle | 2.448e+04 (1926) | 1.055e+04 (1071) | 0.43 |
| scalar | B3_K256 | 384 | 3 | Pack_l2_aw_channel_stall_cycle | 0.7705 (1926) | 0.3557 (1071) | 0.46 |
| scalar | B3_K256 | 384 | 3 | Compute_l2_aw_channel_stall_cycle | 0.0945 (1926) | 0.127 (1071) | 1.34 |
| scalar | B3_K256 | 384 | 3 | Pack_l1d_miss | 5688 (1926) | 1.239e+04 (1071) | 2.18 |
| scalar | B3_K256 | 384 | 3 | Compute_l1d_miss | 4417 (1926) | 1.658e+04 (1071) | 3.75 |
| scalar | B3_K256 | 384 | 3 | Pack_l1d_prefetch_refill | 5366 (1926) | 1786 (1071) | 0.33 |
| scalar | B3_K256 | 384 | 3 | Compute_l1d_prefetch_refill | 3.696e+04 (1926) | 2.103e+04 (1071) | 0.57 |
| scalar | B3_K256 | 384 | 3 | Pack_l1d_prefetch_hit | 7.448e+04 (1926) | 2.282e+04 (1071) | 0.31 |
| scalar | B3_K256 | 384 | 3 | Compute_l1d_prefetch_hit | 4.54e+05 (1926) | 2.242e+05 (1071) | 0.49 |
| scalar | B2 | 384 | 3 | Pack_Cycles | 2.815e+05 (2997) | one speed, no slow reps | - |
| scalar | B2 | 384 | 3 | Compute_Cycles | 2.503e+06 (2997) | one speed, no slow reps | - |
| scalar | B2 | 384 | 3 | Pack_l2_load_access | 4724 (2997) | one speed, no slow reps | - |
| scalar | B2 | 384 | 3 | Compute_l2_load_access | 5.022e+04 (2997) | one speed, no slow reps | - |
| scalar | B2 | 384 | 3 | Pack_l2_load_miss | 175.6 (2997) | one speed, no slow reps | - |
| scalar | B2 | 384 | 3 | Compute_l2_load_miss | 280.3 (2997) | one speed, no slow reps | - |
| scalar | B2 | 384 | 3 | Pack_l2_store_access | 7158 (2997) | one speed, no slow reps | - |
| scalar | B2 | 384 | 3 | Compute_l2_store_access | 5.983e+04 (2997) | one speed, no slow reps | - |
| scalar | B2 | 384 | 3 | Pack_l2_ar_channel_stall_cycle | 6368 (2997) | one speed, no slow reps | - |
| scalar | B2 | 384 | 3 | Compute_l2_ar_channel_stall_cycle | 84.42 (2997) | one speed, no slow reps | - |
| scalar | B2 | 384 | 3 | Pack_l2_aw_channel_stall_cycle | 2.692 (2997) | one speed, no slow reps | - |
| scalar | B2 | 384 | 3 | Compute_l2_aw_channel_stall_cycle | 0.005005 (2997) | one speed, no slow reps | - |
| scalar | B2 | 384 | 3 | Pack_l1d_miss | 1498 (2997) | one speed, no slow reps | - |
| scalar | B2 | 384 | 3 | Compute_l1d_miss | 3402 (2997) | one speed, no slow reps | - |
| scalar | B2 | 384 | 3 | Pack_l1d_prefetch_refill | 2285 (2997) | one speed, no slow reps | - |
| scalar | B2 | 384 | 3 | Compute_l1d_prefetch_refill | 4.836e+04 (2997) | one speed, no slow reps | - |
| scalar | B2 | 384 | 3 | Pack_l1d_prefetch_hit | 3.824e+04 (2997) | one speed, no slow reps | - |
| scalar | B2 | 384 | 3 | Compute_l1d_prefetch_hit | 6.228e+05 (2997) | one speed, no slow reps | - |
| scalar | R24 | 384 | 3 | Pack_Cycles | 2.275e+06 (2997) | one speed, no slow reps | - |
| scalar | R24 | 384 | 3 | Compute_Cycles | 1.915e+07 (2997) | one speed, no slow reps | - |
| scalar | R24 | 384 | 3 | Pack_l2_load_access | 2.438e+04 (2997) | one speed, no slow reps | - |
| scalar | R24 | 384 | 3 | Compute_l2_load_access | 3.417e+05 (2997) | one speed, no slow reps | - |
| scalar | R24 | 384 | 3 | Pack_l2_load_miss | 4176 (2997) | one speed, no slow reps | - |
| scalar | R24 | 384 | 3 | Compute_l2_load_miss | 1.818e+04 (2997) | one speed, no slow reps | - |
| scalar | R24 | 384 | 3 | Pack_l2_store_access | 4.122e+04 (2997) | one speed, no slow reps | - |
| scalar | R24 | 384 | 3 | Compute_l2_store_access | 3.505e+05 (2997) | one speed, no slow reps | - |
| scalar | R24 | 384 | 3 | Pack_l2_ar_channel_stall_cycle | 3.519e+05 (2997) | one speed, no slow reps | - |
| scalar | R24 | 384 | 3 | Compute_l2_ar_channel_stall_cycle | 1.075e+05 (2997) | one speed, no slow reps | - |
| scalar | R24 | 384 | 3 | Pack_l2_aw_channel_stall_cycle | 139.1 (2997) | one speed, no slow reps | - |
| scalar | R24 | 384 | 3 | Compute_l2_aw_channel_stall_cycle | 1.413 (2997) | one speed, no slow reps | - |
| scalar | R24 | 384 | 3 | Pack_l1d_miss | 8268 (2997) | one speed, no slow reps | - |
| scalar | R24 | 384 | 3 | Compute_l1d_miss | 5.9e+04 (2997) | one speed, no slow reps | - |
| scalar | R24 | 384 | 3 | Pack_l1d_prefetch_refill | 1.233e+04 (2997) | one speed, no slow reps | - |
| scalar | R24 | 384 | 3 | Compute_l1d_prefetch_refill | 3.025e+05 (2997) | one speed, no slow reps | - |
| scalar | R24 | 384 | 3 | Pack_l1d_prefetch_hit | 2.125e+05 (2997) | one speed, no slow reps | - |
| scalar | R24 | 384 | 3 | Compute_l1d_prefetch_hit | 3.096e+06 (2997) | one speed, no slow reps | - |
