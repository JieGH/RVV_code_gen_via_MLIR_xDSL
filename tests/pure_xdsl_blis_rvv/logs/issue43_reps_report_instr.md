- run_2026-09-27_152131: jlei@jbpi, INSTRUMENTED (INSTRUMENTED=1), WARM-UP, core 4, raw events x60-l2, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-09-27_152415: jlei@jbpi, INSTRUMENTED (INSTRUMENTED=1), WARM-UP, core 4, raw events x60-l2, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 1% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-09-27_152700: jlei@jbpi, INSTRUMENTED (INSTRUMENTED=1), WARM-UP, core 4, raw events x60-l2, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-09-27_152944: jlei@jbpi, INSTRUMENTED (INSTRUMENTED=1), WARM-UP, core 4, raw events x60-l2, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 1% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-09-27_153227: jlei@jbpi, INSTRUMENTED (INSTRUMENTED=1), WARM-UP, core 4, raw events x60-l2, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-09-27_153512: jlei@jbpi, INSTRUMENTED (INSTRUMENTED=1), WARM-UP, core 4, raw events x60-l2, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%

| packer | layer | nc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | CV % | best rep | slow reps % | clumping | fast start (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| vector | B3 | 384 | 3 | 5.68 (5.44-5.91) | 5.37 | 28.8 | 9.60 | 52 | 0.06 | 0.10 |
| vector | B3_nc128 | 128 | 3 | 8.04 (7.94-8.11) | 8.09 | 3.6 | 8.69 | 0 | 1.00 | 1.95 |
| vector | B3_K256 | 384 | 3 | 5.98 (5.92-6.06) | 5.14 | 28.2 | 9.79 | 58 | 0.08 | 0.00 |
| vector | B2 | 384 | 3 | 11.95 (11.92-11.97) | 12.00 | 1.5 | 12.23 | 0 | nan | 1.58 |
| vector | R24 | 384 | 3 | 7.86 (7.64-8.21) | 7.53 | 10.2 | 9.08 | 0 | nan | 14.42 |
| vector+zeroline | B3 | 384 | 3 | 6.05 (5.77-6.25) | 6.65 | 26.2 | 9.40 | 50 | 0.08 | 0.10 |
| vector+zeroline | B3_nc128 | 128 | 3 | 5.85 (5.76-6.02) | 5.50 | 22.8 | 8.47 | 52 | 0.06 | 0.05 |
| vector+zeroline | B3_K256 | 384 | 3 | 5.75 (5.65-5.82) | 5.14 | 27.9 | 9.64 | 70 | 0.09 | 0.04 |
| vector+zeroline | B2 | 384 | 3 | 11.84 (11.79-11.86) | 11.86 | 1.1 | 12.06 | 0 | nan | 1.59 |
| vector+zeroline | R24 | 384 | 3 | 7.97 (7.82-8.08) | 8.22 | 9.5 | 9.02 | 0 | nan | 14.22 |

| packer | layer | nc | launches | counter | fast reps: mean per rep (n) | slow reps: mean per rep (n) | slow/fast |
|---|---|---|---|---|---|---|---|
| vector | B3 | 384 | 3 | Pack_Cycles | 1.248e+06 (1429) | 2.47e+06 (1568) | 1.98 |
| vector | B3 | 384 | 3 | Compute_Cycles | 2.929e+06 (1429) | 4.356e+06 (1568) | 1.49 |
| vector | B3 | 384 | 3 | Pack_l2_load_access | 1.356e+04 (1429) | 1.364e+04 (1568) | 1.01 |
| vector | B3 | 384 | 3 | Compute_l2_load_access | 6.188e+04 (1429) | 5.835e+04 (1568) | 0.94 |
| vector | B3 | 384 | 3 | Pack_l2_load_miss | 3375 (1429) | 7755 (1568) | 2.30 |
| vector | B3 | 384 | 3 | Compute_l2_load_miss | 2126 (1429) | 5825 (1568) | 2.74 |
| vector | B3 | 384 | 3 | Pack_l2_store_access | 2.535e+04 (1429) | 2.547e+04 (1568) | 1.01 |
| vector | B3 | 384 | 3 | Compute_l2_store_access | 6.316e+04 (1429) | 5.961e+04 (1568) | 0.94 |
| vector | B3 | 384 | 3 | Pack_l2_ar_channel_stall_cycle | 1.98e+05 (1429) | 8.685e+04 (1568) | 0.44 |
| vector | B3 | 384 | 3 | Compute_l2_ar_channel_stall_cycle | 4.131e+04 (1429) | 1.763e+04 (1568) | 0.43 |
| vector | B3 | 384 | 3 | Pack_l2_aw_channel_stall_cycle | 42.91 (1429) | 23.36 (1568) | 0.54 |
| vector | B3 | 384 | 3 | Compute_l2_aw_channel_stall_cycle | 5.328 (1429) | 1.747 (1568) | 0.33 |
| vector | B3 | 384 | 3 | Pack_l1d_miss | 5352 (1429) | 8834 (1568) | 1.65 |
| vector | B3 | 384 | 3 | Compute_l1d_miss | 9704 (1429) | 2.644e+04 (1568) | 2.72 |
| vector | B3 | 384 | 3 | Pack_l1d_prefetch_refill | 5392 (1429) | 2517 (1568) | 0.47 |
| vector | B3 | 384 | 3 | Compute_l1d_prefetch_refill | 5.305e+04 (1429) | 3.124e+04 (1568) | 0.59 |
| vector | B3 | 384 | 3 | Pack_l1d_prefetch_hit | 8.081e+04 (1429) | 3.38e+04 (1568) | 0.42 |
| vector | B3 | 384 | 3 | Compute_l1d_prefetch_hit | 6.235e+05 (1429) | 3.207e+05 (1568) | 0.51 |
| vector | B3_nc128 | 128 | 3 | Pack_Cycles | 1.488e+06 (2996) | 1.556e+06 (1) | 1.05 |
| vector | B3_nc128 | 128 | 3 | Compute_Cycles | 2.738e+06 (2996) | 2.942e+06 (1) | 1.07 |
| vector | B3_nc128 | 128 | 3 | Pack_l2_load_access | 2.152e+04 (2996) | 2.172e+04 (1) | 1.01 |
| vector | B3_nc128 | 128 | 3 | Compute_l2_load_access | 6.698e+04 (2996) | 6.873e+04 (1) | 1.03 |
| vector | B3_nc128 | 128 | 3 | Pack_l2_load_miss | 3496 (2996) | 3704 (1) | 1.06 |
| vector | B3_nc128 | 128 | 3 | Compute_l2_load_miss | 835.5 (2996) | 1919 (1) | 2.30 |
| vector | B3_nc128 | 128 | 3 | Pack_l2_store_access | 3.539e+04 (2996) | 3.551e+04 (1) | 1.00 |
| vector | B3_nc128 | 128 | 3 | Compute_l2_store_access | 6.809e+04 (2996) | 6.876e+04 (1) | 1.01 |
| vector | B3_nc128 | 128 | 3 | Pack_l2_ar_channel_stall_cycle | 1.487e+05 (2996) | 1.628e+05 (1) | 1.10 |
| vector | B3_nc128 | 128 | 3 | Compute_l2_ar_channel_stall_cycle | 2.318e+04 (2996) | 2.553e+04 (1) | 1.10 |
| vector | B3_nc128 | 128 | 3 | Pack_l2_aw_channel_stall_cycle | 0.9166 (2996) | 0 (1) | 0.00 |
| vector | B3_nc128 | 128 | 3 | Compute_l2_aw_channel_stall_cycle | 0.06542 (2996) | 0 (1) | 0.00 |
| vector | B3_nc128 | 128 | 3 | Pack_l1d_miss | 6771 (2996) | 6778 (1) | 1.00 |
| vector | B3_nc128 | 128 | 3 | Compute_l1d_miss | 6658 (2996) | 6317 (1) | 0.95 |
| vector | B3_nc128 | 128 | 3 | Pack_l1d_prefetch_refill | 8627 (2996) | 8644 (1) | 1.00 |
| vector | B3_nc128 | 128 | 3 | Compute_l1d_prefetch_refill | 6.077e+04 (2996) | 6.117e+04 (1) | 1.01 |
| vector | B3_nc128 | 128 | 3 | Pack_l1d_prefetch_hit | 9.307e+04 (2996) | 9.18e+04 (1) | 0.99 |
| vector | B3_nc128 | 128 | 3 | Compute_l1d_prefetch_hit | 7.405e+05 (2996) | 7.487e+05 (1) | 1.01 |
| vector | B3_K256 | 384 | 3 | Pack_Cycles | 7.495e+05 (1267) | 1.433e+06 (1730) | 1.91 |
| vector | B3_K256 | 384 | 3 | Compute_Cycles | 1.861e+06 (1267) | 2.76e+06 (1730) | 1.48 |
| vector | B3_K256 | 384 | 3 | Pack_l2_load_access | 9140 (1267) | 9542 (1730) | 1.04 |
| vector | B3_K256 | 384 | 3 | Compute_l2_load_access | 4.034e+04 (1267) | 3.798e+04 (1730) | 0.94 |
| vector | B3_K256 | 384 | 3 | Pack_l2_load_miss | 1459 (1267) | 3994 (1730) | 2.74 |
| vector | B3_K256 | 384 | 3 | Compute_l2_load_miss | 1131 (1267) | 3390 (1730) | 3.00 |
| vector | B3_K256 | 384 | 3 | Pack_l2_store_access | 1.703e+04 (1267) | 1.755e+04 (1730) | 1.03 |
| vector | B3_K256 | 384 | 3 | Compute_l2_store_access | 4.176e+04 (1267) | 3.937e+04 (1730) | 0.94 |
| vector | B3_K256 | 384 | 3 | Pack_l2_ar_channel_stall_cycle | 1.38e+05 (1267) | 7.307e+04 (1730) | 0.53 |
| vector | B3_K256 | 384 | 3 | Compute_l2_ar_channel_stall_cycle | 2.492e+04 (1267) | 9798 (1730) | 0.39 |
| vector | B3_K256 | 384 | 3 | Pack_l2_aw_channel_stall_cycle | 1.006 (1267) | 0.8798 (1730) | 0.87 |
| vector | B3_K256 | 384 | 3 | Compute_l2_aw_channel_stall_cycle | 0.2021 (1267) | 0.1543 (1730) | 0.76 |
| vector | B3_K256 | 384 | 3 | Pack_l1d_miss | 3343 (1267) | 5922 (1730) | 1.77 |
| vector | B3_K256 | 384 | 3 | Compute_l1d_miss | 5658 (1267) | 1.683e+04 (1730) | 2.97 |
| vector | B3_K256 | 384 | 3 | Pack_l1d_prefetch_refill | 4680 (1267) | 2485 (1730) | 0.53 |
| vector | B3_K256 | 384 | 3 | Compute_l1d_prefetch_refill | 3.554e+04 (1267) | 2.087e+04 (1730) | 0.59 |
| vector | B3_K256 | 384 | 3 | Pack_l1d_prefetch_hit | 5.878e+04 (1267) | 3.001e+04 (1730) | 0.51 |
| vector | B3_K256 | 384 | 3 | Compute_l1d_prefetch_hit | 4.208e+05 (1267) | 2.011e+05 (1730) | 0.48 |
| vector | B2 | 384 | 3 | Pack_Cycles | 1.981e+05 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_Cycles | 2.491e+06 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Pack_l2_load_access | 4123 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_l2_load_access | 5.027e+04 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Pack_l2_load_miss | 242.3 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_l2_load_miss | 283.9 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Pack_l2_store_access | 7194 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_l2_store_access | 5.932e+04 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Pack_l2_ar_channel_stall_cycle | 1.499e+04 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_l2_ar_channel_stall_cycle | 138.2 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Pack_l2_aw_channel_stall_cycle | 4.709 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_l2_aw_channel_stall_cycle | 0 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Pack_l1d_miss | 924.2 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_l1d_miss | 3167 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Pack_l1d_prefetch_refill | 2233 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_l1d_prefetch_refill | 4.805e+04 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Pack_l1d_prefetch_hit | 2.386e+04 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_l1d_prefetch_hit | 6.429e+05 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Pack_Cycles | 2.771e+06 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Compute_Cycles | 2.094e+07 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Pack_l2_load_access | 2.193e+04 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Compute_l2_load_access | 3.407e+05 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Pack_l2_load_miss | 7405 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Compute_l2_load_miss | 2.191e+04 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Pack_l2_store_access | 4.127e+04 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Compute_l2_store_access | 3.491e+05 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Pack_l2_ar_channel_stall_cycle | 3.91e+05 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Compute_l2_ar_channel_stall_cycle | 8.12e+04 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Pack_l2_aw_channel_stall_cycle | 243.5 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Compute_l2_aw_channel_stall_cycle | 0.9156 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Pack_l1d_miss | 8912 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Compute_l1d_miss | 8.484e+04 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Pack_l1d_prefetch_refill | 9366 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Compute_l1d_prefetch_refill | 2.632e+05 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Pack_l1d_prefetch_hit | 8.362e+04 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Compute_l1d_prefetch_hit | 2.097e+06 (2997) | one speed, no slow reps | - |
| vector+zeroline | B3 | 384 | 3 | Pack_Cycles | 1.266e+06 (1486) | 2.248e+06 (1511) | 1.78 |
| vector+zeroline | B3 | 384 | 3 | Compute_Cycles | 2.861e+06 (1486) | 4.087e+06 (1511) | 1.43 |
| vector+zeroline | B3 | 384 | 3 | Pack_l2_load_access | 1.405e+04 (1486) | 1.428e+04 (1511) | 1.02 |
| vector+zeroline | B3 | 384 | 3 | Compute_l2_load_access | 6.196e+04 (1486) | 5.929e+04 (1511) | 0.96 |
| vector+zeroline | B3 | 384 | 3 | Pack_l2_load_miss | 3253 (1486) | 6868 (1511) | 2.11 |
| vector+zeroline | B3 | 384 | 3 | Compute_l2_load_miss | 1925 (1486) | 5239 (1511) | 2.72 |
| vector+zeroline | B3 | 384 | 3 | Pack_l2_store_access | 3.557e+04 (1486) | 3.575e+04 (1511) | 1.01 |
| vector+zeroline | B3 | 384 | 3 | Compute_l2_store_access | 6.323e+04 (1486) | 6.055e+04 (1511) | 0.96 |
| vector+zeroline | B3 | 384 | 3 | Pack_l2_ar_channel_stall_cycle | 2.082e+05 (1486) | 1.179e+05 (1511) | 0.57 |
| vector+zeroline | B3 | 384 | 3 | Compute_l2_ar_channel_stall_cycle | 4.244e+04 (1486) | 2.564e+04 (1511) | 0.60 |
| vector+zeroline | B3 | 384 | 3 | Pack_l2_aw_channel_stall_cycle | 56.94 (1486) | 42.74 (1511) | 0.75 |
| vector+zeroline | B3 | 384 | 3 | Compute_l2_aw_channel_stall_cycle | 7.882 (1486) | 4.056 (1511) | 0.51 |
| vector+zeroline | B3 | 384 | 3 | Pack_l1d_miss | 5693 (1486) | 8671 (1511) | 1.52 |
| vector+zeroline | B3 | 384 | 3 | Compute_l1d_miss | 8934 (1486) | 2.301e+04 (1511) | 2.58 |
| vector+zeroline | B3 | 384 | 3 | Pack_l1d_prefetch_refill | 5461 (1486) | 3173 (1511) | 0.58 |
| vector+zeroline | B3 | 384 | 3 | Compute_l1d_prefetch_refill | 5.434e+04 (1486) | 3.593e+04 (1511) | 0.66 |
| vector+zeroline | B3 | 384 | 3 | Pack_l1d_prefetch_hit | 8.356e+04 (1486) | 4.417e+04 (1511) | 0.53 |
| vector+zeroline | B3 | 384 | 3 | Compute_l1d_prefetch_hit | 6.437e+05 (1486) | 3.764e+05 (1511) | 0.58 |
| vector+zeroline | B3_nc128 | 128 | 3 | Pack_Cycles | 1.663e+06 (1425) | 2.98e+06 (1572) | 1.79 |
| vector+zeroline | B3_nc128 | 128 | 3 | Compute_Cycles | 2.845e+06 (1425) | 3.696e+06 (1572) | 1.30 |
| vector+zeroline | B3_nc128 | 128 | 3 | Pack_l2_load_access | 3.006e+04 (1425) | 3.038e+04 (1572) | 1.01 |
| vector+zeroline | B3_nc128 | 128 | 3 | Compute_l2_load_access | 6.664e+04 (1425) | 6.357e+04 (1572) | 0.95 |
| vector+zeroline | B3_nc128 | 128 | 3 | Pack_l2_load_miss | 3591 (1425) | 8171 (1572) | 2.28 |
| vector+zeroline | B3_nc128 | 128 | 3 | Compute_l2_load_miss | 1012 (1425) | 1894 (1572) | 1.87 |
| vector+zeroline | B3_nc128 | 128 | 3 | Pack_l2_store_access | 4.761e+04 (1425) | 4.759e+04 (1572) | 1.00 |
| vector+zeroline | B3_nc128 | 128 | 3 | Compute_l2_store_access | 6.771e+04 (1425) | 6.465e+04 (1572) | 0.95 |
| vector+zeroline | B3_nc128 | 128 | 3 | Pack_l2_ar_channel_stall_cycle | 1.506e+05 (1425) | 6.162e+04 (1572) | 0.41 |
| vector+zeroline | B3_nc128 | 128 | 3 | Compute_l2_ar_channel_stall_cycle | 2.321e+04 (1425) | 9804 (1572) | 0.42 |
| vector+zeroline | B3_nc128 | 128 | 3 | Pack_l2_aw_channel_stall_cycle | 1.171 (1425) | 0.6724 (1572) | 0.57 |
| vector+zeroline | B3_nc128 | 128 | 3 | Compute_l2_aw_channel_stall_cycle | 0.1214 (1425) | 0.0458 (1572) | 0.38 |
| vector+zeroline | B3_nc128 | 128 | 3 | Pack_l1d_miss | 1.552e+04 (1425) | 2.109e+04 (1572) | 1.36 |
| vector+zeroline | B3_nc128 | 128 | 3 | Compute_l1d_miss | 9239 (1425) | 2.969e+04 (1572) | 3.21 |
| vector+zeroline | B3_nc128 | 128 | 3 | Pack_l1d_prefetch_refill | 8272 (1425) | 3545 (1572) | 0.43 |
| vector+zeroline | B3_nc128 | 128 | 3 | Compute_l1d_prefetch_refill | 5.765e+04 (1425) | 3.128e+04 (1572) | 0.54 |
| vector+zeroline | B3_nc128 | 128 | 3 | Pack_l1d_prefetch_hit | 8.871e+04 (1425) | 3.658e+04 (1572) | 0.41 |
| vector+zeroline | B3_nc128 | 128 | 3 | Compute_l1d_prefetch_hit | 6.874e+05 (1425) | 3.496e+05 (1572) | 0.51 |
| vector+zeroline | B3_K256 | 384 | 3 | Pack_Cycles | 7.25e+05 (895) | 1.405e+06 (2102) | 1.94 |
| vector+zeroline | B3_K256 | 384 | 3 | Compute_Cycles | 1.807e+06 (895) | 2.741e+06 (2102) | 1.52 |
| vector+zeroline | B3_K256 | 384 | 3 | Pack_l2_load_access | 9633 (895) | 1.038e+04 (2102) | 1.08 |
| vector+zeroline | B3_K256 | 384 | 3 | Compute_l2_load_access | 4.032e+04 (895) | 3.779e+04 (2102) | 0.94 |
| vector+zeroline | B3_K256 | 384 | 3 | Pack_l2_load_miss | 1246 (895) | 3836 (2102) | 3.08 |
| vector+zeroline | B3_K256 | 384 | 3 | Compute_l2_load_miss | 975.3 (895) | 3313 (2102) | 3.40 |
| vector+zeroline | B3_K256 | 384 | 3 | Pack_l2_store_access | 2.388e+04 (895) | 2.449e+04 (2102) | 1.03 |
| vector+zeroline | B3_K256 | 384 | 3 | Compute_l2_store_access | 4.173e+04 (895) | 3.918e+04 (2102) | 0.94 |
| vector+zeroline | B3_K256 | 384 | 3 | Pack_l2_ar_channel_stall_cycle | 1.431e+05 (895) | 7.717e+04 (2102) | 0.54 |
| vector+zeroline | B3_K256 | 384 | 3 | Compute_l2_ar_channel_stall_cycle | 2.617e+04 (895) | 1.006e+04 (2102) | 0.38 |
| vector+zeroline | B3_K256 | 384 | 3 | Pack_l2_aw_channel_stall_cycle | 1.034 (895) | 1.152 (2102) | 1.11 |
| vector+zeroline | B3_K256 | 384 | 3 | Compute_l2_aw_channel_stall_cycle | 0.276 (895) | 0.1199 (2102) | 0.43 |
| vector+zeroline | B3_K256 | 384 | 3 | Pack_l1d_miss | 3523 (895) | 6510 (2102) | 1.85 |
| vector+zeroline | B3_K256 | 384 | 3 | Compute_l1d_miss | 4966 (895) | 1.7e+04 (2102) | 3.42 |
| vector+zeroline | B3_K256 | 384 | 3 | Pack_l1d_prefetch_refill | 4951 (895) | 2696 (2102) | 0.54 |
| vector+zeroline | B3_K256 | 384 | 3 | Compute_l1d_prefetch_refill | 3.614e+04 (895) | 2.058e+04 (2102) | 0.57 |
| vector+zeroline | B3_K256 | 384 | 3 | Pack_l1d_prefetch_hit | 6.319e+04 (895) | 3.216e+04 (2102) | 0.51 |
| vector+zeroline | B3_K256 | 384 | 3 | Compute_l1d_prefetch_hit | 4.41e+05 (895) | 2.135e+05 (2102) | 0.48 |
| vector+zeroline | B2 | 384 | 3 | Pack_Cycles | 2.054e+05 (2997) | one speed, no slow reps | - |
| vector+zeroline | B2 | 384 | 3 | Compute_Cycles | 2.509e+06 (2997) | one speed, no slow reps | - |
| vector+zeroline | B2 | 384 | 3 | Pack_l2_load_access | 4386 (2997) | one speed, no slow reps | - |
| vector+zeroline | B2 | 384 | 3 | Compute_l2_load_access | 5.098e+04 (2997) | one speed, no slow reps | - |
| vector+zeroline | B2 | 384 | 3 | Pack_l2_load_miss | 185.9 (2997) | one speed, no slow reps | - |
| vector+zeroline | B2 | 384 | 3 | Compute_l2_load_miss | 317.7 (2997) | one speed, no slow reps | - |
| vector+zeroline | B2 | 384 | 3 | Pack_l2_store_access | 9815 (2997) | one speed, no slow reps | - |
| vector+zeroline | B2 | 384 | 3 | Compute_l2_store_access | 6.003e+04 (2997) | one speed, no slow reps | - |
| vector+zeroline | B2 | 384 | 3 | Pack_l2_ar_channel_stall_cycle | 9843 (2997) | one speed, no slow reps | - |
| vector+zeroline | B2 | 384 | 3 | Compute_l2_ar_channel_stall_cycle | 131.9 (2997) | one speed, no slow reps | - |
| vector+zeroline | B2 | 384 | 3 | Pack_l2_aw_channel_stall_cycle | 10.71 (2997) | one speed, no slow reps | - |
| vector+zeroline | B2 | 384 | 3 | Compute_l2_aw_channel_stall_cycle | 0 (2997) | one speed, no slow reps | - |
| vector+zeroline | B2 | 384 | 3 | Pack_l1d_miss | 1295 (2997) | one speed, no slow reps | - |
| vector+zeroline | B2 | 384 | 3 | Compute_l1d_miss | 3510 (2997) | one speed, no slow reps | - |
| vector+zeroline | B2 | 384 | 3 | Pack_l1d_prefetch_refill | 2137 (2997) | one speed, no slow reps | - |
| vector+zeroline | B2 | 384 | 3 | Compute_l1d_prefetch_refill | 4.876e+04 (2997) | one speed, no slow reps | - |
| vector+zeroline | B2 | 384 | 3 | Pack_l1d_prefetch_hit | 2.472e+04 (2997) | one speed, no slow reps | - |
| vector+zeroline | B2 | 384 | 3 | Compute_l1d_prefetch_hit | 6.193e+05 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Pack_Cycles | 2.743e+06 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Compute_Cycles | 2.064e+07 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Pack_l2_load_access | 2.207e+04 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Compute_l2_load_access | 3.412e+05 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Pack_l2_load_miss | 6880 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Compute_l2_load_miss | 2.135e+04 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Pack_l2_store_access | 5.952e+04 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Compute_l2_store_access | 3.495e+05 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Pack_l2_ar_channel_stall_cycle | 4.405e+05 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Compute_l2_ar_channel_stall_cycle | 9.2e+04 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Pack_l2_aw_channel_stall_cycle | 440.2 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Compute_l2_aw_channel_stall_cycle | 0.9269 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Pack_l1d_miss | 8514 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Compute_l1d_miss | 8.043e+04 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Pack_l1d_prefetch_refill | 9761 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Compute_l1d_prefetch_refill | 2.702e+05 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Pack_l1d_prefetch_hit | 8.954e+04 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Compute_l1d_prefetch_hit | 2.262e+06 (2997) | one speed, no slow reps | - |
