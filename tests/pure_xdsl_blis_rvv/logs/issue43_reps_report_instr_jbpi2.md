- run_2026-09-27_161022: jlei@jbpi2, INSTRUMENTED (INSTRUMENTED=1), WARM-UP, core 4, raw events x60-l2, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-09-27_161307: jlei@jbpi2, INSTRUMENTED (INSTRUMENTED=1), WARM-UP, core 4, raw events x60-l2, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 1% cpu6 0% cpu7 0%
- run_2026-09-27_161552: jlei@jbpi2, INSTRUMENTED (INSTRUMENTED=1), WARM-UP, core 4, raw events x60-l2, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-09-27_161837: jlei@jbpi2, INSTRUMENTED (INSTRUMENTED=1), WARM-UP, core 4, raw events x60-l2, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-09-27_162122: jlei@jbpi2, INSTRUMENTED (INSTRUMENTED=1), WARM-UP, core 4, raw events x60-l2, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 1% cpu1 0% cpu2 0% cpu3 0% cpu4 100% cpu5 0% cpu6 0% cpu7 0%
- run_2026-09-27_162407: jlei@jbpi2, INSTRUMENTED (INSTRUMENTED=1), WARM-UP, core 4, raw events x60-l2, busy fp32_AXPY_FP32_Pack_vlen256_MR16NR15: cpu0 0% cpu1 0% cpu2 0% cpu3 0% cpu4 99% cpu5 0% cpu6 0% cpu7 0%

| packer | layer | nc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | CV % | best rep | slow reps % | clumping | fast start (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| vector | B3 | 384 | 3 | 5.80 (5.53-6.35) | 5.36 | 29.5 | 9.59 | 60 | 0.07 | 0.14 |
| vector | B3_nc128 | 128 | 3 | 8.13 (8.08-8.17) | 8.14 | 2.5 | 8.87 | 0 | nan | 2.32 |
| vector | B3_K256 | 384 | 3 | 6.50 (5.81-7.71) | 6.38 | 26.6 | 9.83 | 54 | 0.10 | 0.00 |
| vector | B2 | 384 | 3 | 11.97 (11.95-12.00) | 12.02 | 1.3 | 12.21 | 0 | nan | 1.58 |
| vector | R24 | 384 | 3 | 7.61 (7.54-7.67) | 7.08 | 10.4 | 9.04 | 0 | nan | 14.89 |
| vector+zeroline | B3 | 384 | 3 | 5.53 (5.41-5.78) | 5.29 | 28.1 | 9.40 | 55 | 0.06 | 0.08 |
| vector+zeroline | B3_nc128 | 128 | 3 | 5.64 (5.51-5.76) | 4.93 | 22.5 | 8.39 | 60 | 0.06 | 0.04 |
| vector+zeroline | B3_K256 | 384 | 3 | 5.68 (5.56-5.76) | 5.11 | 27.4 | 9.67 | 72 | 0.10 | 0.03 |
| vector+zeroline | B2 | 384 | 3 | 11.88 (11.87-11.88) | 11.91 | 1.1 | 12.11 | 0 | nan | 1.59 |
| vector+zeroline | R24 | 384 | 3 | 7.93 (7.90-8.00) | 8.24 | 9.7 | 9.03 | 0 | nan | 14.28 |

| packer | layer | nc | launches | counter | fast reps: mean per rep (n) | slow reps: mean per rep (n) | slow/fast |
|---|---|---|---|---|---|---|---|
| vector | B3 | 384 | 3 | Pack_Cycles | 1.139e+06 (1193) | 2.296e+06 (1804) | 2.02 |
| vector | B3 | 384 | 3 | Compute_Cycles | 2.76e+06 (1193) | 4.199e+06 (1804) | 1.52 |
| vector | B3 | 384 | 3 | Pack_l2_load_access | 1.358e+04 (1193) | 1.365e+04 (1804) | 1.01 |
| vector | B3 | 384 | 3 | Compute_l2_load_access | 6.194e+04 (1193) | 5.87e+04 (1804) | 0.95 |
| vector | B3 | 384 | 3 | Pack_l2_load_miss | 2953 (1193) | 7182 (1804) | 2.43 |
| vector | B3 | 384 | 3 | Compute_l2_load_miss | 1600 (1193) | 5458 (1804) | 3.41 |
| vector | B3 | 384 | 3 | Pack_l2_store_access | 2.536e+04 (1193) | 2.55e+04 (1804) | 1.01 |
| vector | B3 | 384 | 3 | Compute_l2_store_access | 6.329e+04 (1193) | 5.999e+04 (1804) | 0.95 |
| vector | B3 | 384 | 3 | Pack_l2_ar_channel_stall_cycle | 2.066e+05 (1193) | 1.032e+05 (1804) | 0.50 |
| vector | B3 | 384 | 3 | Compute_l2_ar_channel_stall_cycle | 4.282e+04 (1193) | 2.151e+04 (1804) | 0.50 |
| vector | B3 | 384 | 3 | Pack_l2_aw_channel_stall_cycle | 57.23 (1193) | 39.02 (1804) | 0.68 |
| vector | B3 | 384 | 3 | Compute_l2_aw_channel_stall_cycle | 5.916 (1193) | 3.144 (1804) | 0.53 |
| vector | B3 | 384 | 3 | Pack_l1d_miss | 4983 (1193) | 8375 (1804) | 1.68 |
| vector | B3 | 384 | 3 | Compute_l1d_miss | 7853 (1193) | 2.417e+04 (1804) | 3.08 |
| vector | B3 | 384 | 3 | Pack_l1d_prefetch_refill | 5659 (1193) | 2890 (1804) | 0.51 |
| vector | B3 | 384 | 3 | Compute_l1d_prefetch_refill | 5.512e+04 (1193) | 3.4e+04 (1804) | 0.62 |
| vector | B3 | 384 | 3 | Pack_l1d_prefetch_hit | 8.59e+04 (1193) | 4.019e+04 (1804) | 0.47 |
| vector | B3 | 384 | 3 | Compute_l1d_prefetch_hit | 6.693e+05 (1193) | 3.542e+05 (1804) | 0.53 |
| vector | B3_nc128 | 128 | 3 | Pack_Cycles | 1.45e+06 (2997) | one speed, no slow reps | - |
| vector | B3_nc128 | 128 | 3 | Compute_Cycles | 2.724e+06 (2997) | one speed, no slow reps | - |
| vector | B3_nc128 | 128 | 3 | Pack_l2_load_access | 2.157e+04 (2997) | one speed, no slow reps | - |
| vector | B3_nc128 | 128 | 3 | Compute_l2_load_access | 6.7e+04 (2997) | one speed, no slow reps | - |
| vector | B3_nc128 | 128 | 3 | Pack_l2_load_miss | 3343 (2997) | one speed, no slow reps | - |
| vector | B3_nc128 | 128 | 3 | Compute_l2_load_miss | 862.6 (2997) | one speed, no slow reps | - |
| vector | B3_nc128 | 128 | 3 | Pack_l2_store_access | 3.555e+04 (2997) | one speed, no slow reps | - |
| vector | B3_nc128 | 128 | 3 | Compute_l2_store_access | 6.816e+04 (2997) | one speed, no slow reps | - |
| vector | B3_nc128 | 128 | 3 | Pack_l2_ar_channel_stall_cycle | 1.596e+05 (2997) | one speed, no slow reps | - |
| vector | B3_nc128 | 128 | 3 | Compute_l2_ar_channel_stall_cycle | 2.288e+04 (2997) | one speed, no slow reps | - |
| vector | B3_nc128 | 128 | 3 | Pack_l2_aw_channel_stall_cycle | 1.177 (2997) | one speed, no slow reps | - |
| vector | B3_nc128 | 128 | 3 | Compute_l2_aw_channel_stall_cycle | 0.08141 (2997) | one speed, no slow reps | - |
| vector | B3_nc128 | 128 | 3 | Pack_l1d_miss | 6596 (2997) | one speed, no slow reps | - |
| vector | B3_nc128 | 128 | 3 | Compute_l1d_miss | 6259 (2997) | one speed, no slow reps | - |
| vector | B3_nc128 | 128 | 3 | Pack_l1d_prefetch_refill | 8655 (2997) | one speed, no slow reps | - |
| vector | B3_nc128 | 128 | 3 | Compute_l1d_prefetch_refill | 6.102e+04 (2997) | one speed, no slow reps | - |
| vector | B3_nc128 | 128 | 3 | Pack_l1d_prefetch_hit | 9.453e+04 (2997) | one speed, no slow reps | - |
| vector | B3_nc128 | 128 | 3 | Compute_l1d_prefetch_hit | 7.479e+05 (2997) | one speed, no slow reps | - |
| vector | B3_K256 | 384 | 3 | Pack_Cycles | 6.782e+05 (1371) | 1.35e+06 (1626) | 1.99 |
| vector | B3_K256 | 384 | 3 | Compute_Cycles | 1.756e+06 (1371) | 2.688e+06 (1626) | 1.53 |
| vector | B3_K256 | 384 | 3 | Pack_l2_load_access | 9076 (1371) | 9502 (1626) | 1.05 |
| vector | B3_K256 | 384 | 3 | Compute_l2_load_access | 4.036e+04 (1371) | 3.808e+04 (1626) | 0.94 |
| vector | B3_K256 | 384 | 3 | Pack_l2_load_miss | 1165 (1371) | 3702 (1626) | 3.18 |
| vector | B3_K256 | 384 | 3 | Compute_l2_load_miss | 830.5 (1371) | 3232 (1626) | 3.89 |
| vector | B3_K256 | 384 | 3 | Pack_l2_store_access | 1.694e+04 (1371) | 1.752e+04 (1626) | 1.03 |
| vector | B3_K256 | 384 | 3 | Compute_l2_store_access | 4.179e+04 (1371) | 3.947e+04 (1626) | 0.94 |
| vector | B3_K256 | 384 | 3 | Pack_l2_ar_channel_stall_cycle | 1.407e+05 (1371) | 7.767e+04 (1626) | 0.55 |
| vector | B3_K256 | 384 | 3 | Compute_l2_ar_channel_stall_cycle | 2.59e+04 (1371) | 1.168e+04 (1626) | 0.45 |
| vector | B3_K256 | 384 | 3 | Pack_l2_aw_channel_stall_cycle | 0.9292 (1371) | 1.092 (1626) | 1.17 |
| vector | B3_K256 | 384 | 3 | Compute_l2_aw_channel_stall_cycle | 0.08753 (1371) | 0.2325 (1626) | 2.66 |
| vector | B3_K256 | 384 | 3 | Pack_l1d_miss | 2971 (1371) | 5654 (1626) | 1.90 |
| vector | B3_K256 | 384 | 3 | Compute_l1d_miss | 4330 (1371) | 1.566e+04 (1626) | 3.62 |
| vector | B3_K256 | 384 | 3 | Pack_l1d_prefetch_refill | 4952 (1371) | 2699 (1626) | 0.55 |
| vector | B3_K256 | 384 | 3 | Compute_l1d_prefetch_refill | 3.698e+04 (1371) | 2.224e+04 (1626) | 0.60 |
| vector | B3_K256 | 384 | 3 | Pack_l1d_prefetch_hit | 6.347e+04 (1371) | 3.275e+04 (1626) | 0.52 |
| vector | B3_K256 | 384 | 3 | Compute_l1d_prefetch_hit | 4.562e+05 (1371) | 2.197e+05 (1626) | 0.48 |
| vector | B2 | 384 | 3 | Pack_Cycles | 1.973e+05 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_Cycles | 2.486e+06 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Pack_l2_load_access | 4122 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_l2_load_access | 4.994e+04 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Pack_l2_load_miss | 211.1 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_l2_load_miss | 314 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Pack_l2_store_access | 7218 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_l2_store_access | 5.9e+04 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Pack_l2_ar_channel_stall_cycle | 1.633e+04 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_l2_ar_channel_stall_cycle | 294.4 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Pack_l2_aw_channel_stall_cycle | 11.04 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_l2_aw_channel_stall_cycle | 0 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Pack_l1d_miss | 889.6 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_l1d_miss | 2977 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Pack_l1d_prefetch_refill | 2254 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_l1d_prefetch_refill | 4.796e+04 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Pack_l1d_prefetch_hit | 2.4e+04 (2997) | one speed, no slow reps | - |
| vector | B2 | 384 | 3 | Compute_l1d_prefetch_hit | 6.494e+05 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Pack_Cycles | 3.017e+06 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Compute_Cycles | 2.143e+07 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Pack_l2_load_access | 2.197e+04 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Compute_l2_load_access | 3.388e+05 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Pack_l2_load_miss | 8370 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Compute_l2_load_miss | 2.315e+04 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Pack_l2_store_access | 4.132e+04 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Compute_l2_store_access | 3.472e+05 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Pack_l2_ar_channel_stall_cycle | 3.739e+05 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Compute_l2_ar_channel_stall_cycle | 7.152e+04 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Pack_l2_aw_channel_stall_cycle | 218.5 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Compute_l2_aw_channel_stall_cycle | 0.3467 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Pack_l1d_miss | 9647 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Compute_l1d_miss | 8.998e+04 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Pack_l1d_prefetch_refill | 8743 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Compute_l1d_prefetch_refill | 2.53e+05 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Pack_l1d_prefetch_hit | 7.462e+04 (2997) | one speed, no slow reps | - |
| vector | R24 | 384 | 3 | Compute_l1d_prefetch_hit | 1.85e+06 (2997) | one speed, no slow reps | - |
| vector+zeroline | B3 | 384 | 3 | Pack_Cycles | 1.309e+06 (1348) | 2.524e+06 (1649) | 1.93 |
| vector+zeroline | B3 | 384 | 3 | Compute_Cycles | 2.963e+06 (1348) | 4.354e+06 (1649) | 1.47 |
| vector+zeroline | B3 | 384 | 3 | Pack_l2_load_access | 1.416e+04 (1348) | 1.423e+04 (1649) | 1.00 |
| vector+zeroline | B3 | 384 | 3 | Compute_l2_load_access | 6.176e+04 (1348) | 5.832e+04 (1649) | 0.94 |
| vector+zeroline | B3 | 384 | 3 | Pack_l2_load_miss | 3461 (1348) | 7830 (1649) | 2.26 |
| vector+zeroline | B3 | 384 | 3 | Compute_l2_load_miss | 2204 (1348) | 5803 (1649) | 2.63 |
| vector+zeroline | B3 | 384 | 3 | Pack_l2_store_access | 3.567e+04 (1348) | 3.585e+04 (1649) | 1.00 |
| vector+zeroline | B3 | 384 | 3 | Compute_l2_store_access | 6.307e+04 (1348) | 5.958e+04 (1649) | 0.94 |
| vector+zeroline | B3 | 384 | 3 | Pack_l2_ar_channel_stall_cycle | 2.028e+05 (1348) | 9.018e+04 (1649) | 0.44 |
| vector+zeroline | B3 | 384 | 3 | Compute_l2_ar_channel_stall_cycle | 4.383e+04 (1348) | 1.865e+04 (1649) | 0.43 |
| vector+zeroline | B3 | 384 | 3 | Pack_l2_aw_channel_stall_cycle | 64.06 (1348) | 30.39 (1649) | 0.47 |
| vector+zeroline | B3 | 384 | 3 | Compute_l2_aw_channel_stall_cycle | 11.11 (1348) | 3.047 (1649) | 0.27 |
| vector+zeroline | B3 | 384 | 3 | Pack_l1d_miss | 5924 (1348) | 9364 (1649) | 1.58 |
| vector+zeroline | B3 | 384 | 3 | Compute_l1d_miss | 1.022e+04 (1348) | 2.63e+04 (1649) | 2.57 |
| vector+zeroline | B3 | 384 | 3 | Pack_l1d_prefetch_refill | 5352 (1348) | 2510 (1649) | 0.47 |
| vector+zeroline | B3 | 384 | 3 | Compute_l1d_prefetch_refill | 5.261e+04 (1348) | 3.134e+04 (1649) | 0.60 |
| vector+zeroline | B3 | 384 | 3 | Pack_l1d_prefetch_hit | 8.038e+04 (1348) | 3.36e+04 (1649) | 0.42 |
| vector+zeroline | B3 | 384 | 3 | Compute_l1d_prefetch_hit | 6.113e+05 (1348) | 3.216e+05 (1649) | 0.53 |
| vector+zeroline | B3_nc128 | 128 | 3 | Pack_Cycles | 1.698e+06 (1196) | 3.001e+06 (1801) | 1.77 |
| vector+zeroline | B3_nc128 | 128 | 3 | Compute_Cycles | 2.854e+06 (1196) | 3.683e+06 (1801) | 1.29 |
| vector+zeroline | B3_nc128 | 128 | 3 | Pack_l2_load_access | 3.18e+04 (1196) | 3.131e+04 (1801) | 0.98 |
| vector+zeroline | B3_nc128 | 128 | 3 | Compute_l2_load_access | 6.686e+04 (1196) | 6.351e+04 (1801) | 0.95 |
| vector+zeroline | B3_nc128 | 128 | 3 | Pack_l2_load_miss | 3725 (1196) | 8276 (1801) | 2.22 |
| vector+zeroline | B3_nc128 | 128 | 3 | Compute_l2_load_miss | 1027 (1196) | 1841 (1801) | 1.79 |
| vector+zeroline | B3_nc128 | 128 | 3 | Pack_l2_store_access | 4.779e+04 (1196) | 4.771e+04 (1801) | 1.00 |
| vector+zeroline | B3_nc128 | 128 | 3 | Compute_l2_store_access | 6.799e+04 (1196) | 6.459e+04 (1801) | 0.95 |
| vector+zeroline | B3_nc128 | 128 | 3 | Pack_l2_ar_channel_stall_cycle | 1.465e+05 (1196) | 6.139e+04 (1801) | 0.42 |
| vector+zeroline | B3_nc128 | 128 | 3 | Compute_l2_ar_channel_stall_cycle | 2.266e+04 (1196) | 9287 (1801) | 0.41 |
| vector+zeroline | B3_nc128 | 128 | 3 | Pack_l2_aw_channel_stall_cycle | 0.9515 (1196) | 0.7229 (1801) | 0.76 |
| vector+zeroline | B3_nc128 | 128 | 3 | Compute_l2_aw_channel_stall_cycle | 0.03595 (1196) | 0.02277 (1801) | 0.63 |
| vector+zeroline | B3_nc128 | 128 | 3 | Pack_l1d_miss | 1.732e+04 (1196) | 2.196e+04 (1801) | 1.27 |
| vector+zeroline | B3_nc128 | 128 | 3 | Compute_l1d_miss | 9473 (1196) | 2.958e+04 (1801) | 3.12 |
| vector+zeroline | B3_nc128 | 128 | 3 | Pack_l1d_prefetch_refill | 8182 (1196) | 3508 (1801) | 0.43 |
| vector+zeroline | B3_nc128 | 128 | 3 | Compute_l1d_prefetch_refill | 5.75e+04 (1196) | 3.131e+04 (1801) | 0.54 |
| vector+zeroline | B3_nc128 | 128 | 3 | Pack_l1d_prefetch_hit | 8.696e+04 (1196) | 3.614e+04 (1801) | 0.42 |
| vector+zeroline | B3_nc128 | 128 | 3 | Compute_l1d_prefetch_hit | 6.817e+05 (1196) | 3.483e+05 (1801) | 0.51 |
| vector+zeroline | B3_K256 | 384 | 3 | Pack_Cycles | 7.31e+05 (825) | 1.398e+06 (2172) | 1.91 |
| vector+zeroline | B3_K256 | 384 | 3 | Compute_Cycles | 1.81e+06 (825) | 2.753e+06 (2172) | 1.52 |
| vector+zeroline | B3_K256 | 384 | 3 | Pack_l2_load_access | 9539 (825) | 1.01e+04 (2172) | 1.06 |
| vector+zeroline | B3_K256 | 384 | 3 | Compute_l2_load_access | 4.03e+04 (825) | 3.77e+04 (2172) | 0.94 |
| vector+zeroline | B3_K256 | 384 | 3 | Pack_l2_load_miss | 1249 (825) | 3807 (2172) | 3.05 |
| vector+zeroline | B3_K256 | 384 | 3 | Compute_l2_load_miss | 973.9 (825) | 3343 (2172) | 3.43 |
| vector+zeroline | B3_K256 | 384 | 3 | Pack_l2_store_access | 2.387e+04 (825) | 2.45e+04 (2172) | 1.03 |
| vector+zeroline | B3_K256 | 384 | 3 | Compute_l2_store_access | 4.172e+04 (825) | 3.909e+04 (2172) | 0.94 |
| vector+zeroline | B3_K256 | 384 | 3 | Pack_l2_ar_channel_stall_cycle | 1.437e+05 (825) | 7.69e+04 (2172) | 0.54 |
| vector+zeroline | B3_K256 | 384 | 3 | Compute_l2_ar_channel_stall_cycle | 2.569e+04 (825) | 9477 (2172) | 0.37 |
| vector+zeroline | B3_K256 | 384 | 3 | Pack_l2_aw_channel_stall_cycle | 1.229 (825) | 1.022 (2172) | 0.83 |
| vector+zeroline | B3_K256 | 384 | 3 | Compute_l2_aw_channel_stall_cycle | 0.3818 (825) | 0.1179 (2172) | 0.31 |
| vector+zeroline | B3_K256 | 384 | 3 | Pack_l1d_miss | 3455 (825) | 6221 (2172) | 1.80 |
| vector+zeroline | B3_K256 | 384 | 3 | Compute_l1d_miss | 5069 (825) | 1.69e+04 (2172) | 3.33 |
| vector+zeroline | B3_K256 | 384 | 3 | Pack_l1d_prefetch_refill | 4891 (825) | 2685 (2172) | 0.55 |
| vector+zeroline | B3_K256 | 384 | 3 | Compute_l1d_prefetch_refill | 3.608e+04 (825) | 2.058e+04 (2172) | 0.57 |
| vector+zeroline | B3_K256 | 384 | 3 | Pack_l1d_prefetch_hit | 6.27e+04 (825) | 3.208e+04 (2172) | 0.51 |
| vector+zeroline | B3_K256 | 384 | 3 | Compute_l1d_prefetch_hit | 4.386e+05 (825) | 2.107e+05 (2172) | 0.48 |
| vector+zeroline | B2 | 384 | 3 | Pack_Cycles | 2.039e+05 (2997) | one speed, no slow reps | - |
| vector+zeroline | B2 | 384 | 3 | Compute_Cycles | 2.5e+06 (2997) | one speed, no slow reps | - |
| vector+zeroline | B2 | 384 | 3 | Pack_l2_load_access | 4376 (2997) | one speed, no slow reps | - |
| vector+zeroline | B2 | 384 | 3 | Compute_l2_load_access | 5.059e+04 (2997) | one speed, no slow reps | - |
| vector+zeroline | B2 | 384 | 3 | Pack_l2_load_miss | 184.4 (2997) | one speed, no slow reps | - |
| vector+zeroline | B2 | 384 | 3 | Compute_l2_load_miss | 267.9 (2997) | one speed, no slow reps | - |
| vector+zeroline | B2 | 384 | 3 | Pack_l2_store_access | 9841 (2997) | one speed, no slow reps | - |
| vector+zeroline | B2 | 384 | 3 | Compute_l2_store_access | 5.965e+04 (2997) | one speed, no slow reps | - |
| vector+zeroline | B2 | 384 | 3 | Pack_l2_ar_channel_stall_cycle | 6438 (2997) | one speed, no slow reps | - |
| vector+zeroline | B2 | 384 | 3 | Compute_l2_ar_channel_stall_cycle | 82.21 (2997) | one speed, no slow reps | - |
| vector+zeroline | B2 | 384 | 3 | Pack_l2_aw_channel_stall_cycle | 5.223 (2997) | one speed, no slow reps | - |
| vector+zeroline | B2 | 384 | 3 | Compute_l2_aw_channel_stall_cycle | 0 (2997) | one speed, no slow reps | - |
| vector+zeroline | B2 | 384 | 3 | Pack_l1d_miss | 1272 (2997) | one speed, no slow reps | - |
| vector+zeroline | B2 | 384 | 3 | Compute_l1d_miss | 3385 (2997) | one speed, no slow reps | - |
| vector+zeroline | B2 | 384 | 3 | Pack_l1d_prefetch_refill | 2151 (2997) | one speed, no slow reps | - |
| vector+zeroline | B2 | 384 | 3 | Compute_l1d_prefetch_refill | 4.877e+04 (2997) | one speed, no slow reps | - |
| vector+zeroline | B2 | 384 | 3 | Pack_l1d_prefetch_hit | 2.481e+04 (2997) | one speed, no slow reps | - |
| vector+zeroline | B2 | 384 | 3 | Compute_l1d_prefetch_hit | 6.246e+05 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Pack_Cycles | 2.804e+06 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Compute_Cycles | 2.067e+07 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Pack_l2_load_access | 2.206e+04 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Compute_l2_load_access | 3.402e+05 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Pack_l2_load_miss | 7125 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Compute_l2_load_miss | 2.153e+04 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Pack_l2_store_access | 5.954e+04 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Compute_l2_store_access | 3.486e+05 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Pack_l2_ar_channel_stall_cycle | 4.312e+05 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Compute_l2_ar_channel_stall_cycle | 8.789e+04 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Pack_l2_aw_channel_stall_cycle | 440.5 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Compute_l2_aw_channel_stall_cycle | 0.6159 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Pack_l1d_miss | 8684 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Compute_l1d_miss | 7.992e+04 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Pack_l1d_prefetch_refill | 9610 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Compute_l1d_prefetch_refill | 2.697e+05 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Pack_l1d_prefetch_hit | 8.737e+04 (2997) | one speed, no slow reps | - |
| vector+zeroline | R24 | 384 | 3 | Compute_l1d_prefetch_hit | 2.268e+06 (2997) | one speed, no slow reps | - |
