# Issue #1: Performance Instability in B3_M7 (RVV Packing vs Scalar Packing & Reference BLIS)

**Role / Label:** `needs-info`  
**Target:** `jbpi2` (BananaPi M2 Pro, VLEN=256)  
**Shape:** `B3_M7` ($64 \times 384 \times 384$, K-large, M-small), Kernel `MR16_NR15`  
**Primary Sources:**
- `tests/pure_xdsl_blis_rvv/debug_unstablePerf/BUG_REPORT_B3_M7_INSTABILITY.md`
- `tests/pure_xdsl_blis_rvv/debug_unstablePerf/INVESTIGATION_B3_M7_PERF.md`
- `tests/pure_xdsl_blis_rvv/debug_unstablePerf/PACKING_INSTABILITY_REPORT.md`

---

## Summary
The pure xDSL BLIS+RVV design produces bimodal and degraded performance on the `B3_M7` layer:
- **Reference BLIS:** ~8.2–8.8 GOPs, CV% < 1% (stable)
- **Scalar Packing:** ~10.0–12.5 GOPs, CV% 2–19% (moderate variance)
- **RVV Packing:** ~7.2–7.5 GOPs, CV% 17–30% (bimodal: drops to ~6.4 GOPs, occasional spikes to ~12.7 GOPs)

All other benchmark layers (B2_M5, L3–L7) show stable performance under both packing modes.

---

## What Has Been Verified / Ruled Out
1. **Cache Blocking Parameters:** Confirmed identical between xDSL and reference (`mc:64, nc:384, kc:128`).
2. **`memset` Cache Flush:** Initial inner-loop full buffer `memset` was replaced by edge-zeroing (`cur_nc % nr != 0`), boosting peak throughput from ~8 to ~13 GOPs.
3. **Hardware Specs:** BananaPi M2 Pro L1 (32 KB), L2 (512 KB), L3 (4 MB) confirmed correct against upstream.

---

## Open Suspects
- **L2 Cache Eviction by RVV Packer:** The RVV packer uses two vector passes per KC row (`vlse32`/`vse32`) across `Bc` (196 KB in a 512 KB L2 cache). The burst of strided memory loads and stores may evict `Bc` cache lines before the microkernel starts reading them.
- **Interplay with Uncommitted Changes:** Local code contains newer generator and benchmark logic that has not yet been benchmarked post-holiday.

---

## Next Steps to Unblock (`needs-info`)
1. [ ] Rerun `compile_macro_bench.sh` on `jbpi2` using the latest uncommitted working tree.
2. [ ] Capture current runtime measurements: verify if bimodal behavior persists on the latest code.
3. [ ] If still unstable, transition role to `ready-for-agent` and launch `/diagnosing-bugs` to isolate the packer instruction schedule.
