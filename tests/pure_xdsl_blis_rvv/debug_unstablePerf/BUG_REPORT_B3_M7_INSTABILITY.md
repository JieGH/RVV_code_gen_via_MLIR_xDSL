# Bug Report: B3_M7 Performance Instability in pure xDSL BLIS Design

**Date:** 2026-07-27  
**Reporter:** jlei  
**Status:** Open — root cause identified, fix not yet confirmed  
**Severity:** High — 30% performance gap and 20–27% CV% on a real BERT-Large workload layer

---

## Summary

The pure xDSL BLIS+RVV design produces **unstable and underperforming** kernel GOPs for the
`B3_M7` workload shape (`64×384×384`, K-large, M-small). All other shapes tested show ≤5% gap
vs the reference and <2% CV%. This specific shape is an outlier.

---

## Reproduction

**Hardware:** BananaPi M2 Pro (RISC-V, `jbpi2`), VLEN=256  
**Design:** `tests/pure_xdsl_blis_rvv/pj_gen/fp32_AXPY_FP32_Pack_vlen256/`  
**Kernel:** `MR16_NR15` (forced, to match reference)  
**Command:**
```bash
REMOTE_PASS="" REMOTE_HOST="fej" REMOTE_PORT="3322" \
./tests/pure_xdsl_blis_rvv/compile_macro_bench.sh \
    --config fp32_axpy_config.json \
    --jump \
    --riscv-user jlei --riscv-ip jbpi2 --riscv-port 22
```

---

## Observed vs Expected

| Shape | Ref GOPs | xDSL GOPs | Gap | xDSL CV% |
|-------|----------|-----------|-----|-----------|
| B2_M5 (384×384×64) | ~12.5 | ~12.9 | **+3%** | **1%** ✅ |
| B3_M7 (64×384×384) | ~8.8  | ~7.0 median | **−20–30%** | **20–27%** ❌ |

**Bimodal distribution observed for B3_M7:**
```
B3_M7_2: median=7.22, min=6.4, p5=6.9, p95=12.6, max=12.8, CV=24%
```
Most runs cluster at ~7 GOPs, some randomly spike to ~12.7 GOPs.
This is a two-state behavior, NOT random noise.

---

## Root Cause Analysis

### Why B3_M7 is uniquely affected: Geometry

| Shape | K | B panel in memory (k×n×4 bytes) | Fits in L2 (512 KB)? |
|-------|----|--------------------------------|----------------------|
| B2_M5 | 64 | 64×384×4 = **98 KB** | ✅ Yes |
| B3_M7 | 384 | 384×384×4 = **576 KB** | ❌ No — L3 only |

When the B operand overflows L2, every microkernel call must fetch data from L3/DDR.
This makes the kernel time highly sensitive to L3 eviction and OS scheduler interference.

### Suspect 1 — `memset(Bc)` inside the KC loop (HIGH CONFIDENCE)

In `gemm_dispatch_xdsl.cpp`, inside the KC loop:
```cpp
for (int32_t pc = 0; pc < k; pc += kc) {
    ...
    memset(Ac, 0, mc_pad * kc_pad * sizeof(float));  // mc × kc
    memset(Bc, 0, kc_pad * nc_pad * sizeof(float));  // kc × nc  <-- THIS
    ...
}
```
If `nc` = full N = 384 and `kc` = 384 (or even 128), `memset(Bc)` writes
between 98 KB and 576 KB of zeros to memory on **every iteration of the outer
loop**. This aggressively evicts L2 on every pass, guaranteeing a cold-cache
miss for Bc on the next microkernel call.

The reference design (`gemm_blis.c`) does NOT zero Bc unconditionally — it only
packs into it, touching each byte exactly once.

### Suspect 2 — KC calculation divergence (MEDIUM CONFIDENCE — NEEDS VERIFICATION)

Both the xDSL driver and the reference call `get_optim_mc_nc_kc()` from
`modelLevel/model_level.c`. For BananaPi with MR=16, NR=15, FP32:
- Expected KC ≈ 64–128 (from L1 analytical model)
- `*kc = min(k, *kc)` caps at actual K

If something causes `kc = k = 384` in the xDSL path (e.g. wrong MR/NR passed
to `get_optim_mc_nc_kc`), the KC blocking collapses entirely, making the B
panel 576 KB on every loop iteration.

**Verification needed:** Print actual kc, mc, nc for B3_M7 at runtime in both designs.

### Suspect 3 — NC too large for B3_M7 (LOWER CONFIDENCE)

The L3 model calculates `nc` assuming all of L3 (4 MB) is available. For B3_M7,
a large `nc` combined with `kc` creates a Bc buffer that may exceed L3 on busy
systems, making performance dependent on OS memory pressure.

---

## Evidence Timeline

| Date | Observation |
|------|-------------|
| ~Jul 16 | prev_xdsl reference shows B3_M7 at ~8.8 GOPs (16×15) |
| ~Jul 23 | New pure xDSL design: B3_M7 at 7.24 GOPs (16×15) — 30% gap found |
| ~Jul 26 | Forced 16×15 on pure xDSL: median 7–8.7 GOPs, high CV% confirmed |
| ~Jul 26 | Repeat runs confirm bimodal: most at ~7 GOPs, some at ~12.7 GOPs |
| ~Jul 26 | ref LDX macro benchmark also shows instability for B3 (CV 10–17%) |
| ~Jul 27 | Confirmed: B2_M5 (small K) is stable in all designs → K-size is the differentiator |

---

## Impact

- B3_M7 is a real BERT-Large layer: `[64, 384, 384]` (attention projection).
- A 20–30% kernel GOPs gap directly reduces transformer inference throughput.
- The instability (CV% 20–27%) makes it impossible to reliably benchmark or compare designs.

---

## Steps to Confirm Root Cause

1. **Print kc, mc, nc at runtime** for B3_M7 in the xDSL driver.
   Compare against reference output. If kc differs, fix the call site.

2. **Remove or move the `memset(Bc)`** outside the microkernel timing window
   in `gemm_dispatch_xdsl.cpp`, or reduce its scope to only zero the fringe
   padding (not the entire buffer). Rerun B3_M7 and check if CV% drops.

3. **Manually force kc=64 for B3_M7** and measure again. If median GOPs
   rises to ~12 and CV% drops to <2%, KC blocking is confirmed as the fix.

---

## Files Involved

| File | Role |
|------|------|
| `pj_gen/fp32_AXPY_FP32_Pack_vlen256/gemm_dispatch_xdsl.cpp` | Contains the KC loop with suspect `memset(Bc)` |
| `pj_gen/fp32_AXPY_FP32_Pack_vlen256/driver_benchmark.cpp` | Calls `get_optim_mc_nc_kc` — verify correct MR/NR passed |
| `modelLevel/model_level.c` | Reference KC analytical model — known correct |
| `modelLevel/archs/bananapi.h` | Cache parameters for jbpi2: L1=32KB/4-way, L2=512KB/8-way, L3=4MB/16-way |
| `tests/pure_xdsl_blis_rvv/test_ext_ldx_macc/` | Standalone ref harness created for this investigation |

---

## Related Notes

- Full investigation context: `tests/pure_xdsl_blis_rvv/INVESTIGATION_B3_M7_PERF.md`
- Reference design: `ext/RVV_ukernels_benchmark` (adcastel GitHub repo, commit eb84ded)
- Cache model paper: *"Analytical Modeling is Enough for High Performance BLIS"* (T.M. Low et al., 2016)
