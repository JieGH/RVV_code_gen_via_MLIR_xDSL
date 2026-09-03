# Issue #1: Performance Instability in B3_M7 (RVV Packing vs Scalar Packing & Reference BLIS)

**Role / Label:** `ready-for-agent`  
**Target:** `jbpi` / `jbpi2` (BananaPi M2 Pro, VLEN=256)  
**Shape:** `B3_M7` ($64 \times 384 \times 384$, K-large, M-small), Kernel `MR16_NR15`  
**Primary Sources:**
- `tests/pure_xdsl_blis_rvv/debug perfromance differnces 3a968a923c2a8043b3c4fdf2c741fe55 3.md`
- `tests/pure_xdsl_blis_rvv/debug_unstablePerf/BUG_REPORT_B3_M7_INSTABILITY.md`
- `tests/pure_xdsl_blis_rvv/debug_unstablePerf/INVESTIGATION_B3_M7_PERF.md`
- `tests/pure_xdsl_blis_rvv/debug_unstablePerf/PACKING_INSTABILITY_REPORT.md`
- Log run: `tests/pure_xdsl_blis_rvv/logs/fp32_AXPY_FP32_Pack_vlen256_2026-09-03_122419/`

---

## 1. Executive Summary & Live Reproduction (2026-09-03)
The pure xDSL BLIS+RVV design produces severe bimodal instability and performance degradation specifically on the `B3_M7` layer ($64 \times 384 \times 384$).

### Live Benchmark Evidence (`testCase_repeat.dat`, 1000 reps on `jbpi`):

| Layer / Variant | Dimensions | Median KerGOPs | Min / Max | CV% | mx/md | RVV Pack | Scalar Pack |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **B2_M5 (10 runs)** | $384 \times 384 \times 64$ | **13.11 – 13.17** | 7.2 / 13.3 | **1% – 3%** | **1.01** | 2.0 – 2.2 GB/s | 1.3 GB/s |
| **B3_M7 (10 runs)** | $64 \times 384 \times 384$ | **7.15 – 7.41** | 4.6 / 13.0 | **16% – 27%** | **1.75 – 1.82** | **0.5 GB/s** | **0.8 – 1.0 GB/s** |
| **B3_M7_Reuse8** | $128 \times 384 \times 384$ | **7.99 – 8.05** | 5.1 / 12.2 | **17% – 21%** | **1.50** | 0.5 GB/s | 0.8 GB/s |
| **B3_M7_Reuse12** | $192 \times 384 \times 384$ | **8.57 – 8.65** | 6.3 / 11.8 | **14% – 15%** | **1.36** | 0.6 GB/s | 0.9 GB/s |
| **B3_M7_Reuse24** | $384 \times 384 \times 384$ | **9.57 – 9.81** | 6.0 / 10.1 | **7% – 9%** | **1.04** | 0.7 – 0.9 GB/s | 1.0 GB/s |

---

## 2. Key Observations
1. **The Issue is Bimodal, Not Random Noise:**
   - `B2_M5` is rock-solid (CV 1–3%, max/median 1.01).
   - `B3_M7` has a median cluster around ~7.2 GOPs with random spikes to ~13.0 GOPs, causing CV 27% and mx/md 1.82.
2. **$M$-Reuse Direct Correlation:**
   - As $M$ increases from $64 \to 128 \to 192 \to 384$, the packed $Bc$ buffer is reused more times ($M / MR = 4 \to 8 \to 12 \to 24$ microkernel calls per pack).
   - Consequently, CV% drops steadily from **27% down to 7%**, and median performance climbs from **7.2 to 9.8 GOPs**.
3. **RVV Packing Bandwidth Inversion:**
   - On `B3_M7`, **RVV Packing throughput is only 0.5 GB/s**, whereas **Scalar Packing throughput is 0.8–1.0 GB/s**!
   - On `B2_M5`, RVV Packing is faster (2.2 GB/s vs 1.3 GB/s).
   - On `B3_M7`, the RVV packer is severely degraded and actively slower than scalar code.

---

## 3. Root Cause Hypotheses
- **Hypothesis A (Packer Instruction & Register Spill):** The RVV packer in `gemm_pack_xdsl.cpp` uses two passes of `vlse32`/`vse32` per row for $NR=15$ (`[0..8)` then `[8..15)`). The interleaved stride loads/stores thrash the L2 cache lines and tie up registers.
- **Hypothesis B (Cache Eviction of $Bc$):** For $nc=384, kc=128$, $Bc$ is 196 KB. In a 512 KB 8-way L2 cache, the high-miss burst of strided loads from $B$ (576 KB, outside L2) completely evicts the freshly written $Bc$ lines before the microkernel executes. With scalar packing, the slower pace lets prefetchers keep $Bc$ hot.
- **Hypothesis C (Unrolling Dimension Search):** In the user's historical search (`debug perfromance differnces...`), kernels with smaller $NR$ (such as $MR16\_NR7$ and $MR16\_NR6$) reached **10.15 KerGOPs** because narrower packing passes reduce instantaneous cache pressure.

---

## 4. One-Command Reproduction Loop
```bash
source .venv/bin/activate && REMOTE_PASS="" REMOTE_HOST="fej" REMOTE_PORT="3322" \
./tests/pure_xdsl_blis_rvv/compile_macro_bench.sh --jump --riscv-user jlei --riscv-ip jbpi --riscv-port 22
```

---

## 5. Next Steps for Agent Implementation (`/diagnosing-bugs`)
1. [ ] Compare side-by-side run of `--scalar-pack` vs RVV pack to verify if scalar pack eliminates the bimodal drops.
2. [ ] Profile/inspect the emitted assembly of `gemm_pack_xdsl.cpp` for $NR=15$.
3. [ ] Explore single-pass vector packing or cache-conscious loop ordering for $B3\_M7$.
4. [ ] Implement regression test ensuring $CV\% < 5\%$ and KerGOPs $> 10$ on $B3\_M7$.
