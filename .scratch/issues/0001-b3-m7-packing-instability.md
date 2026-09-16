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
- **Hypothesis A (Packer Instruction & Register Spill):** The RVV packer in `gemm_pack_xdsl.cpp` uses two passes of `vlse32`/`vse32` per row for $NR=15$ (`[0..8)` then `[8..15)`). The interleaved stride loads/stores thrash the L2 cache lines and tie up registers. **⚠️ See §6 — this two-pass, LMUL=1-only structure is not an incidental artifact; it is the *only* code path the live generator has, and it already matches the "throttled" mitigation once proposed as the fix (§6.2). It does not resolve the instability.**
- **Hypothesis B (Cache Eviction of $Bc$):** For $nc=384, kc=128$, $Bc$ is 196 KB. In a 512 KB 8-way L2 cache, the high-miss burst of strided loads from $B$ (576 KB, outside L2) completely evicts the freshly written $Bc$ lines before the microkernel executes. With scalar packing, the slower pace lets prefetchers keep $Bc$ hot. **⚠️ See §6.3 — the "8-way" premise is unverified against real hardware and appears to be wrong (real L2 is 16-way per sysfs); the model header also disagrees with itself (`WL2 8` with a commented-out correct `16` directly above it).**
- **Hypothesis C (Unrolling Dimension Search):** In the user's historical search (`debug perfromance differnces...`), kernels with smaller $NR$ (such as $MR16\_NR7$ and $MR16\_NR6$) reached **10.15 KerGOPs** because narrower packing passes reduce instantaneous cache pressure.

---

## 4. One-Command Reproduction Loop
```bash
source .venv/bin/activate && REMOTE_PASS="" REMOTE_HOST="fej" REMOTE_PORT="3322" \
./tests/pure_xdsl_blis_rvv/compile_macro_bench.sh --jump --riscv-user jlei --riscv-ip jbpi --riscv-port 22
```

---

## 6. Agent Investigation Findings (2026-09-16)

Verified directly against the live generator output and the real target hardware (not just inference from GOPs numbers). Session: this conversation.

### 6.1 The debug docs point at the wrong generator file
There are four different copies of `pack_builder.py` in this repo (`tests/packing/`, `tests/pure_xdsl_blis_rvv/`, `tests/gemm_blis_xdsl/`, plus `update_pack_builder_safe.py` at repo root). `debug_unstablePerf/implementation_plan.md` proposes patching `tests/packing/pack_builder.py` (untouched since 2026-07-11). But `generate_pure_blis_project.py` (in `tests/pure_xdsl_blis_rvv/`) does a bare `from pack_builder import build_pack_module`, which resolves to **`tests/pure_xdsl_blis_rvv/pack_builder.py`** — a different, independently-written file added in the same 2026-09-03 checkpoint commit (`4d6b1d9`) as the live repro data in §1. The plan targets a file the live pipeline doesn't even import.

### 6.2 The proposed "LMUL throttling" fix is already the only code path — and it doesn't fix the bug
Regenerated the project fresh (`python3 generate_pure_blis_project.py --config config.json`) and diffed against `git HEAD` for `pj_gen/fp32_AXPY_FP32_Pack_vlen256/gemm_pack_xdsl.cpp`: **byte-identical**, confirming this is genuinely what gets compiled and benchmarked.

The real `build_pack_CB_v_if` (`tests/pure_xdsl_blis_rvv/pack_builder.py:50-87`) has no LMUL=2 option at all:
```python
num_vecs = (RR + vlen - 1) // vlen   # vlen=8 elements (fp32, VLEN=256)
for v in range(num_vecs):
    ...vlse32_v_f32m1Op...           # always LMUL=1, chunked
```
For $NR=15$ this always emits two chunked LMUL=1 passes (≤8 elements each) — exactly the mitigation `implementation_plan.md` proposes as an untried fix ("reduce LMUL, break 16-element bursts into 8-element chunks"). **It is not untried.** The §1 live benchmark (0.5 GB/s RVV-pack bandwidth, 16–27% CV) was measured with this already-chunked, already-LMUL1-only packer. LMUL reduction alone is a dead end for fixing this instability.

### 6.3 The hardware doesn't match what the reports assumed
SSH'd directly into `jbpi` and `jbpi2` (both live, identical specs):
- `model name: Spacemit(R) X60` — a **RISC-V** core (`uname -a` → `riscv64`), not the "ARM Cortex-A55" that `INVESTIGATION_B3_M7_PERF.md` and `PACKING_INSTABILITY_REPORT.md` describe. Part of the hardware narrative in those docs is simply about the wrong CPU family.
- Real L2 cache via `/sys/devices/system/cpu/cpu0/cache/index2`: **512K, 16-way associative** — but `tests/pure_xdsl_blis_rvv/modelLevel/archs/bananapi.h:26` hardcodes `#define WL2 8`, with `//#define WL2 16` (the value that matches real hardware) commented out directly above it. The "8-way associativity conflict / pseudo-LRU chaos" hypothesis (Hypothesis B here, and Hyp. 2/4 in `PACKING_INSTABILITY_REPORT.md`) is built on a cache model that disagrees with the real chip, which has double the real associativity.
- No L3 cache reported in sysfs (only L1I/L1D/L2 exist) — this confirms what the docs already independently concluded (the `nc` heuristic assumes an L3 that doesn't physically exist on this board).
- **No `perf` binary on the board** (Bianbu 2.2.1 minimal image). `objdump`/`gcc`/`readelf`/`taskset`/`chrt` are present; CPU governor is already pinned to `performance`. **No hypothesis in this issue about L2 eviction, pseudo-LRU behavior, or cache-miss counts has ever been backed by an actual hardware counter** — all of it is inferred from GOPs throughput numbers alone.

### 6.4 Revised priorities
1. Fix `WL2` in `bananapi.h` to `16` (matches real hardware) and see whether `get_optim_mc_nc_kc` picks a different `nc`/`kc` for `B3_M7` — this directly drives the buffer size the rest of the theory depends on. Cheap, do first.
2. Get real cache-miss counts. No `perf` CLI on-board; either install `linux-tools`/`perf` (needs board network + package availability, unverified) or write a small harness using raw `perf_event_open` syscalls. This is the only way to confirm or kill the eviction-chaos theory instead of continuing to infer it from throughput.
3. Retest the one lever the reports partially validated but never turned into a permanent fix: forcing `nc=128`/`192` for `B3_M7` (already showed CV 25%→13% in the `PACKING_INSTABILITY_REPORT.md` nc=192 experiment).
4. Port whichever mitigation works from `tests/pure_xdsl_blis_rvv/pack_builder.py` (confirm `tests/packing/pack_builder.py` and `tests/gemm_blis_xdsl/pack_builder.py` are dead code before touching them, or reconcile/delete the duplicates — four divergent copies of the same generator is itself a latent bug).

---

## 5. Next Steps for Agent Implementation (`/diagnosing-bugs`)
1. [x] ~~Compare side-by-side run of `--scalar-pack` vs RVV pack to verify if scalar pack eliminates the bimodal drops.~~ Superseded — see §6.4; scalar-vs-RVV comparison already captured in §1/§2, the open question is now *why* both remain memory-bound, not whether they differ.
2. [x] Profile/inspect the emitted assembly of `gemm_pack_xdsl.cpp` for $NR=15$ — done in §6.2; it's already LMUL=1, chunked, matching the once-proposed fix.
3. [ ] ~~Explore single-pass vector packing~~ — already the *only* path (§6.2), doesn't help. Instead: fix `WL2` (§6.4.1), get real perf counters (§6.4.2), retest smaller `nc` (§6.4.3).
4. [ ] Implement regression test ensuring $CV\% < 5\%$ and KerGOPs $> 10$ on $B3\_M7$.
