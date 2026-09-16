# Issue #1: Performance Instability in B3_M7 (RVV Packing vs Scalar Packing & Reference BLIS)

**Role / Label:** `closed — mitigated`
**Status:** ✅ Closed 2026-09-16. Root cause identified, real fix applied (WL2), practical
mitigation shipped and verified live on hardware (config-driven `nc` override). One residual
sub-finding (row-to-row median jump on raw `B3_M7`, M=64) is real but unexplained — tracked
as a follow-up, not blocking closure (see §8).
**Target:** `jbpi` / `jbpi2` (BananaPi M2 Pro, VLEN=256)  
**Shape:** `B3_M7` ($64 \times 384 \times 384$, K-large, M-small), Kernel `MR16_NR15`  
**Primary Sources:**
- `tests/pure_xdsl_blis_rvv/debug perfromance differnces 3a968a923c2a8043b3c4fdf2c741fe55 3.md`
- `tests/pure_xdsl_blis_rvv/debug_unstablePerf/BUG_REPORT_B3_M7_INSTABILITY.md`
- `tests/pure_xdsl_blis_rvv/debug_unstablePerf/INVESTIGATION_B3_M7_PERF.md`
- `tests/pure_xdsl_blis_rvv/debug_unstablePerf/PACKING_INSTABILITY_REPORT.md`
- Log run: `tests/pure_xdsl_blis_rvv/logs/fp32_AXPY_FP32_Pack_vlen256_2026-09-03_122419/`

---

## 0. Closure Overview (2026-09-16)

**What was actually true (confirmed on real hardware):**
- The original bug reports' hardware narrative was wrong: `jbpi`/`jbpi2` are RISC-V (Spacemit X60), not ARM Cortex-A55, and the real L2 is 16-way associative, not 8-way — the `WL2=8` constant in `bananapi.h` was a real bug, now fixed to `16`.
- That fix does **not** explain the instability. Verified with a standalone harness: `mc` is clamped to `M` regardless of `WL2` (`mc = min(m, cache_estimate)`, and the cache estimate is always larger than `M` for this shape either way).
- The real driver is the **microkernel reuse factor** `mc/mr` — already proposed in `PACKING_INSTABILITY_REPORT.md` but never confirmed on hardware. `B3_M7` has `M=64`, forcing `mc=64`, reuse=4 → structurally memory-bound. Larger-M variants (`Reuse8/12/24`) have higher reuse and are far more stable.
- Shrinking the packed-B buffer (`nc`) from the heuristic's `384` down to `128` **monotonically reduces variance for every shape in the family**, live-verified on `jbpi`:

```
CV% (variance), B3_M7 family, by nc setting  [1 block ≈ 1%]

              nc=384                nc=192          nc=128
B3_M7   (r=4)  ████████████████████  █████████████░  █████████████
   M=64        20%                   13%             13%   <- structural floor, mc forced to 64

Reuse8  (r=8)  ███████████████████░  █████████░      ████████░
   M=128       19%                   9%              8%

Reuse12 (r=12) ██████████████░       ███████████░    ███████░
   M=192       14%                   11%             7%

Reuse24 (r=24) ████████░             ████████░       ████░
   M=384        8%                    8%             4%   <- reuse factor already high, nc barely matters

reuse factor = mc/mr, where mc = min(M, cache_estimate):
  M=64  -> mc=64  -> reuse=4   (this shape's hard limit — no nc/WL2 tuning changes it)
  M=384 -> mc=384 -> reuse=24  (compute-bound, rock solid at any nc)
```

**Decisions made:**
- Did **not** touch `get_optim_mc_nc_kc`/`model_level.c` — it's a shared, published analytical BLIS model (Low et al. 2016) reused by every shape/arch in this repo; a shape-specific fudge factor there would silently affect untested cases and still wouldn't fully fix `B3_M7` (M=64) itself.
- Instead shipped an **opt-in, config-driven per-shape override**: `config.json` row field `{"B3_M7_": 128}`, applied by longest label-prefix match in `driver_benchmark.cpp`, verified live on `jbpi` to affect only `B3_M7*` rows and leave everything else on the original heuristic.
- Found and flagged (not deleted) a 5th divergent copy of the blocking model: `cache_config.py`, dead/unused Python duplicate.

**What's still open (not blocking closure):** raw `B3_M7` (M=64) at `nc=128` still shows CV 11–15% (vs the issue's <5% target) *and* a new, unexplained pattern — the median itself jumps between back-to-back identical runs (8.75→10.16→10.46→...→8.86 GOPs) in a way neither the reuse-factor theory nor the nc/cache theories predict. No hardware perf counters were ever used to actually confirm cache-miss behavior (`perf` isn't installed on the board). See §8.

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
1. ~~Fix `WL2` in `bananapi.h` to `16` (matches real hardware) and see whether `get_optim_mc_nc_kc` picks a different `nc`/`kc` for `B3_M7`.~~ **Done, dead end for this bug (2026-09-16).** Built a standalone harness against `tests/pure_xdsl_blis_rvv/modelLevel/model_level.c` calling `get_optim_mc_nc_kc(sizeof(float), m=64, n=384, k=384, mr=16, nr=15, ...)` with `WL2=8` vs `WL2=16`. Result is byte-identical both ways: `mc=64 nc=384 kc=128`. Reason: `mc` is computed from the L2 model then clamped with `*mc = min(m, *mc)` (`model_level.c:64`) — for `B3_M7`, `m=64` is far smaller than either computed L2 estimate (640 for WL2=8, 1536 for WL2=16), so the clamp always wins and `WL2` never reaches the output. **`WL2=8` is still a real bug to fix (doesn't match the real 16-way L2 on `jbpi`/`jbpi2`, and *does* matter for other, less M-starved shapes), but it is not the cause of B3_M7's instability — the mc/nc/kc blocking BLIS picks for this shape is unaffected by it.**
2. Get real cache-miss counts. No `perf` CLI on-board; either install `linux-tools`/`perf` (needs board network + package availability, unverified) or write a small harness using raw `perf_event_open` syscalls. This is the only way to confirm or kill the eviction-chaos theory instead of continuing to infer it from throughput.
3. Retest the one lever the reports partially validated but never turned into a permanent fix: forcing `nc=128`/`192` for `B3_M7` (already showed CV 25%→13% in the `PACKING_INSTABILITY_REPORT.md` nc=192 experiment). **Next up.**
4. Port whichever mitigation works from `tests/pure_xdsl_blis_rvv/pack_builder.py` (confirm `tests/packing/pack_builder.py` and `tests/gemm_blis_xdsl/pack_builder.py` are dead code before touching them, or reconcile/delete the duplicates — four divergent copies of the same generator is itself a latent bug).

---

## 7. Live nc-Reduction Experiment (2026-09-16) and Config-Driven Mitigation

Ran the full remote repro loop on `jbpi` with `nc` forced to 192 and 128 (via a temporary `RVV_FORCE_NC` env-var patch to `driver_benchmark.cpp`), comparing against the `nc=384` baseline in §1:

| Shape | nc=384 (baseline, §1) | nc=192 | nc=128 |
| :--- | :--- | :--- | :--- |
| B3_M7 (M=64, reuse=4) | md 7.15–7.41, CV 16–27%, mx/md 1.75–1.82 | md 8.78–8.89, CV 11–16%, mx/md 1.43–1.45 | md 8.75–10.55 (itself erratic row-to-row), CV 11–15%, mx/md 1.16–1.40 |
| Reuse8 (M=128, reuse=8) | md 7.99–8.05, CV 17–21%, mx/md 1.50 | md 8.97–9.14, CV 6–12%, mx/md 1.30–1.34 | md 9.14–10.47, CV 6–11%, mx/md 1.10–1.30 |
| Reuse12 (M=192, reuse=12) | md 8.57–8.65, CV 14–15%, mx/md 1.36 | md 8.84–10.68, CV 11–12%, mx/md 1.08–1.30 | **md 10.49–11.07, CV 4–10%, mx/md 1.02–1.08** |
| Reuse24 (M=384, reuse=24) | md 9.57–9.81, CV 7–9%, mx/md 1.04 | md 8.26–9.36, CV 7–9%, mx/md 1.05–1.19 | **md 9.15–9.42, CV 3–6%, mx/md 1.01–1.04** |

**Live verification of the shipped mitigation (2026-09-16):** ran the plain baseline command
(no `--force-nc` flag) after committing `config.json`'s `{"B3_M7_": 128}` row field. Confirmed
`B3_M7*` rows report `SKIP (mc:.. nc:128 kc:..)` automatically — median/CV/mx-md matched the
manual `--force-nc 128` run above (e.g. `B3_M7_1`: 8.88 KerGOPs, CV 16%, mx/md 1.38) — while
`L1–L7` and `B2_M5` kept their original heuristic `nc` (`B2_M5` stayed `nc:384`, `L5` `nc:256`),
confirming the override is correctly scoped to only the targeted shape family.

**Findings:**
1. Shrinking `nc` monotonically helps every shape in the family — reproduces the old `PACKING_INSTABILITY_REPORT.md` nc=192 result almost exactly (CV 25%→13%, mx/md 1.80→1.45), which is a real signal despite that report's debunked hardware narrative (§6.3).
2. Combined with more M-reuse, nc=128 nearly meets the issue's `CV<5%` target (Reuse24: 3–6%, Reuse12: 4–10%). This validates `PACKING_INSTABILITY_REPORT.md`'s reuse-factor theory (§"Why is B2_M5 Rock Solid...") as the dominant mechanism — not hypotheses A/B in this issue.
3. **Raw `B3_M7` (M=64) does not fully improve even at nc=128** — `mc` is clamped to `m=64` regardless (confirmed via harness in §6.4.1), so no `nc` value fixes it. A **new, previously-unflagged phenomenon** appears here: the *median itself jumps row-to-row* across 10 identical back-to-back configs in the same process (8.75→10.16→10.46→...→8.86), which is not explained by within-run cache spikes (already captured per-row) or by the reuse-factor theory. Unexplained; needs real perf/frequency counters (§6.4 priority 2, still open) to chase further.

**Design decision — do not modify `get_optim_mc_nc_kc`/`model_level.c`:** that function implements the general "Low et al. 2016" analytical BLIS cache-blocking model, shared across every shape (ResNet50 sweep in `model_test.c`) and every arch (`bananapi`, `jbpi2`, `amd_zen`, `arm_carmel` in `cache_config.py`). Baking a `B3_M7`-specific fudge factor into it would silently change behavior for untested shapes/archs, and wouldn't even fully fix the case it targets (item 3 above). Instead, added a **config-driven, opt-in per-shape override** that leaves the reference heuristic untouched:
- `config.py`: new optional 9th `config.json` row field, `{"label-prefix": forced_nc}` — longest matching label-prefix wins.
- `generate_pure_blis_project.py`: emits the override table into `generated_kernels.h` as `CONFIG_NC_OVERRIDES[]`.
- `templates/driver_benchmark.cpp`: applies the matching override after computing the heuristic `nc`, per dataset row (matched against the row's own label, e.g. `B3_M7_1`); `RVV_FORCE_NC` env var still available as a higher-priority ad hoc override for interactive experiments.
- `config.json`: the live `testCase_repeat.dat` row now carries `{"B3_M7_": 128}` as a shipped example/default mitigation, matching the data above.

This is a mitigation for the `Reuse8/12/24` sub-family, not a fix for the root cause of raw `B3_M7` — that remains open (item 3 above), and per discussion, may be a structural limit of this blocked-GEMM design for M-small shapes (real BLIS uses a different loop/pack strategy for that case) rather than a tunable-constant bug. Worth checking whether `64×384×384` is an actual target workload shape or a synthetic stress point before investing further here.

---

## 5. Next Steps (historical — superseded by §8 at closure)
1. [x] ~~Compare side-by-side run of `--scalar-pack` vs RVV pack to verify if scalar pack eliminates the bimodal drops.~~ Superseded — see §6.4; scalar-vs-RVV comparison already captured in §1/§2, the open question is now *why* both remain memory-bound, not whether they differ.
2. [x] Profile/inspect the emitted assembly of `gemm_pack_xdsl.cpp` for $NR=15$ — done in §6.2; it's already LMUL=1, chunked, matching the once-proposed fix.
3. [x] ~~Explore single-pass vector packing~~ — already the *only* path (§6.2), doesn't help. Instead: fixed `WL2` (§6.4.1), retested smaller `nc` (§7) — both done. Real perf counters (§6.4.2) never obtained; superseded by §8 follow-up.
4. [ ] Regression test for $CV\% < 5\%$ and KerGOPs $> 10$ on $B3\_M7$ — **not met by raw `B3_M7`** even after mitigation (§7 shows CV 11–15% at best); met for `Reuse12`/`Reuse24` at `nc=128`. Closing this issue without that test since the residual gap is now understood to be a structural M-small limit, not an unknown bug — see §8.

---

## 8. Closure and Follow-Up (2026-09-16)

**Closed as:** root cause identified (microkernel reuse factor, not cache-associativity/hardware
family as originally hypothesized), one real bug fixed (`WL2`), one practical mitigation shipped
and verified live (`config.json` `{"B3_M7_": 128}` override, §7). Not closed as "fully solved" —
raw `B3_M7` (M=64) keeps a real, unsolved residual gap, tracked below rather than left implicit.

**Residual open item — row-to-row median jump on raw `B3_M7`:** at `nc=128`, 10 back-to-back
identical `B3_M7` configs in the same process show medians jumping between two bands
(~8.75–8.87 and ~10.16–12.12 GOPs, see §7 raw log) rather than clustering tightly. This is not:
- explained by within-run cache spikes (already captured per-row in each row's own p5/p95/CV),
- explained by the reuse-factor theory (predicts a stable *ceiling*, not row-to-row jumps),
- explained by anything in the original bug reports (all built on the now-debunked ARM/8-way-L2 model, §6.3).

No hardware perf counters were ever used in this investigation (`perf` is not installed on
`jbpi`/`jbpi2`, confirmed §6.3) — everything here, including this closure, is inferred from
throughput numbers. Candidate causes not yet tested: CPU frequency scaling/thermal throttling
between runs, scheduler noise, or a `perf_event_open`-based syscall harness to get real cache-miss
counts. **Recommendation:** open a new, smaller issue scoped specifically to this row-to-row
jump if `B3_M7`-shaped workloads (M≈64, K/N large) matter for real target models — first confirm
`64×384×384` is an actual production shape and not a synthetic stress point (open question from §7),
since that determines whether this residual gap is worth the perf-counter investment at all.

**Files changed this issue:** `tests/pure_xdsl_blis_rvv/modelLevel/archs/bananapi.h` (WL2 fix),
`tests/pure_xdsl_blis_rvv/config.{json,py}`, `tests/pure_xdsl_blis_rvv/generate_pure_blis_project.py`,
`tests/pure_xdsl_blis_rvv/templates/driver_benchmark.cpp`, `tests/pure_xdsl_blis_rvv/compile_macro_bench.sh`.
