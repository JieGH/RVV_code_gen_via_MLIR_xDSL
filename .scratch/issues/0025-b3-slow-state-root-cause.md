# Issue #41: Why does B3 fall into a slow state? (L2 counters per rep, then the other suspects)

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/41

**Role / Label:** `needs-triage` — follow-up of #36 (its "Next steps"), and step 2 of the #35 recommendation.
**Status:** 🟡 Open. One check done 2026-09-27 (§3: the generic L2 counters do not work on the X60).
**Brief:** [issues-35-36-39-brief.pdf](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-35-36-39-brief.pdf), §2 and §4.
**Target:** `jbpi` (then `jbpi2`), core 4 unless a step says otherwise. House rules of #39 apply.
**Shape(s):** B3 (64×384×384), B3 with K=256; controls B2 and Reuse24.
**Primary Sources:**
- [[0021-b3-vector-packer-slow]] (#36) Results — the slow state, fast/slow pack and compute split.
- [[0020-b3-smaller-nc-exploration]] (#35) §6 — the `nc` sweep and per-rep runs.
- `tests/pure_xdsl_blis_rvv/templates/perf_counters.h` — the counters we open today.
- `tests/pure_xdsl_blis_rvv/logs/run_2026-09-27_114341`, `run_2026-09-27_115023` — instrumented per-rep logs.
- `.scratch/issues/0025-pmu-probe.c` / `.txt` — §3's probe and its output.

---

## 1. Question

B3 runs at two speeds. Every run starts fast and, after 0.1-0.5 s (vector packer) or 0.5-8 s (scalar packer),
falls into a state about half as fast, where it mostly stays (#36). In that state both the packing (1.6-3.1×)
and the microkernel (+16-66%) slow down at an unchanged clock, so the core is waiting on memory. **What is
that state, and can we keep B3 out of it?**

## 2. What is already ruled out (#36)

CPU clock (governor `performance`, per-rep clock unchanged); transparent huge pages (`madvise` only, 0 kB in
use); B not fitting in the L2 (B3 with K=256 is the slowest case); a different memory walk in the vector packer
(same addresses, same order as the scalar one). Leading guess: a state of the shared L2 (512 KiB, 16-way,
cores 4-7) or of the hardware prefetcher.

## 3. Done so far (2026-09-27): the generic L2 counters read 0

A probe on `jbpi`, core 4 (`0025-pmu-probe.c`, output `0025-pmu-probe.txt`) opened every generic cache event
perf offers and streamed 64 KiB and 4 MiB 20 times. All events open, but **only `cycles` and `L1D-read-miss`
count** (1.00 miss per 64 B line on the 4 MiB stream, as expected). `cache-references`, `cache-misses`, every
`LL-*` (last-level = L2) event, every prefetch event and `bus-cycles` read **0**. The kernel exposes no named
events (`/sys/bus/event_source/devices/cpu/events` does not exist). So the L2 cannot be counted through the
generic events; it needs the X60's **raw event codes** (`PERF_TYPE_RAW`), if the SBI firmware lets them through.

## 4. Plan

Steps 1-3 are specified in #42 ([[0026-l2-counters-per-rep-spec]], `ready-for-agent`).

1. **Find the X60 raw event codes** for L2 access / miss / refill / writeback and prefetch (SpacemiT K1 / X60
   manuals, the vendor kernel or OpenSBI PMU tables, the device tree's `riscv,event-to-mhpmcounters`). Check
   with the probe that each counts sensibly on the 64 KiB / 4 MiB streams.
2. **Per-rep L2 counts in the driver** (instrumented build): add the working raw events to
   `perf_counters.h` and write them per rep into `reps_*.txt` next to `Pack_Cycles` / `Compute_Cycles` (TDD on
   the host driver seam, as for `--save-reps`). Then compare fast against slow reps for B3 (both packers), B3
   with K=256, and B2 / Reuse24 as controls.
3. **Is it the shared L2?** Run B3 on core 2 (the 0-3 cluster, not core 0) against core 4, inside the house
   rules; and log what else runs on cores 5-7 (e.g. `/proc/stat` per core before and after each case).
4. **`ldb` a power of two?** Pad `ldb` (and `lda`/`ldc`) off 256 / 384 for B3 with K=256 (ties to #38).
5. **Repeat the key runs on `jbpi2`.**

## 5. How to read it

- If slow reps show many more L2 misses (or prefetches) than fast reps for the same work, the state is in the
  L2 / prefetcher; then look for what switches it (step 3, 4) and for a way to avoid it (blocking, padding,
  prefetch hints, `zicbo*` cache-block ops the X60 has).
- If L2 misses are the same in both states, look past the L2 (DRAM, the TLB: dTLB misses are already counted
  per case in the perf log, not yet per rep).

## 6. Out of scope

Changing the packer or the cache model before the cause is known.

## 7. Closure

- [ ] L2 (and prefetch) counts per rep, fast against slow, for B3 and a control.
- [ ] Core-2 against core-4 result, inside one set of launches.
- [ ] `ldb` padding result.
- [ ] A named cause, or a written list of what it is not, with the logs committed.
