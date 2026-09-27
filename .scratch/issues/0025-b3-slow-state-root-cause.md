# Issue #41: Why does B3 fall into a slow state? (L2 counters per rep, then the other suspects)

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/41

**Role / Label:** `needs-triage` — follow-up of #36 (its "Next steps"), and step 2 of the #35 recommendation.
**Status:** 🟡 Open. 2026-09-27: §3 (generic L2 counters read 0), then plan steps 1-3 done through #42 (§8): the slow state is the L1D prefetcher doing about half its work, not the shared L2. Steps 4-5 open.
**Brief:** [issues-35-36-39-brief.pdf](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-35-36-39-brief.pdf), §2 and §4; results of §8: [issue-42-brief.pdf](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issue-42-brief.pdf).
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

- [x] L2 (and prefetch) counts per rep, fast against slow, for B3 and a control (§8.2).
- [x] Core-2 against core-4 result, inside one set of launches (§8.3).
- [ ] `ldb` padding result.
- [ ] A named cause, or a written list of what it is not, with the logs committed. **Partly:** named as far as
  "the L1D prefetcher does half its work" (§8.4); what makes it drop is still open.

## 8. Results (2026-09-27, through #42)

All from the **instrumented** build (warm-up protocol, 1000 fixed reps per case, `--no-nc-overrides`,
`--raw-events x60-l2`), `jbpi`, MR16×NR15, the model's `nc` (384). These are counts for the fast/slow
comparison, **not speed numbers to quote**. Nine launches, in the order V4 S4 V2 | V2 S4 V4 | V4 S4 V2
(V/S = vector/scalar packer, 4/2 = core), so each pairing runs A-B-B-A-A-B, 3 launches per setting:

- core 4, vector: `run_2026-09-27_131302`, `run_2026-09-27_132606`, `run_2026-09-27_132843`
- core 4, scalar: `run_2026-09-27_131537`, `run_2026-09-27_132328`, `run_2026-09-27_133117`
- core 2, vector: `run_2026-09-27_131815`, `run_2026-09-27_132052`, `run_2026-09-27_133354`
- check of the per-run busy share (not in the tables): `run_2026-09-27_133825`

Reports: `0025-reps-report-core4.md`, `0025-reps-report-core2.md` (`reps_report.py` over those folders).

### 8.1 The raw events (plan step 1)

The X60's L2 and prefetch events are raw codes 0xa0-0xbf in the vendor kernel's perf tables
(spacemit-com/linux-6.6, `tools/perf/pmu-events/arch/riscv/spacemit/x60/cache.json`); the board's device
tree lets any code 0x00-0xff through to counters 3-18. All of them count (`0026-raw-probe.c` / `.txt`,
and the on-board check in `test_perf_counters.py`); the table with every count is in `docs/metrics.md`,
"Raw events". One surprise: on a streaming read, demand L2 load misses stay near 0 even at 4 MiB,
because the prefetcher runs ahead; the L2 AR-channel stall cycles show the traffic instead.

### 8.2 Fast reps against slow reps (plan step 2), core 4, mean per rep

| case (packer) | slow reps | cycles pack / compute | L2 loads | L2 load misses pack / compute | L1D prefetch hits pack / compute | L1D misses compute |
|---|---|---|---|---|---|---|
| B3 (vector) | 51% | 2.09× / 1.51× | 1.01× / 0.95× | 2.52× / 3.13× | 0.40× / 0.51× | 3.06× |
| B3_K256 (vector) | 60% | 1.97× / 1.50× | 1.05× / 0.95× | 3.22× / 3.73× | 0.52× / 0.52× | — |
| B3_K256 (scalar) | 36% | 2.10× / 1.46× | 1.27× / 0.93× | 4.21× / 3.39× | 0.31× / 0.49× | — |
| B3 (scalar) | 0% (one speed in these 3 launches) | | | | | |
| B2 (both) | 0% | | | | | |
| Reuse24 (vector) | 7% | 1.44× / 1.11× | | 1.62× / 1.23× | 0.62× / 0.48× | |

(slow/fast ratio; the cell is empty where a case has one speed.) In every case with slow reps, the same
thing moves: **the same number of L2 accesses, 2-4× the demand L2 misses, and half the L1D prefetch hits
and refills.** The AR-channel stall cycles also halve (0.43× / 0.48× for B3 vector), i.e. fewer requests
go past the L2 while the core waits longer.

### 8.3 Core 2 against core 4 (plan step 3)

Core 2 sits in the other cluster (cores 0-3, their own L2). Vector B3 on core 2 has two speeds just the
same: 64% slow reps (core 4: 51%), GOPs by mean time 5.82 against 5.92, the same counter pattern (L2 load
misses 2.25× / 3.02×, prefetch hits 0.50× / 0.56×). So it is **not** something the cores 4-7 cluster
does, nor another process on cores 5-7. The busy share of the nine launches was read over the whole
launch, board build included (cores 1, 2, 6 at 15-58% are the build; fixed in #42 after review); the check
launch `run_2026-09-27_133825`, read around the timed run only, shows core 4 at 100% and every other core
at 0%.

### 8.4 What the slow state is, so far

The L1D hardware prefetcher, at the same work, prefetches about half as much in the slow state, and the
lines it no longer brings in become demand misses in the L1D and the L2. It is not the L2 filling up (same
access count; B3_K256 slows more though its B fits), not the other cluster's neighbour, not the clock (#36).
Open: **what makes the prefetcher back off** (it switches within 0.1 s for the vector packer and stays
switched, #36), and what can stand in for it. (Correction, 2026-09-27: the X60 has **no** prefetch hint
instructions: `zicbop` is not in its ISA string and the device tree has no `riscv,cbop-block-size`, so
`prefetch.r` would be a no-op. It does have `zicbom` and `zicboz`; `cbo.zero` runs in user mode, 64-byte lines.)

**Decided 2026-09-27 (user):** B3's `nc`=128 override is back in `config.json` as the baseline (#35's file has
the history). Two fixes are specified, implementation on hold:
- #43 ([[0027-cbo-zero-pack-buffers-spec]]): clear each Ac/Bc line with `cbo.zero` before packing into it, so
  the packer's writes stop reading lines from the L2.
- #44 ([[0028-pack-b-fewer-streams-spec]]): pack B one column at a time, so the prefetcher follows one stream
  instead of the 15 of today's panel-wide walk. Next: plan step 4 (`ldb` padding off 384, #38) and the same counts per 100-rep window around the
switch, then `jbpi2`.
