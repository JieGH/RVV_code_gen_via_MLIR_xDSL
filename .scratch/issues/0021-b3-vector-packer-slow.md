# Issue #36: Why is the vector packer slow on B3 (64x384x384)?

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/36

**Role / Label:** `needs-triage` — from the #31-#34 audit report ([issues-31-34-brief.pdf](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-31-34-brief.pdf)), next step 2.
**Performance results per test:** Appendix A of the audit report, rows 6, 7 and 22 (B3 and B2 packer results; B3 L2 test).
**Status:** 🔍 Measured 2026-09-27 (Results below). Cause narrowed to a memory-side slow state, not the packer's own speed; root cause still open.

## Question
Why is the vector packer slow on B3 (64×384×384), when it is equal to or faster than the scalar packer on every other layer?

## What is known
- Scalar pack beats vector pack on B3 by 21-78% (by mean) in every #26 and #32 launch, on both boards and at every NR (16x8-16x16). On B2, B4, the square shapes and 2048³, vector is equal or up to 6% ahead.
- B3 has two speeds inside one run (~4.4 and ~9.9 GOPs). The vector packer's best rep is as fast as the scalar one's, but most of its reps are slow.
- #34 §7.2: making B fit in L2 does not help the vector packer. With the scalar packer, the slow case is N = `nc` = 384.
- #26 §5: pack time alone predicts only ~5% of the gap, so the packer probably changes something the kernel sees afterwards (cache state is the leading guess, untested).

## Proposed work (run 2026-09-27, see Results)
1. Instrumented (`INSTRUMENTED=1`) pack/compute split for B3 at 16x14, 16x15 and 16x16 (plan §4.4 of #32, never run), and for B3_N320 against B3_K256 with each packer.
2. Save per-rep times, to see whether slow reps come in runs or at random.
3. Read the vector `pack_CB` code and its memory access order for B.

## Out of scope
Changing the packer. The `nc` override decision is a separate issue.

---

## Results (2026-09-27)

Board `jbpi`, **core 4**. Two kinds of run, never mixed in one table:
- **Uninstrumented** (`INSTRUMENTED=0`, reference protocol, `--tmin 10`): the #35 `nc` sweep, 3 launches per
  packer, every rep saved with `--save-reps`. Logs and full tables: #35 §6.
- **Instrumented** (`--instr`, warm-up protocol, 1000 fixed reps, **debug numbers only**):
  `config_issue36_instr_split.json`, one launch per packer, `--save-reps` (each rep's pack and compute time
  and cycles). Vector: `logs/run_2026-09-27_114341`; scalar: `logs/run_2026-09-27_115023`.

### 1. Step 2: the slow reps come in long stretches, and each case starts fast

Every B3 case begins in the fast speed and falls into the slow one after **0.1-0.5 s** (vector packer) or
**0.5-8 s** (scalar), then mostly stays there. Slow reps are 10-100 times more bunched together than chance
(clumping 0.01-0.1, see `reps_report.py`). First 80 reps of B3 at 16x14 (instrumented, vector; F fast, s slow):

```
FFFFFFFFFFFFFFFFFFFFFsFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFsssssssssssss
```

So the slow mode is a **state** that sets in and lasts, not random per-rep noise.

### 2. Step 1: in a slow rep both halves slow down, and the core is waiting on memory

Median per rep, fast reps against slow reps of the same case (instrumented; `nc` = 384 unless named):

| tile | case | packer | fast reps: pack / compute (ms) | slow reps: pack / compute (ms) | slow reps |
|---|---|---|---|---|---|
| 16x14 | B3 | vector | 0.64 / 1.78 | 1.58 / 2.79 | 412 / 1000 |
| 16x15 | B3 | vector | 0.75 / 1.88 | 1.46 / 2.59 | 842 / 1000 |
| 16x15 | B3, `nc`=128 | vector | 0.74 / 1.58 | 1.53 / 2.12 | 533 / 1000 |
| 16x16 | B3 | vector | 0.55 / 3.20 | 1.68 / 4.40 | 569 / 1000 |
| 16x15 | B3_N320 | vector | 0.63 / 1.47 | 1.29 / 2.05 | 545 / 1000 |
| 16x15 | B3_K256 | vector | 0.33 / 1.02 | 0.83 / 1.69 | 967 / 1000 |
| 16x14 | B3 | scalar | 0.76 / 1.71 | 1.24 / 1.98 | 721 / 1000 |
| 16x15 | B3 | scalar | 0.69 / 1.52 | 1.11 / 1.85 | 103 / 1000 |
| 16x15 | B3, `nc`=128 | scalar | 0.97 / 1.56 | — | 0 / 1000 |
| 16x16 | B3 | scalar | 0.71 / 3.24 | — | 0 / 1000 |
| 16x15 | B3_N320 | scalar | 0.63 / 1.26 | 1.04 / 1.51 | 374 / 1000 |
| 16x15 | B3_K256 | scalar | 0.47 / 1.00 | 1.19 / 1.57 | 981 / 1000 |
| 16x14, 16x16 | B2 | both | one speed | — | 0 / 1000 |
| 16x15 | Reuse24, `nc` 128 and 384 | both | one speed | — | 0 / 1000 |

- In the slow reps the **pack takes 1.6-3.1 times as long and the compute 16-66% longer**, with either
  packer. With the vector packer the kernel loses about as many ms as the packer does (e.g. 16x15: +0.71 ms
  each); with the scalar one somewhat fewer. So this is not "the vector packer is slow".
- Cycles rise with the time: the effective clock (cycles / seconds) is 1.63-1.81 GHz, with no drop in the
  slow reps. The core is not throttled; it **stalls on memory** in the slow state.
- In a fast rep the vector packer is as fast as or faster than the scalar one (e.g. 16x16: 0.55 vs 0.71 ms).
  #26's finding (pack time explains ~5% of the gap) fits: the gap is how often each run is in the slow state.
- The vector packer enters the slow state sooner and more often; with it the slow state also costs more.
- Tiles (step 1's tile question): with the vector packer all three tiles (16x14/15/16) are two-speed; with the
  scalar packer 16x14 and 16x15 are, and 16x16 did not go slow in this launch. The slow state is not tied to
  the 16x16 tile's spills.

### 3. Step 3: the vector `pack_CB` walks memory exactly like the scalar one

Generated `pack_CB_v_if_AXPY_FP32_Pack_fp32_NR15` (`pj_gen/.../gemm_pack_xdsl.cpp`, built by
`pack_builder.py:build_pack_CB_v_if`) against `pack_CB_scalar` (`gemm_dispatch_xdsl.cpp`): for each 15-column
panel, for each k, read the 15 elements at that k (column stride `ldb`·4 = 1536 B for B3), and write them as
15 contiguous floats of `Bc`. **Same source addresses, same order, same `Bc` layout.** The only differences:
the vector one reads each row with two strided loads (`vlse32`, 8 + 7 elements) and writes with two vector
stores, where the scalar one does 15 scalar loads and stores. So the two cannot differ in *which* memory they
touch, only in how fast they issue the requests (the vector one has more of them in flight at once).

### 4. What is ruled out, and what is left

Ruled out on `jbpi`:
- **CPU clock:** governor `performance` on all cores (provenance), and the per-rep clock is the same in both speeds.
- **Transparent huge pages** re-placing memory mid-run: THP is `madvise` only and `AnonHugePages` is 0 kB.
- **B not fitting in the L2:** B3_K256 (B = 384 KiB) is the slowest case of all.
- **A different memory walk in the vector packer:** §3.

Left (leading guess, **not tested**): a state of the shared L2 (512 KiB, 16-way, shared by cores 4-7) or of the
hardware prefetcher that, once entered, makes this access pattern miss much more. It fits: the state lasts,
it is entered sooner when requests come faster (vector packer), a smaller packed block (`Bc` within L1) avoids
it, and more reuse of each packed block (Reuse24, B2) avoids it.

### 5. Next steps (not done)

1. Per-rep **L2 miss and prefetch counters** (X60 raw PMU events) in fast against slow reps. The perf log
   today has no L2 events and only whole-case averages.
2. Check whether other cores of the 4-7 cluster are active when the slow state starts (the L2 is shared);
   try the same run on core 0-3's cluster (not core 0).
3. Test whether `ldb` being a power of two (B3_K256) matters: pad `ldb` (ties to #38).
4. Repeat on `jbpi2`.
