# Issue #7 (local): The per-tile timer tax is 11–17%, not 1–3%

**Role / Label:** `done` — implemented and measured on hardware
**Status:** 🟢 Closed by the issue #14 implementation. This note records the A/B measurement,
because the number it produced contradicts the estimate that both GitHub #11 (finding 3) and
`ref_baseline/README.md` carried, and every pack-inclusive figure we have published is
affected.
**Target:** `jbpi` (BananaPi M2 Pro, VLEN=256, `perf_event_paranoid=2`), reached via `fej`.
**Primary Sources:**
- GitHub #14 (`[Track 1] Finding 3`), child of #11 — the issue implemented here.
- [[0006-match-reference-design-everything-but-the-two-intended-differences]] — where
  finding 3 was first written down, with the 125–160 ns/tile estimate.
- [[0004-reference-comparison-metric-mismatch]] — the earlier `e2e` fix this builds on.
- `tests/pure_xdsl_blis_rvv/templates/perf_counters.h` (the `RVV_INSTRUMENTED` switch),
  `templates/Makefile`, `templates/driver_benchmark.cpp`, `generate_pure_blis_project.py`.

---

## 1. What the estimate was, and why it was low

Finding 3 derived the timer cost from the run logs themselves:

```
gap = RVV_Time_ms − (Vector_Pack_Time_ms + Compute_Time_ms)
```

and got 125–160 ns/tile, flat across shapes. It then assumed the pair's cost splits roughly
evenly — half inside `Compute_Time_ms`, half in that gap — and concluded the tax was ~2–3% on
large shapes, ~6% on B2.

That derivation can only ever see the part of the pair that falls *between* tiles. It cannot
see the part that falls inside the compute window, and it cannot see anything the timer costs
indirectly. It is self-referential: using the timers to measure the timers.

## 2. What the A/B actually shows

`make INSTRUMENTED=0` compiles the per-tile `chrono::now()` pairs and the `perf_event_open()`
reads out entirely. Both builds run the same passes (including the scalar-pack pass) and do
the same arithmetic — verified `PASS` against the scalar reference in both.

| case | shape | instrumented `RVV_Time_ms` | uninstrumented | tax |
|---|---|---|---|---|
| B1 | 1024×384×1024 | 118.824 / 119.453 / 118.131 | 98.634 / 101.427 | **14.6–17.0%** |
| B5 | 1024×384×4096 | 484.003 / 492.761 / 486.062 | 411.642 / 422.998 | **13.6–14.9%** |
| B2 | 384×384×64 | 1.686 (mean of 5) | 1.500 (mean of 5) | **11.0%** |

Two A/B sessions, the second re-run after the code-review fixes. The spread is board
noise; the sign and magnitude are stable across both.

Five runs each way on B2: instrumented 1.662–1.731, uninstrumented 1.487–1.516. The
uninstrumented build is both faster and *tighter*.

## 3. Where the 20 ms on B1 goes

A `chrono::now()` pair costs **~160 ns** on this board — measured directly, 200k iterations,
two runs agreeing (158.6 and 167.7 ns/pair). That is ~80 ns/call, which matches finding 3's
60–75 ns/call guess. The clock is in the vDSO; it is not a syscall.

B1 at mc=768 nc=384 kc=128 mr=32 nr=6 runs 24576 tiles/rep. Against the 17.4–20.2 ms
removed:

| component | per rep | share |
|---|---|---|
| 24576 clock pairs × 160 ns | 3.93 ms | 19–23% |
| 80 `read()` syscalls (40 perf regions × 2) | ≲0.8 ms | ≲5% |
| everything else | 12.7–15.5 ms | ~73–79% |

That remaining ~630 ns/tile is the cost of *having an opaque external call in the innermost
loop*: the call clobbers caller-saved registers, forces spills around the microkernel
invocation, and blocks the scheduling the loop would otherwise get. `.text` for
`gemm_dispatch_xdsl.o` drops from 3586 to 1894 bytes between the two builds, and the
uninstrumented object has no `read` and no `std::chrono::...::now` symbol at all.

This is exactly why finding 3's second bullet insisted on compiling the timers out rather than
estimating them: the dominant term is invisible to any estimate derived from the timers.

## 4. B3 is excluded, deliberately

B3 (64×384×384, the `nc: 128` override shape) does not settle:

| run set | instrumented | uninstrumented |
|---|---|---|
| 5× at N_REPS=30 | 2.529–2.644 (CV ~1.8%) | 2.860–3.942 (CV ~12%) |
| 1× at N_REPS=3, after a reference GEMM | 2.939 | 2.066 |

The uninstrumented build is sometimes much slower and sometimes much faster, depending on
warm-up and rep count, while the instrumented one is steady. This is the same shape as
[[0001-b3-m7-packing-instability]]. The plausible reading is that the per-tile clock call was
incidentally pacing B3's memory behaviour, and removing it lets the underlying instability
show — but that is a hypothesis, not a measurement, and B3 should not be quoted either way
until it is chased down separately.

## 5. A false start worth not repeating

The first implementation skipped the scalar-pack pass in the uninstrumented build, on the
reasoning that a build with no pack timer can measure nothing from it. That made B3 come out
**14% slower** uninstrumented. The scalar-pack pass is the RVV loop's de facto warm-up — it
first-touches `Ac`/`Bc`/`C` and trains the predictors — and without it one cold rep dominated
the mean on short shapes. The pass now runs in both builds and only its *reporting* is
conditional. Anything that differs between the two builds besides the timers destroys the A/B.

## 6. Consequences

1. Every pack-inclusive number we have published from an instrumented build is deflated by
   ~11–17%, not ~1–3%. `ref_baseline/README.md`'s caveat #2 has been corrected.
2. `compare_ref_baseline.py`'s `e2e` is now `2mnk / RVV_Time_ms`. On B1 against the
   `refbase_2026-09-18_164740` reference of 7.02 GFLOPS: **6.68, −4.8%** (was 6.86, −2.3%).
3. Those two corrections point opposite ways. `e2e` moving to the wall clock makes us look
   ~2.5% worse; removing the timer tax makes us ~15% faster. An uninstrumented B1 at
   98.6–101.4 ms is 7.9–8.2 GFLOPS pack-inclusive, i.e. **+13 to +16% over the reference's
   7.02**, not −4.8%. The honest headline needs the uninstrumented build on both halves of
   that comparison — and note the reference harness has its own (much smaller) timing
   overhead that this does not account for, so the true figure is somewhat below that.

## 7. Next steps

1. Re-run the full `refbase` comparison with `--no-instr` and `--raw-name
   all_raw_fp32_AXPY_FP32_Pack_noinstr.txt` to get a clean cross-design table. Not done here:
   it needs a full board sweep, not the two-shape A/B above.
2. Decide whether `KerGOPs` should move to the wall clock in the instrumented build too. It
   currently uses `kernel_time + pack_time`, which is why instrumented and uninstrumented
   `KerGOPs` are not directly comparable even though `RVV_Time_ms` is.
3. Chase B3 (§4) as part of [[0001-b3-m7-packing-instability]].
