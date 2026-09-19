# Issue #5: Swap RVV Vector Pack ↔ Scalar Pack in the Live Dispatch Path to Isolate the Perf Bottleneck

**Role / Label:** `ready-for-agent` — pending, not started
**Status:** 🟡 Pending. Carried over from issue #2's still-open next-step #1 ("investigate the
Square-shape RVV-pack-vs-scalar-pack regression"). Not the same experiment as issue #3/#7,
which only *timed* both pack implementations side by side without changing which one the
benchmark actually runs — see §1 for why that distinction matters.
**Target:** `jbpi` / `jbpi2` (BananaPi M2 Pro, VLEN=256), reached via jump host `fej`.
**Primary Sources:**
- [[0002-fixed-dim-pack-compute-breakdown]] — origin of the open question (finding #2,
  next-step #1).
- [[0003-scalar-vs-vector-pack-timing]] — measured scalar-pack vs. RVV-pack timing
  head-to-head, but both ran unmodified in their normal role (vector pack is what
  `gemm_dispatch_xdsl.cpp` actually calls; scalar pack was timed as a separate,
  parallel pass, never substituted into the real compute path).
- [[0004-reference-comparison-metric-mismatch]] — found the Square "gap" issue #2 originally
  flagged was a metric/reference artifact, not a packing-time effect. Still open there: B3's
  genuine 40.6% pack fraction (§5 of that issue), which is the shape this issue should focus on.
- `tests/pure_xdsl_blis_rvv/gemm_dispatch_xdsl.cpp` (dispatch tables — only `pack_*_v_if`
  wired in today, per issue #3/#7 §1 finding).
- `tests/pure_xdsl_blis_rvv/pack_builder.py` (RVV pack codegen), `generate_pure_blis_project.py`
  (`render_scalar_pack_functions()`, scalar pack codegen).

---

## 1. Why this is a different experiment from issue #3/#7

Issue #3/#7 proved both pack implementations are *correct* and timed them *independently* —
each ran once per benchmark row, purely for measurement, alongside whichever pack function the
macro-kernel actually used for real (always `pack_*_v_if`, the RVV path — confirmed in that
issue's §1, dead-but-correct `_v_else` aside). That answered "how long does each pack take in
isolation" but not "does the choice of pack implementation change end-to-end kernel
performance" — because the scalar-pack numbers in that issue never fed into the actual
microkernel compute step; they were a shadow measurement.

This issue is the actual swap: build and run the benchmark with **scalar packing wired into
the live dispatch path** (replacing `pack_*_v_if`) for at least one case, and compare
`KerGOPs`/`Compute_Time_ms` end-to-end against the normal RVV-pack build, same shape, same
board, same rep count. If end-to-end performance is statistically the same regardless of which
pack implementation is live, that's a stronger, more direct confirmation that packing isn't
the bottleneck (extending issue #4's finding beyond the Square-vs-reference metric artifact to
a real head-to-head on this repo's own pipeline). If it differs by more than pack's own
time-share would predict, that points at a second-order effect (cache warmth handed off from
packing into compute, instruction-cache effects, etc.) that issue #3/#7's isolated timing can't
see.

## 2. Candidate approach

1. Add a build-time or config-driven switch (e.g. `config.json` field, mirroring the
   `nc`-override pattern from issue #1) that lets `gemm_dispatch_xdsl.cpp` call
   `pack_*_scalar` instead of `pack_*_v_if` for a given row, without touching the microkernel
   compute call itself.
2. Pick the shape(s) most worth resolving first: **B3** (`64×384×384`, `MR16_NR15`) is the
   priority target — it's the one shape where issue #3/#7 found vector packing *losing* to
   scalar (finding #2 there) and the one shape issue #4 flagged as the real, metric-independent
   performance problem (40.6% pack fraction, §5 of [[0004-reference-comparison-metric-mismatch]]).
   Re-run one or two Square shapes (e.g. `S1`) too, since that's where issue #2's original
   question came from.
3. Run both builds (RVV-pack-live vs. scalar-pack-live) back-to-back on the same board session,
   same `N_REPS`, and compare `KerGOPs` end-to-end — not just the pack-phase ms columns already
   captured in issue #3/#7.
4. If B3 flips to genuinely faster with scalar packing live (not just in isolated timing), that
   is the concrete case for issue #3/#7's next-step #1 ("amortize/skip the vector pack path
   below some block-size threshold") — this issue would supply the missing end-to-end evidence
   for that follow-up.

## 3. Open questions this should answer

- Does swapping which pack implementation actually drives the compute path change end-to-end
  `KerGOPs`, or does it wash out exactly as issue #3/#7's isolated pack-ms timing predicts?
- On B3 specifically: does a scalar-pack-live build outperform the current RVV-pack-live build
  end-to-end, matching the sign of issue #3/#7's -12.7% isolated-timing finding?
- Does this change anything about issue #4's open items (§10-11 there), e.g. the B3
  pack-fraction follow-up it recommended splitting into its own issue?

## 4. Not yet done

No code, config, or hardware run for this issue has started. This file exists to record the
task per issue #2's carried-over next-step #1 before work begins.
