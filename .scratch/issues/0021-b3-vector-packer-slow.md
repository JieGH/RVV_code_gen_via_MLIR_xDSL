# Issue #36: Why is the vector packer slow on B3 (64x384x384)?

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/36

**Role / Label:** `needs-triage` — from the #31-#34 audit report ([issues-31-34-brief.pdf](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-31-34-brief.pdf)), next step 2.
**Status:** ⏸️ Issue only. **Implementation on hold.**

## Question
Why is the vector packer slow on B3 (64×384×384), when it is equal to or faster than the scalar packer on every other layer?

## What is known
- Scalar pack beats vector pack on B3 by 21-78% (by mean) in every #26 and #32 launch, on both boards and at every NR (16x8-16x16). On B2, B4, the square shapes and 2048³, vector is equal or up to 6% ahead.
- B3 has two speeds inside one run (~4.4 and ~9.9 GOPs). The vector packer's best rep is as fast as the scalar one's, but most of its reps are slow.
- #34 §7.2: making B fit in L2 does not help the vector packer. With the scalar packer, the slow case is N = `nc` = 384.
- #26 §5: pack time alone predicts only ~5% of the gap, so the packer probably changes something the kernel sees afterwards (cache state is the leading guess, untested).

## Proposed work (not started)
1. Instrumented (`INSTRUMENTED=1`) pack/compute split for B3 at 16x14, 16x15 and 16x16 (plan §4.4 of #32, never run), and for B3_N320 against B3_K256 with each packer.
2. Save per-rep times, to see whether slow reps come in runs or at random.
3. Read the vector `pack_CB` code and its memory access order for B.

## Out of scope
Changing the packer. The `nc` override decision is a separate issue.
