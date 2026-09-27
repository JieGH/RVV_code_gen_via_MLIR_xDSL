# Issue #35: B3 nc override: keep {"B3": 128} in config.json or drop it?

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/35

**Role / Label:** `needs-triage` — from the #31-#34 audit report ([issues-31-34-brief.pdf](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-31-34-brief.pdf)), next step 1.
**Status:** ⏸️ Issue only. **Implementation on hold** until the user decides.

## Question
Should `config.json` keep B3's shipped `nc` override (`{"B3": 128}`), or drop it as the uncommitted change from #32 does?

## What is known
- The #32 NR sweep removed the override in the working tree (`tests/pure_xdsl_blis_rvv/config.json`, uncommitted). With the change, 5 tests fail (they expect the override).
- Issue #1 added the override because B3 was unstable at `nc`=384.
- #34 §7.2 (one scalar-pack launch): B3 is fast and steady when N ≤ 320 and slow when N = 384, even with a small B. With no override, N = 384 means `nc` = 384. This points toward keeping the override.
- All #32 numbers were taken without the override (`--no-nc-overrides`), so they do not show how the shipped setting behaves.

## Proposed work (not started)
1. One launch on a non-zero core with B3 at `nc` = 128, 192, 256, 320 and 384, for both packers, all in one launch.
2. Decide: restore the override (and leave the tests as they are), or drop it and update the tests.

## Out of scope
Why the vector packer is slow on B3 (separate issue).
