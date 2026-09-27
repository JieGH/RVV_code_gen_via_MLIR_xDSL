# Issue #39: Benchmark house rules: no core 0, several launches for small gaps, keep every quoted log

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/39

**Role / Label:** `needs-triage` — from the #31-#34 audit report ([issues-31-34-brief.pdf](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-31-34-brief.pdf)), next step 5.
**Status:** ⏸️ Issue only. **Implementation on hold** (documentation change only, when approved).

## Problem
The #31-#34 audit found conclusions resting on setups the project already knew were weak:
- #32 ran all 32 launches on core 0, the same day #31 found core 0 runs 5-10% slower.
- #31 and #33 quote numbers from runs whose logs were never saved (kernel swaps, reference-side runs).
- #33 reads differences of 0.3-1.6% from single launches, while the same binary moves up to 1.7% between launches in the same hour.

## Proposed rules (not written yet)
1. Never pin to core 0; say which core in every result.
2. One launch is not a result: when a difference is under ~3%, compare inside one launch and repeat over several.
3. Every quoted number must come from a saved log under `tests/pure_xdsl_blis_rvv/logs/`.
4. Reconsider the stored default run command (`--cpu 0`).

Where: `docs/agents/execution.md` and `docs/metrics.md`.
