# 0001: The Reference Design patchers stay separate

**Status:** accepted 2026-11-20 (user, on issue #120).

## Context

`tests/pure_xdsl_blis_rvv/ref_baseline/` holds three modules that change a staged copy of the Reference Design: `ref_pack_fix.py` (`gemm_blis.c`, issue #54), `ref_per_rep.py` (`test_gemm.c` and the rep logs, issues #54/#55) and `repair_ref_kernels.py` (six truncated kernel files). An architecture review (#115) proposed merging them into one patch step. `fringe.py` sits in the same folder but patches nothing; it computes our fringe-tile overhead.

## Decision

Keep them separate. Each owns one file, its anchors and its replacement text.

## Why

- They already fail loudly: `ref_pack_fix.py` and `ref_per_rep.py` raise `PatchError` (exit 1) unless each anchor occurs exactly once; `repair_ref_kernels.py` prints `[SKIP]` and does not guess.
- A merged module would have the union of three unrelated anchor sets: a bigger interface, no behaviour moved (deletion test: nothing concentrates).
- `compare_ref_sweep.py` already imports `load_reference` from `compare_ref_baseline.py`, so the result CSVs have one parser.

## Consequences

A future architecture review should not propose merging these patchers again unless one of them starts to fail silently. Closed as `wontfix`: issue #120.
