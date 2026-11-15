# Issue #56: Make the design parallel for the GEMM test, and test it on one specific program

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/56

**Role / Label:** `needs-triage` — opened 2026-11-12 at the user's request. A general new direction, not a spec yet.
**Status:** 📝 Open. Nothing built or measured.
**Target:** to be decided (board, core count and which program).

---

## 1. Direction

Make the design run in parallel (several cores) for the GEMM test, and test it on one specific program.

## 2. Open points (to settle before a spec)

- Which specific program to test on (to be named by the user).
- How the work is split across cores, and how that fits the house rules of #39 (never core 0, saved logs, idle board).
