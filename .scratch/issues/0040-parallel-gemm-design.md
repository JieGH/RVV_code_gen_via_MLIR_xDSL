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

---

## 3. Update 2026-11-15: the program is YOLO26 (Héctor)

- **Program:** the YOLO26 GEMM shapes on the K1 (Héctor, email of 13 Nov 2026). This issue is now **S5** of the YOLO parent #58 [[0041-yolo-best-microkernel-size-parent]].
- **Starting point for the split:** Héctor's `gemm_B3A2C0` (OpenMP, loop 5 `ir` split, B packed in slices by every thread, A packed per thread, two barriers). Read in S4 [[0045-read-hector-parallel-gemm]]. S4's copy / change / skip list lands here before a spec is written.
- **Built in our design.** The generated macrokernel gets the thread split. Héctor's code is only read (S4) and compared against (S6), never copied in.
- **Rule: opt-in, nothing that runs today changes.** A new `--threads N` (default 1) on `compile_macro_bench.sh` and the generator. With 1 thread the generated code is byte-identical to today (test), so every result since #39 stays valid. The search from S2 #60 runs with any thread count.
- **Still open:** the core-0 rule for 8 threads (S6 [[0046-yolo-eight-thread-sweep]] §2), and whether the cache model needs a per-thread `mc`.
- **Blocked by:** #62 (S4).
