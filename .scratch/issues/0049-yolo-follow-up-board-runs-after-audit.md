# YOLO follow-up: three board runs left open by the audit (#79)

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/82
**Role / Label:** `ready-for-human`. Each run needs the user's go-ahead.
**Status:** not started. The user will do it later (2026-11-17).
**Branch:** `yolo-follow-up-board-runs-after-audit`, cut from `pure-xdsl`.
**Parent:** #58. **Comes from:** #79 §6.3 ([brief PDF](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issue-79/issue-79-audit-brief.pdf), reports in `.scratch/reports/issue-79/`).

All runs: `jbpi`, 8 threads, idle board (check other users first), nothing else of ours running, uninstrumented, reference protocol. Each run is its own yes/no.

---

## 1. Which question did Héctor mean? (the one that matters)

- **Why:** #63 found the best tiles for *our* GEMM. Héctor most likely asked which tiles make *his* GEMM faster. Never measured (#63 Q7 skipped; #79 A1, A2).
- **Run (part D):** rebuild his `gemm_B3A2C0` (GEMMLib_yolo `b292c92`) to pick, by n, among kernels his library already has (32x8, 16x16). Local patch in a scratch copy, no new kernel code, his repo untouched. Time against his stock build, A-B-B-A (stock, patched, patched, stock).
- **Decide first:** if he meant ours, skip; #63 answers it with the #79 caveats. Only the user or Héctor can say which.

## 2. Why his Y00 swings (6.7 to 105.5 ms over 9 launches)

- **Why:** the "ours faster" headline rests on this one shape of his (#79 B5, B6).
- **Run:** one short `gemm.x` launch with Y00 moved to a later row.
- **Tells apart:** his 300 barriers per call vs. Y00 being the first call of each process.
- **Worth it if:** ours-vs-his gets quoted to Héctor. Otherwise keep quoting the numbers without Y00.

## 3. Our slow reps at 8 threads

- **Why:** 2-10% of reps cost an extra 8-13 ms. Likely cause: core 0, which takes every device interrupt (13.1% sys time vs. 0.2-0.5% on cores 1-7; #79 B6).
- **Run:** 8 threads on cores 0-7 against 7 threads on cores 1-7.
- **Worth it if:** cleaner 8-thread numbers are wanted in general. It doesn't change the tile answer for Héctor.

---

## Order

Settle 1 first: it decides whether #63 answered Héctor at all. 2 and 3 are optional cleanup.

## Closure

- [ ] 1 run or skipped, with reason
- [ ] 2 run or skipped, with reason
- [ ] 3 run or skipped, with reason
- [ ] Results copied to `HectorYolo/issue-<N>/` with README + report; logs committed
