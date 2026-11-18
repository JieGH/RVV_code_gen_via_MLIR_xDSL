# Present the findings: Héctor's YOLO request, BERT/Square sweep, vector against scalar packing

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/91

**Role / Label:** `needs-triage`. Write-up from committed data; no new code.
**Status:** deliverables done 2026-11-18, waiting on the user (§3 decision, merge). Report [PDF](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/present-yolo-bert-square-and-packing-findings/.scratch/reports/issue-91/report/issue91_report.pdf) / [LaTeX](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/present-yolo-bert-square-and-packing-findings/.scratch/reports/issue-91/report/issue91_report.tex) (A4, 19 pages, added at the user's request), slides [PDF](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/present-yolo-bert-square-and-packing-findings/.scratch/reports/issue-91/deck/issue91_deck.pdf) / [LaTeX](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/present-yolo-bert-square-and-packing-findings/.scratch/reports/issue-91/deck/issue91_deck.tex), [README](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/present-yolo-bert-square-and-packing-findings/.scratch/reports/issue-91/README.md).
**Metric correction:** #83 and #85 quoted the average of per-rep GOPs as "by mean time". The deck and report recompute every packing number with 2mnk / mean time per call (`deck/make_data.py`). The answers hold; details move (YOLO 0.929 / 0.945, best tile changes 13 / 20, model within 0.02 on 24 / 20 of 46; B3 at 1T is a steady 1.227, not noise). The numbers in §2 below are the pre-correction ones. The #83/#85 reports themselves are not corrected yet.
**Branch:** `present-yolo-bert-square-and-packing-findings`, cut from `pure-xdsl`.
**Covers:** #58 (parent: #59-#63, #56), #79 audit, #82 (open runs), #71, #83, #85.
**Deliverable:** one deck (PDF, LaTeX beamer, pgfplots charts drawn from the committed CSVs), plus `.scratch/reports/issue-91/README.md`. Linked from every covered issue.

---

## 1. Purpose

One talk that answers three questions, every number taken from a committed log:

1. **Héctor:** which microkernel sizes run fastest for the YOLO26 GEMM, at 1 thread and at 8 threads?
2. **Paper shapes:** which tile wins on BERT B1-B5 / Square S1-S5 at 1 and 8 threads, and do the paper's picks hold?
3. **Packing:** does vector packing beat scalar packing, and which parameters (m, n, k, tile, threads) decide it?

## 2. Deck outline and sources

Every slide names its board, build, protocol, cores and launches. GOPs by mean time and by median rep side by side.

| # | slide | key numbers (already measured) | source |
|---|---|---|---|
| 0 | Setup + glossary | K1 / X60, VLEN 256; jbpi, jbpi2; uninstrumented, reference protocol; 1T core 4, 8T cores 0-7; launch, A-B-B-A, tile, GOPs by mean / median rep | #39, #83 and #85 glossaries |
| 1 | Héctor's request | 53 YOLO26 shapes (50 without n = 1); his 16x8 / 16x4; goal: inference under 0.5 s | #58 §1, email PDF in `.scratch/reports/issue-58/` |
| 2 | Parallel engine | ours 698.6 → 152.1 ms per pass (4.59x); his 583.3 → 135.7 ms (4.30x); per-n scaling | #56 README, #62 README |
| 3 | 1T best tile | 24x8 4.26 / 4.25 GOPs against 16x8 3.79 and 16x4 3.44; per n: 24x4, 24x9, 24x8, 32x7 | #61 (`0044` §5a) |
| 4 | 8T best tile | best per shape 134.1 ms (22.17 GOPs), best single 16x12 145.6 ms; per-n family: MR 32 / NR 4 (n ≤ 8), MR 32 / NR 6-7 (n 16-64), MR 16 / NR 12-15 (n ≥ 80) | #63 README, #79 B3-B4 |
| 5 | Ours against his GEMM | tile for tile (16x8, without Y00): his 123.4 ms, ours 141.6 ms; ours gains only from the tile choice; his Y00 swings 6.7-105.5 ms | #63, #79 B5-B6 |
| 6 | Correctness | 950/950 PASS at 8T (19 tiles x 50 shapes); scalar packer C1-C4 PASS; #85 30/30 per packer per board | #79 C3, #83, #85 |
| 7 | BERT/Square | 1T: 32x7 9.53 GOPs, paper picks hold (16x15 on B2); 8T: 16x8 48.27 GOPs, paper's 32x7 44.07 | #71 README |
| 8 | Scalar against vector, real workloads | YOLO 0.930 (1T) / 0.946 (8T), n = 4 down to 0.59; best tile changes on 14 / 17 layers; BERT/Square 0.983 / 0.987 (no measurable difference) | #83 README, brief |
| 9 | Why: pack share × extra pack cost | n sets pack share (87% at n = 4 → 7% at n = 1024); k and m set extra cost (1.2-3.4x small blocks, 1.02-1.12x large) | #83 T4, #85 brief |
| 10 | Parameter sweeps | one chart per sweep (n, m, k, tile) at 1T and 8T; tile hardly matters | #85 brief Part 1 |
| 11 | Decision table | vector packing for n ≤ 16 and k ≤ 64 (12-50%), either at n ≥ 128 on 8T | #85 README |
| 12 | Model check | predicted within 0.02 on 25 / 46 points per thread count | #83 Part 2, #85 Part 2 |
| 13 | What to tell Héctor | tile family per n group; keep the vector packer for YOLO; caveat: tiles timed in our GEMM, not his | slides 3-5, 8 |
| 14 | Open items + limits | see §3 and §4 | #79 §6.3, #82, #85 open losses |

## 3. Essential extra test (one)

**Our tile family inside Héctor's GEMM (#82 §1, #79 part D).**
- **Why essential:** Héctor most likely asked which tiles make *his* parallel GEMM faster. Every tile answer so far was timed in *our* GEMM (#79 A1, A2: GAP). Without this run, slide 13 can't say "use these tiles" for his code.
- **Run:** his `gemm_B3A2C0` (GEMMLib_yolo `b292c92`) in a scratch copy, choosing by n among kernels his library already has (32x8, 16x16; no new kernel code). Stock against patched, A-B-B-A, 8 threads, idle jbpi, his `TIMIN=2.0`.
- **Skip if** the user or Héctor confirms he meant "best tiles in a parallel GEMM" in general. Then slide 13 keeps the #79 caveat.

## 4. Not essential (stated as limits, not run)

- #82 §2 (his Y00 swing): the deck quotes ours against his without Y00.
- #82 §3 (our slow reps on core 0 at 8T): doesn't change any tile ranking.
- #83 BERT/Square ran vector on jbpi, scalar on jbpi2: gaps under 3% are already not called a difference. #85 confirms on one board that large n shows no packer effect at 8T.
- #85 open losses (1T 24x8 compute at large n; 8T 1200×256×256 16x4, uninstrumented only): shown as open, neither from packing.
- fp16 (Héctor's MR = 32 plan) and n = 1 GEMV: out of scope.

## 5. Closure criteria

- [x] Deck compiled (PDF + LaTeX) in `.scratch/reports/issue-91/`, charts from committed CSVs, glossary slide.
- [x] Every number traced to a committed log or report; board, build, protocol, cores named per slide.
- [ ] §3 run done (results added to slides 5 and 13) or skipped with the reason.
- [x] Linked from #58, #71, #79, #82, #83, #85 (issue files and GitHub bodies).
- [ ] Merged: PR → `pure-xdsl` → `main`, after the user approves.
