# Audit: did the YOLO work (#58, #63, #56) answer Héctor's request?

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/79

**Combined YOLO report (all findings, read first):** `.scratch/reports/yolo-multithread/report/yolo_report.pdf` ([LaTeX](../reports/yolo-multithread/report/yolo_report.tex)); index of every YOLO report, newest first: `.scratch/reports/yolo-multithread/README.md`.

**Combined report #91 (2026-11-18):** report [PDF](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/present-yolo-bert-square-and-packing-findings/.scratch/reports/issue-91/report/issue91_report.pdf) / [LaTeX](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/present-yolo-bert-square-and-packing-findings/.scratch/reports/issue-91/report/issue91_report.tex), slides [PDF](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/present-yolo-bert-square-and-packing-findings/.scratch/reports/issue-91/deck/issue91_deck.pdf) / [LaTeX](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/present-yolo-bert-square-and-packing-findings/.scratch/reports/issue-91/deck/issue91_deck.tex), [README](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/present-yolo-bert-square-and-packing-findings/.scratch/reports/issue-91/README.md). Covers #58 (#56, #61-#63), #71, #79, #82, #83, #85.
Reports: [A: fit to the request](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/audit-did-the-yolo-work-answer-hectors-request/.scratch/reports/issue-79/A-request-fit.md) · [B: numbers](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/audit-did-the-yolo-work-answer-hectors-request/.scratch/reports/issue-79/B-numbers.md) · [C: engine](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/audit-did-the-yolo-work-answer-hectors-request/.scratch/reports/issue-79/C-engine.md) · brief [PDF](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/audit-did-the-yolo-work-answer-hectors-request/.scratch/reports/issue-79/issue-79-audit-brief.pdf) / [LaTeX](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/audit-did-the-yolo-work-answer-hectors-request/.scratch/reports/issue-79/issue-79-audit-brief.tex)

**Role / Label:** `needs-triage`. Independent audit plus fixes. Host only; no board runs unless the user approves part D.
**Status:** A, B, C done 2026-11-17 (§6). Open items moved to #82 (§6.3). PR #80 merged to `pure-xdsl`, #81 to `main`.
**Branch:** `audit-did-the-yolo-work-answer-hectors-request`, cut from `pure-xdsl`.
**Scope:** the parent #58 and its result sub-issue #63 (S6, 8-thread tile sweep), plus #56 (S5, the parallel engine #63 runs on) where #63 relies on it.
**Primary sources:**
- `.scratch/reports/issue-58/Re: Optimizaciones Yolo sobre RISC-V.pdf`: the email thread, 10-13 Nov 2026
- `.scratch/reports/yolo26_k1_matrix_sizes.csv` (53 shapes)
- `.scratch/reports/issue-63/`, `.scratch/reports/issue-56/` (reports, raw logs, provenance)
- `.scratch/issues/0041`, `0040`, `0046` (#58, #56, #63)
- https://github.com/martineh/GEMMLib_yolo at `b292c92`, routine `gemm_B3A2C0`

---

## 0. Purpose: why a separate issue and branch

- **Test the whole system end to end.** Each of #56, #58, #63 checked its own part. This issue checks the chain as one: Héctor's request → shapes → parallel engine → tile sweep → comparison with his GEMMLib → the answer sent back.
- **Fix whatever was left behind.** Anything the audit finds broken, missing or wrong (code, tests, tables, reports, issue text, links) is fixed here, on this branch. Each fix is listed in §6 with its finding.
- **Stay independent.** The closed issues and their merged results stay as they are. Findings and fixes land here, so the audit trail is in one place and the closed work is not rewritten silently.

## 1. The request

Héctor, 13 Nov 2026: *"sería interesante que observaras los tamaños de micro-kernels que mejor rinden para la versión paralela de la GEMM con 8 hilos."* (look at which microkernel sizes run fastest in the parallel GEMM with 8 threads).

Context from the same thread:
- Adrián asks whether a size other than Héctor's 16x8 / 16x4 would help.
- Héctor's goal is whole YOLO inference under 0.5 s, all multi-threaded.
- His fast path packs A inside im2col. An fp16 version with MR = 32 is planned.

## 2. What #63 delivered

- Best tile per n group, timed in **our** generated GEMM at 8 threads (#56 engine): 32x4 (n = 4, 8), 32x7 (16, 32), 32x6 (64), 16x15 (80, 128), 16x14 (256).
- His GEMMLib timed as-is at 16x8 / 16x4, 8 threads, for comparison.
- Our best tiles inside **his** GEMM: skipped (#63 decision Q7).

## 3. Audit plan

### A. Did we answer the right question? (desk check)

1. Weigh the two readings of the request, with evidence for each:
   - best size in some parallel GEMM (what #63 answered);
   - best size for **his** parallel GEMM (needs the skipped Q7 run).
2. List every difference that could stop our best tile carrying over to his GEMM:
   - blocking: our cache model vs his MC = KC = 256, NC = n;
   - packing: his reads-consecutive packing vs ours;
   - thread split: our k-split B pack and fewer barriers vs his;
   - his A-prepacked im2col path vs the plain path `gemm.bin` times;
   - n = 1 (GEMV) shapes dropped: 53 shapes became 50.
3. Check the scope against his setup: same X60 chip (his OrangePi vs our BPI-F3), 8 threads on cores 0-7, fp32 only, the fp16 MR = 32 plan.

### B. Are the numbers right? (host, committed logs only)

1. Recompute #63's tables from the raw logs: best tile per shape and per n group, both GOPs metrics for ours, mean time for his.
2. Check every row PASS, 3 + 3 launches in A-B-B-A order, idle board, and `provenance.txt` (his commit `b292c92`, threads, cores, build).
3. Check each "best" tile beats the runner-up by more than the launch-to-launch spread.
4. Explain the two open anomalies: his Y00 (12.9 → 105.5 ms across launches) and our median rep far above our mean at 8 threads. Check whether either flips "ours 22.17 vs his 16.18 GOPs".

### C. Is the engine sound? (#56, light check)

1. Re-run `test_parallel_gemm.py` on the host: 1-thread output byte-identical, 3- and 8-thread output correct. Not during a board series; restore `generated/` after.
2. Read `render_parallel_nest` against the split and barriers #56 §5 promises.

### D. Optional, not in this issue's scope

Build his GEMMLib with our best tile per n group (local patch, never pushed) and time it against his stock tiles at 8 threads on an idle board. Only after A's verdict and the user's go.

## 4. Output

- A pass / gap / fail verdict per item of A, B, C, in this file.
- Fixes for every gap or fail, committed on this branch (§6).
- A link from #58.

## 5. Closure criteria

- [x] A: verdict on the reading, list of transfer gaps.
- [x] B: tables recomputed and matched (or differences listed), anomalies explained or flagged.
- [x] C: tests run, code read against #56 §5.
- [x] Every gap or fail fixed, or listed as left open with a reason.
- [ ] Verdicts committed, linked from #58, branch merged per the usual flow (PR → `pure-xdsl` → `main`).

## 6. Findings and fixes

### 6.1 Verdicts

| item | verdict | finding |
|---|---|---|
| A1 the question | **GAP** | Héctor most likely asked for the best tile for *his* parallel GEMM. #63 measured *our* GEMM; the step that tests our tiles in his GEMM (#63 Q7) was skipped. |
| A2 transfer | **GAP** | His blocking (MC = KC = 256), barriers (one per `ic` block), kernels (none of our winners exist in his library), im2col coupling (`PK_MR = YOLO_MR`, one MR for the whole network) and fp16 plans all differ from what #63 timed. |
| B1 correctness | **FAIL → fixed** | All 2850 rows of #63's 8-thread sweep ran with the oracle off (`skip_ref_run: true`). Only 24x8 had been checked at 8 threads (#56). |
| B2 headline | PASS | Every #63 number reproduces within 0.1 ms with an independent script. |
| B3 best per shape | PASS | Picked on one launch and scored on the other two, it is still 8-11 ms per pass faster than the best single tile. |
| B4 best per n | **GAP** | Only n = 4, 8, 32 keep the same winner in all 3 launches. Read the list as a family: MR 32 / NR 4 for n ≤ 8, MR 32 / NR 6-7 for n 16-64, MR 16 / NR 12-15 for n ≥ 80. |
| B5 ours vs his | **GAP** | "Ours faster" rests on his Y00 (6.7-105.5 ms over 9 launches). Without Y00, at the same tile (16x8), his engine is 15% faster: 123.4 against 141.6 ms per pass. |
| B6 anomalies | named, not proven | His Y00: 300 barriers per call, and it is the first call of each process. Our slow reps: core 0 spends 13.1% in the kernel against 0.2-0.5% on cores 1-7, with threads bound. |
| C1 host tests | PASS | 21/21 `test_parallel_gemm.py`; 623/623 full suite. |
| C2 code read | PASS | `render_parallel_nest` matches #56 §5; no race found. |
| C3 board check | PASS | 19 tiles x 50 shapes at 8 threads against the scalar oracle: 950/950 PASS, `logs/run_2026-11-17_092354/` (`jbpi`, 2026-11-17 08:34-09:24, other users none). |

### 6.2 Fixes on this branch

- `tests/pure_xdsl_blis_rvv/config_yolo_correctness_8t.json` and the board pass C3 (fixes B1).
- `.scratch/reports/issue-63/README.md`: "Read this before using the tile list": whose GEMM, family not exact winners, tile-for-tile numbers, correctness (A1, A2, B4, B5).
- Audit links in #56, #58, #63 (`.scratch/issues/0040`, `0041`, `0046` and their GitHub bodies).
- `.scratch/reports/issue-79/`: the three reports, `recompute_63.py`, `rep_tails.py`, the brief (PDF + LaTeX).

### 6.3 Left open (needs the user; each is a board run)

Moved to #82 (`.scratch/issues/0049-yolo-follow-up-board-runs-after-audit.md`) on 2026-11-17; the user will run them later.

1. **Which reading Héctor meant.** If his GEMM: part D, his `gemm_B3A2C0` with his own 32x8 / 16x16 kernels chosen by n (local patch, no new kernel code), A-B-B-A against stock at 8 threads.
2. **His Y00:** one short `gemm.x` run with Y00 moved to a later row.
3. **Our slow reps:** 8 threads on cores 0-7 against 7 threads on cores 1-7.
