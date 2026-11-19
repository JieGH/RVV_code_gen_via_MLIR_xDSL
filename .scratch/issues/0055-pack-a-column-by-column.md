# Pack A column by column, as Héctor's packer does (pending)

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/96

**Combined YOLO report (all findings, read first):** `.scratch/reports/yolo-multithread/report/yolo_report_v2.pdf` ([LaTeX](../reports/yolo-multithread/report/yolo_report_v2.tex)); index of every YOLO report, newest first: `.scratch/reports/yolo-multithread/README.md`.

**Role / Label:** `needs-triage`. Opened 2026-11-18 at the user's request, from the #95 finding. **Pending: no implementation until the user says go.**
**Status:** open, pending.
**Related:** #95 [[0054-why-our-yolo-gemm-is-slower-than-hectors]] (the finding; report [PDF](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issue-95/report/issue95_report.pdf) / [LaTeX](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issue-95/report/issue95_report.tex)), #94 (the gap), #83/#85 (vector against scalar packer).

---

## 1. Finding (from #95)

- Our A packer (`pack_RB_v_*`, generated) fills one MR-row panel at a time. For each panel it reads MR floats from each of the kc columns of the column-major A, jumping lda floats between reads.
- Héctor's `pack_generic_MRxkc_A` reads each column once from top to bottom and spreads it over the panels. Same bytes, same packed layout.
- On jbpi, 1 thread on core 4, instrumented build (shares only), the pack is 77% of our time on Y10 (4800x16x32) and 90% on Y05 (19200x8x144).
- Pack test on the board (`tests/pure_xdsl_blis_rvv/issue95/pack_order.c`), same blocks as our GEMM: our order 2.19 ms against his order 0.71 ms (Y10), 34.69 against 12.13 ms (Y00), 46.27 against 16.20 ms (Y05).
- The 1T gap to Héctor (#94) is 131.0 of 140.5 ms on the 23 shapes with m > 300 and k <= kc. The pack test puts the order cost there at 162.2 ms, the same size. It overstates absolute pack times (4 shapes read above our whole call), so the exact gain needs an in-GEMM change.

## 2. Observations to keep in mind

- Our scalar packer has the same read order and is as slow, so both packers would change.
- MR 16 packed in his order was faster than MR 24 (0.34 against 0.71 ms for Y10's A, two separate tests). This is a second, smaller effect.
- A smaller mc (`RVV_FORCE_MC=240`) helped Y10 by about 25% only.
- The k > kc shapes show a small gap (9.5 ms) even though the test reports a large order cost. Measure them in the GEMM; don't assume.
- At 8 threads each thread packs its own A block the same way. The 8T gap also includes Héctor's Y00 behaviour (#94).

## 3. What a fix would be (not started)

- Generate the A packer in column-outer order (loop over the kc columns outside the MR-row panels). Keep the packed layout, so the microkernel is unchanged.
- Check: correctness (scalar oracle, all rows PASS), then rerun the #94 series (V and H, 1T and 8T, 3 launches, A-B-B-A, idle jbpi) and compare with #94.

## 4. Closure

- [ ] User says go
- [ ] Column-order A packer, host tests
- [ ] Correctness on the board
- [ ] #94 series rerun, report
