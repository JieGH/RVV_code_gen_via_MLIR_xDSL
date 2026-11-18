# Why our YOLO GEMM is slower than Héctor's: find the reasons in the algorithm

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/95

**Combined YOLO report (all findings, read first):** `.scratch/reports/yolo-multithread/report/yolo_report.pdf` ([LaTeX](../reports/yolo-multithread/report/yolo_report.tex)); index of every YOLO report, newest first: `.scratch/reports/yolo-multithread/README.md`.

**Role / Label:** `needs-triage`. Opened 2026-11-18 at the user's request. Diagnosis, board runs on `jbpi`.
**Status:** open. Main cause found and measured (§7): our A packer reads A in a slow order. Report written (§8). The fix is the pending issue #96, not started.
**Report:** [PDF](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issue-95/report/issue95_report.pdf) / [LaTeX](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issue-95/report/issue95_report.tex), [README](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issue-95/README.md).
**Branch:** `why-our-yolo-gemm-is-slower-than-hectors`, cut from the #94 branch (not `pure-xdsl`), because this work reads #94's committed logs, which are not on `pure-xdsl` until #94 merges.
**Related:** #94 [[0053-yolo-gemm-at-1-and-8-threads-our-vector-packer-our-scalar-packer-and-hectors]] (the gap), #62 [[0045-read-hector-parallel-gemm]] (his code read), #79, #82.

---

## 1. Question

User: in #94, our GEMM is slower than Héctor's on the same YOLO shapes. Go deeper and find why, in the algorithm.

#94 numbers. Both sides on jbpi, GOPs by mean time, mean of 3 launches. Ours: tile 24x8, uninstrumented build, reference protocol, `--tmin 2`. Héctor's: `gemm.bin`, `TIMIN=2.0`.

| | 1T (core 4) | 8T (cores 0-7, bound), without Y00 |
|---|---|---|
| ours, vector packer | 4.27 | 19.59 |
| Héctor's | 5.10 | 22.98 |

## 2. Where the gap is (#94 per-shape data, 1T)

Ratio = our GOPs / his GOPs, by mean time, 1T core 4. Source: `.scratch/reports/issue-94/report/data/shape_rows.tex`.

- **Largest losses:** large m with small k.
  - Y10 4800x16x32: 0.31
  - Y01 19200x16x72: 0.39
  - Y00 76800x16x27: 0.41
  - Y13 4800x32x32: 0.42
  - Y03 19200x32x32: 0.43
  - Y05 19200x8x144: 0.46
- **Gap closes as k grows** at the same m and n:
  - Y10 4800x16x32: 0.31
  - Y08 4800x16x144: 0.51
  - Y09 4800x16x288: 0.93
  - Y11 4800x16x576: 0.93
- **We win on every m = 300 shape:** 1.03 to 1.73.

## 3. Plan (diagnosis)

1. **Test loop.** A script runs one shape on both sides, back to back, 1T core 4 on an idle jbpi, and prints both GOPs. It is red while our GOPs are clearly below his on Y10. The same build and protocol as #94, with a shorter time per shape.
2. **Smallest case.** Cut the shape down while the gap stays.
3. **Hypotheses.** 3 to 5, ranked, each with a prediction, listed here before testing.
4. **Probes.** Change one thing at a time on our side and measure with the loop.
5. **Report.** Causes with measured numbers.

Limits:
- Héctor's code stays unchanged.
- Our vector path changes only in scratch probes, unless the user asks for a fix.
- Every quoted number comes from a committed log.
- Idle board, never core 0 for 1T, never kill other users' jobs.

## 4. Test loop

`tests/pure_xdsl_blis_rvv/probe_issue95.sh M N K [--tile 24x8] [--tmin 1]`.

- It runs one shape on both sides, 1T core 4 on jbpi, back to back, and stops if another user is busy.
- Ours: uninstrumented build, reference protocol, `--no-nc-overrides`, vector packer, as in #94. Héctor's: his built `gemm.x`, `TEST=T`.
- It prints both GOPs (by mean time) and RED when ours / his < 0.8.
- Logs go to `logs/issue95_probe_<stamp>_<label>/`.
- One run takes about 2 minutes, mostly our build.

| run | shape | ours | his | ratio |
|---|---|---|---|---|
| y10 | 4800x16x32 | 1.82 | 5.12 | 0.36 RED |
| y10b | 4800x16x32 | 1.82 | 5.15 | 0.35 RED |
| y11 | 4800x16x576 | 2.07 | 2.15 | 0.96 GREEN |

Our GOPs barely move with k (1.82 to 2.07), while his fall from 5.15 to 2.15. So he gains something at small k that we lack.

## 5. Code read: the two loop nests

Both are BLIS loops in the same order: jc, pc (pack B), ic (pack A), jr, ir, microkernel.

| | ours | Héctor's (`gemm_B3A2C0`, `gemm_blis.c`) |
|---|---|---|
| mc | analytical model, per shape | fixed `YOLO_MC` = 256 |
| kc | model | fixed `YOLO_KC` = 256 |
| nc | n (no overrides) | n |
| tile | 24x8 (generated) | 16x8; 16x4 at n = 4 (hand-written intrinsics) |
| pack A | vector packer (RVV) | `pack_generic_MRxkc_A` |
| compiler | g++ 13 `-O2` | gcc-14 `-O3` |

- Checked: his build has `-DGEMMLIB_ON` and no `OPENBLAS_ON` or `BLIS_ON`. So `gemm.bin` times his own code, not OpenBLAS.

Our model's blocking (#94 console log `parallel_gemm_2026-11-18_115959/L1_T1_jbpi.log`):

- For m > 300, mc x kc is near 96 K floats (about 384 KB, an L2-sized A block):
  - Y10: 2560 x 32
  - Y01: 1365 x 72
  - Y17: 1536 x 64
  - Y11: 578 x 170
- For m = 300, mc = 300 (all of m). We win on every m = 300 shape.
- His A block is 256 x min(k, 256) floats: 32 KB for Y10 (k = 32), so it fits the 32 KB L1D. At k = 256 it is 256 KB.

## 6. Hypotheses (ranked)

1. **H1: our mc is too large at small k.**
   - The model sizes the packed A block for L2. On every jr pass the microkernel then streams A from L2; his MC = 256 keeps A in L1 when k is small.
   - Prediction: forcing our mc to 256 on Y10 lifts ours from 1.8 to 4 GOPs or more. On Y11 (k = 576) it changes little.
2. **H2: our 24x8 microkernel is slower than his 16x8 on data in L1.**
   - Prediction: after H1, a gap of more than 10% stays. Our GEMM at tile 16x8 with mc = 256 then gets within 10% of his.
3. **H3: our A pack costs more.**
   - Prediction: the instrumented build shows a pack share above 30% on Y10, and it drops with mc = 256.
4. **H4: compiler flags (-O2 against -O3).**
   - Prediction: building ours with -O3 changes Y10 by less than 10%.

Probe for H1: an opt-in `RVV_FORCE_MC` env knob next to the existing `RVV_FORCE_NC`. It is off by default and does not change the default build.

## 7. Results

All on jbpi core 4, 1 thread, idle board. Probe logs are in `tests/pure_xdsl_blis_rvv/logs/issue95_probe_*` and pack tests in `logs/issue95_pack_order_*`.

### 7.1 H1 (mc too large): small effect, prediction failed

| run | shape | mc | ours | his |
|---|---|---|---|---|
| y10 | 4800x16x32 | 2560 (model) | 1.82 | 5.12 |
| y10_mc240 | 4800x16x32 | 240 (`RVV_FORCE_MC`) | 2.27 | 5.09 |
| y11_mc240 | 4800x16x576 | 240 | 1.84 (model mc: 2.07) | 2.21 |

Uninstrumented build, reference protocol, `--tmin 1`, GOPs by mean time. A smaller mc helps Y10 by about 25%; it does not close the gap.

### 7.2 H3 (A pack): confirmed. Our pack takes most of the time

Instrumented build (only the shares count; its timers cost 11-17%):

| shape | vector pack | compute | pack share |
|---|---|---|---|
| Y10 4800x16x32 | 2.16 ms | 0.63 ms | 77% |
| Y05 19200x8x144 | 41.53 ms | 4.64 ms | 90% |

- On Y10 our compute alone is about 7.8 GOPs (4.9 MFLOP / 0.63 ms). That is faster than Héctor's whole call (5.1 GOPs). The kernel is not the problem (H2 is not supported).
- The Y10 pack runs at IPC 0.04, with 96% back-end stall and 24.5% L1D read misses.
- Our scalar packer is as slow (2.46 ms). So the cause is not vector against scalar.

### 7.3 The cause: the order in which A is read

A is column-major (lda = m). Both packers write the same MR-row panels:

- **Ours** (`pack_RB_v_*`, generated): for each MR-row panel, for each column, it reads MR floats. Each read jumps lda floats (19.2 KB on Y10) to the next column, so every read hits a different 4 KB page, and the next panel goes back over the same pages.
- **Héctor's** (`pack_generic_MRxkc_A`): for each column, it reads the whole mc block, panel after panel. Each column is one long run of contiguous memory.

Test: `tests/pure_xdsl_blis_rvv/issue95/pack_order.c`, run on the board. It packs A both ways with the same RVV copy, the same bytes and the same packed layout (checked with `memcmp`). It packs in mc x kc blocks, the same blocks as our GEMM. On Y10 and Y05 it agrees with the in-GEMM instrumented pack within about 10% (Y10: 2.19 against 2.16 ms; Y05: 46.3 against 41.5 ms).

| shape | MR | mc | kc | our order | his order |
|---|---|---|---|---|---|
| Y10 4800x16x32 | 24 | 2560 | 32 | 2.19 ms | 0.71 ms |
| Y00 76800x16x27 | 24 | 3034 | 27 | 34.69 ms | 12.13 ms |
| Y05 19200x8x144 | 24 | 682 | 144 | 46.27 ms | 16.20 ms |

All 50 shapes (`logs/issue95_pack_order_50_2026-11-18_*.txt`, our model's mc and kc, MR 24), set beside the #94 1T pass times:

| shapes | #94 ours | #94 his | gap | A pack, our order | A pack, his order | order cost |
|---|---|---|---|---|---|---|
| all 50 | 696.9 ms | 583.4 ms | 113.6 ms | 454.9 ms | 200.4 ms | 254.5 ms |
| m > 300 (33) | 607.5 ms | 467.0 ms | 140.5 ms | 445.0 ms | 193.3 ms | 251.7 ms |
| m = 300 (17) | 89.4 ms | 116.4 ms | -27.0 ms | 9.9 ms | 7.1 ms | 2.8 ms |

- On the 33 shapes with m > 300, the pack test puts the read-order cost at 251.7 ms per pass, against a gap of 140.5 ms. **Correction (report step):** the test overstates absolute pack times. On Y01, Y05, Y09 and Y11 its pack time is above our whole call, and it overstates most on the k > kc shapes. Split by k:
  - m > 300, k <= kc (23 shapes): gap 131.0 ms, order cost 162.2 ms. Same size; the test matches the GEMM best here.
  - m > 300, k > kc (10 shapes): gap 9.5 ms, test order cost 89.5 ms. Not usable.
  - So the gap sits on the large-m shapes where A is packed in one block of k, and there the order effect is the same size as the gap. The exact gain needs the in-GEMM change (#96).
- On the m = 300 shapes A is small, the order costs little, and we are ahead.

### 7.4 Not yet measured

- Our GEMM with a column-order A packer. This needs a change to the generated packer (the vector path): pending issue #96.
- 8 threads. The same packer runs per thread, but the 8T gap also includes Héctor's Y00 behaviour (#94).
- MR 16 against MR 24 in his order: 0.34 ms against 0.71 ms for Y10's A. This is a smaller second effect.
- H4 (compiler): not tested. The kernels already build at `-O3` (`CFLAGS`); only the driver and pack C++ build at `-O2`.

## 8. Report

`.scratch/reports/issue-95/report/issue95_report.pdf` (5 pages; [PDF](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issue-95/report/issue95_report.pdf), [LaTeX](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issue-95/report/issue95_report.tex)). Data from `report/make_data.py`, which reads only committed logs.

It contains the answer, a glossary, where the gap is, the method, H1, H3, the cause with code side by side, two charts (gap against order cost per shape for m > 300; pack time in both orders for Y00, Y05, Y01, Y10), the group table split by k, the limits of the pack test, the not-measured list, and all 50 shapes.

## 9. Closure

- [x] Test loop, hypotheses, probes
- [x] Cause found: A pack read order (§7)
- [x] Report written and linked
- [x] Pending fix issue opened: #96 (no implementation)
