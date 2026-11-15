# S1: YOLO26 shape set: dataset file, weight of each shape, and which tiles fit each shape

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/59

**Role / Label:** `needs-triage`. Sub-issue of #58 ([[0041-yolo-best-microkernel-size-parent]]). Desk work, no board.
**Status:** 📝 Open. The shape list is saved. Nothing else is done.
**Shapes:** 53 shapes, `HectorYolo/yolo26_k1_matrix_sizes.csv` (same as `gemm_study/yolo.dat` in Héctor's repo).
**Primary sources:** the CSV; `tests/pure_xdsl_blis_rvv/datasets/*.dat` (format); `generate_pure_blis_project.py` (register budget, issue #32 plan).

---

## 1. Goal

Turn Héctor's shape list into the inputs the sweep (S3) needs, and find out on paper which tiles can do well on which shapes before any board time.

## 2. What the shapes look like (computed 2026-11-15 from the CSV)

- **m** (im2col rows, i.e. output pixels): 76800, 19200, 4800, 1200, 300.
  - 300 is not a multiple of 16 or 32 (fringe rows). 1200 is not a multiple of 32.
- **n** (output channels): 1 (3 shapes), 4 (3), 8 (1), 16 (10), 32 (6), 64 (10), 80 (6), 128 (9), 256 (5).
  - n is small, so **NR decides how much of each column block is wasted**. For example, NR=15 on n=16 wastes almost a whole second tile, and NR=8 fits every n except 1 and 4.
- **k**: 9 to 2304. Seven shapes have k <= 32, so the microkernel loop is very short there and packing and loop overhead weigh more.
- **Total**: 2.97 GFLOP for one pass over the 53 distinct shapes. The top 12 shapes hold 64% of it. Top 4:

| # | m | n | k | MFLOP | cumulative |
|---|---|---|---|---|---|
| 22 | 1200 | 128 | 1152 | 353.9 | 11.9% |
| 16 | 4800 | 64 | 576 | 353.9 | 23.8% |
| 42 | 300 | 256 | 1152 | 176.9 | 29.8% |
| 2 | 19200 | 32 | 144 | 176.9 | 35.7% |

These weights assume each shape runs once per image. That is not known (parent §5.4).

## 3. Work

1. Add `tests/pure_xdsl_blis_rvv/datasets/yolo26_k1.dat`: one row per shape, with label `Y<l>_m<m>_n<n>_k<k>` and a header pointing at the CSV and at Héctor's repo commit. Leave out the three n = 1 shapes (GEMV, parent §5.2), and say so in the header. Check that the M, N, K columns follow the same order as Héctor's (A is m x k, B is k x n, C is m x n, MR runs along m). His driver passes `m, n, k` to `gemm_B3A2C0` in that order (`src/driver_convolution.c:298`).
2. Add a shape table script, or extend `shape_breakdown.py` if it fits. For each shape, print:
   - MFLOP and share of the total
   - for each candidate tile: the full tiles and the fringe waste along m and along n, as extra multiply-adds over the useful ones
3. **Candidate tiles** (fp32, VLEN 256 = 8 floats per register; register budget `(MR/8) x (1+NR) <= 32`, issue #32):
   - MR=8: NR in {4, 8, 12, 16, 24}
   - MR=16: NR in {4, 6, 8, 10, 12, 14, 15}
   - MR=24: NR in {4, 6, 8, 9}
   - MR=32: NR in {4, 6, 7}

   That is 19 tiles. They include Héctor's 16x8 and 16x4. The list is a proposal for review: drop or add tiles here, not in S3.
4. Ask Héctor (by email, user's call) for the number of calls per shape per image, or his time per shape. Write the answer into this issue.

## 4. Not done

- No timing. This is all paper.
- The calls-per-image weights are unknown until Héctor answers.

## 5. Closure criteria

- [ ] `datasets/yolo26_k1.dat` committed (50 shapes) with its header.
- [ ] Shape table (MFLOP, share, fringe waste per candidate tile) committed and linked here.
- [ ] Candidate tile list agreed by the user.
- [ ] Calls-per-image answer from Héctor recorded, or marked as still missing.
