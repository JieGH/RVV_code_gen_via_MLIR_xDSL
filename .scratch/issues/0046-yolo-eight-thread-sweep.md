# S6: 8-thread sweep: best tile per YOLO shape with 8 threads, and our GEMM against Héctor's GEMMLib

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/63

**Role / Label:** `needs-triage`. Sub-issue of #58 ([[0041-yolo-best-microkernel-size-parent]]). Board run.
**Status:** 📝 Open. Not run.
**Blocked by:** #61 (S3, one-core sweep), #56 (S5, parallel GEMM).
**Target:** `jbpi` or `jbpi2`, all 8 cores.
**Shapes:** `datasets/yolo26_k1.dat`.

---

## 1. Question

This is Héctor's request: **with 8 threads, which tile is fastest on each YOLO shape?** It also asks two more things:
- Does the answer differ from one core (S3)?
- How does our parallel GEMM compare with his `gemm_B3A2C0` on the same board, the same shapes and 8 threads?

## 2. Decision needed before running: core 0

8 threads on 8 cores must use core 0. House rule #1 (issue #39) forbids core 0 because the board sends every device interrupt there, and all scripts refuse `--cpu 0` unless `--allow-core-0` is given. Options:

- **(a)** Allow core 0 for this series only, through a new `--threads 8` path that pins threads 0-7 and records it in `provenance.txt`. Every number carries the note.
- **(b)** Also run 7 threads (cores 1-7) and 4 threads (cores 4-7, one cluster) as clean points beside the 8-thread one.
- Proposed: (a) + (b), so the 8-thread number answers Héctor and the 4- and 7-thread numbers show what core 0 costs.

## 3. Run

1. Our side: **our design's size search** (S2 #60) with **our design's** `--threads 8` (#56), 3 launches, idle board, both GOPs metrics.
2. Héctor's side: his `GEMMLib_yolo` built **on the board from his repo at a fixed commit**, with `gemm.bin` (`TH=8`, his `yolo.dat`), as the program to compare against. The same launches alternate with ours (A-B-B-A). His tile stays 16x8 / 16x4.
   - Do not edit his code. Wrapper scripts live in our repo.
   - Record his commit hash in `provenance.txt`.
3. Optionally, if S4 shows it is easy: rebuild his GEMM with our best tile per n group (change `YOLO_MR`/`YOLO_NR` and the `uk` choice in a local patch, not committed to his repo). This shows whether the tile alone helps his library.

## 4. Output

- Table per shape: best tile on 1 core (S3) and on 8 threads, GOPs by mean time and by median rep, his GEMMLib GOPs at 8 threads.
- The answer for Héctor: which tiles to use per n group at 8 threads, with the board, build, protocol, threads and cores named.
- Logs committed.

## 5. Limits to state with the numbers

- His fast path takes A pre-packed from im2col (`gemm_prepack_B3A2C0`). `gemm.bin` times the plain path, so check which one it runs (S4 §3.5).
- n = 1 shapes are left out (his GEMV).
- Any core-0 run carries the house-rule note.

## 6. Closure criteria

- [ ] Core-0 decision recorded.
- [ ] 3 launches per side, alternating, idle board, logs committed.
- [ ] Table and short answer committed and linked from the parent.
