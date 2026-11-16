# S6: 8-thread sweep: best tile per YOLO shape with 8 threads, and our GEMM against Héctor's GEMMLib

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/63

**Role / Label:** `needs-triage`. Sub-issue of #58 ([[0041-yolo-best-microkernel-size-parent]]). Board run.
**Status:** 📝 Open. Not run.
**Blocked by:** nothing. #61 (S3) done; #56 (S5, parallel GEMM) done 2026-11-16: `--threads 8`, report `HectorYolo/issue-56/`.
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

**Decided 2026-11-16 (grilling, defaults accepted): (a) only.** Fact: `--threads N` pins cores `0-(N-1)` (`compile_macro_bench.sh:241`), so 7 or 4 threads would still use core 0; (b) needs a new core-list option. #56 and #62 already ran 8 threads on cores 0-7 under the same exception. (b) stays an optional follow-up, not part of this issue.

## 2a. Decisions (grilling, 2026-11-16, defaults accepted)

| # | Question | Decision |
|---|---|---|
| Q1 | Core 0 | (a) only: 8 threads on cores 0-7, `OMP_PROC_BIND=close OMP_PLACES=cores`, note in `provenance.txt` and every table. No 4-/7-thread runs. |
| Q2 | Our runner | Add `--threads N` to `run_yolo_sweep.sh` (passed to `compile_macro_bench.sh`, replaces `--cpu`). No new sweep script. |
| Q3 | Tiles | All 19 Candidate Tiles of `config_yolo_search.json` (#61), rotated per launch as in #61. |
| Q4 | Settings | Board `jbpi` (`jbpi2` if busy, one board for the whole series), uninstrumented, reference protocol, `--tmin 2`, `--no-nc-overrides`, vector packer: as #56 and #61. |
| Q5 | Héctor's side | Rerun, not reuse #62: his `gemm.x` at commit `b292c92`, 8 threads on cores 0-7, `TIMIN=2.0`, via `run_hector_gemm.sh`. Add an 8-thread-only mode to it. |
| Q6 | Alternation | One thin series script runs launches ours, his, his, ours, ours, his (3 each, A-B-B-A) on one board, with the usual idle check before each launch. |
| Q7 | §3 step 3 (patched GEMMLib with our tiles) | Skip. Optional follow-up only. |
| Q8 | Output | `HectorYolo/issue-63/` with series folders, `README.md` and `report/` in the issue-62 style; 1-core column from #61's committed logs; both GOPs metrics for ours, mean time only for his. |

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
- Blocking differs: ours from the cache model, his fixed at MC=256, KC=256, NC=n. Our `nc` overrides set `nc` only, so his blocking is not reproduced on our side (decided in #60, 2026-11-15).
- Any core-0 run carries the house-rule note.

## 6. Closure criteria

- [ ] Core-0 decision recorded.
- [ ] 3 launches per side, alternating, idle board, logs committed.
- [ ] Table and short answer committed and linked from the parent.
