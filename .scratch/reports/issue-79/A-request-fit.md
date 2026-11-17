# Issue #79, Part A: did #58/#63 answer the question Héctor asked?

Audit date 2026-11-17. Desk check only, no board. Sources:
- the email thread `.scratch/reports/issue-58/Re: Optimizaciones Yolo sobre RISC-V.pdf` (10-13 Nov 2026);
- Héctor's code, `github.com/martineh/GEMMLib_yolo` at `b292c92` (cloned fresh for this audit, read only): `src/GEMM/gemm_blis.c`, `gemm_blis.h`, `uk.c`, `src/driver_gemm.c`, `gemm.bin`;
- `.scratch/issues/0041` (#58), `0046` (#63), `0040` (#56); `.scratch/reports/issue-63/`.

**Verdict: GAP.** #63 answers a nearby question ("which tile is fastest in *our* parallel GEMM on his shapes"), not the most likely reading of his ("which tile would make *his* parallel GEMM faster"). The step that would bridge the two (#63 §3 step 3, decision Q7) was skipped. The tile list #63 reports also can't be dropped into his library as it stands (A2). Nothing in #63 is wrong; the answer is indirect, and the issue text and README don't say so.

---

## A1. What was asked

> Héctor, 13 Nov: *"Jie, comentamos ayer que sería interesante que observaras los tamaños de micro-kernels que mejor rinden para la versión paralela de la GEMM con 8 hilos."*
> Adrián, 10 Nov: *"... si ha conseguido medir el rendimiento de los micro-kernels. Más que nada por si interesa otro tamaño."*

| reading | evidence for | evidence against |
|---|---|---|
| **R1: best tile for *his* parallel GEMM** (`gemm_B3A2C0`) | "**la** versión paralela de **la** GEMM": the GEMM named in the same email, whose code he sends in the same message. Adrián: "whether another size is of interest", meaning another size for their library (today 16x8 / 16x4). The paper is about their GEMMLib inside PyTorch YOLO. | none in the thread |
| R2: best tile in any parallel GEMM, ours included | #58 §1 chose this plan: build both features in our design, use his GEMM only to read and to compare. | the thread never mentions our generator as the GEMM to tune |

R1 is the most likely meaning. #58 took R2 on purpose (§1: "We do not run our kernels inside his GEMM"), and #63 Q7 skipped the one step that would carry the answer over to R1 (rebuild his GEMM with our best tiles). Only the user or Héctor can settle the reading for certain.

## A2. Why our best tile may not carry over to his GEMM

Each row is a difference, read from his code at `b292c92`, between the GEMM #63 timed and the GEMM Héctor ships.

| # | what | ours (#63) | his (`gemm_B3A2C0`) | effect on the tile answer |
|---|---|---|---|---|
| 1 | blocking | cache model per shape: `mc` 300-4096, `nc` = n, `kc` from the model | fixed `MC = 256`, `KC = 256`, `NC = n` (`gemm_blis.h:23-25`) | the tile's cost depends on `mc`/`kc`. With `MC = 256` and 8 threads, MR = 32 gives each thread exactly 1 panel per `ic` block, MR = 16 gives 2, and MR = 24 doesn't divide 256. Our ranking was made at other `mc` values |
| 2 | thread split | B pack split over (panel x k-chunk); one barrier per `pc` plus one before a short last `ic` block | B pack split over NR panels only; a barrier after the B pack and one after **every** `ic` block | his barrier per `ic` block costs more for small MR (more panels to share out) and for large m (Y00: 300 barriers per call). The best tile can differ for this alone |
| 3 | which kernels exist | any MR x NR, generated | hand-written fp32 only: 16x16, 16x8, 16x4, 16x1, 64x1, 32x8, 8x8, 8x16 (`uk.c`) | none of #63's per-n winners (32x4, 32x7, 32x6, 16x15, 16x14) or its best single tile (16x12) exists in his library. 32x8, 16x16 and 8x16 do exist and are the closest |
| 4 | MR coupled to im2col | n/a | `PK_MR = YOLO_MR` ("debe coincidir con la GEMM", `gemm_blis.c:245`): the im2col+pack step packs A in MR-row panels for the whole network | one MR for the whole network. #63's answer changes MR by n (32 for n ≤ 64, 16 for n ≥ 80); his fast path can't take that without per-layer MR in `_im2col_pack_rvv` |
| 5 | path timed | plain GEMM call (A packed inside the call) | `gemm.x` calls `gemm_rvv` → `gemm_B3A2C0` (plain path, `pack_ready = 0`). In YOLO inference most layers take `gemm_prepack_B3A2C0` (A already packed by im2col) | #63 times the path his paper doesn't use for im2col layers. The prepack path skips the A pack, so the tile's share of the time is larger there |
| 6 | data type | fp32 | fp32 today; the planned last step is fp16 inside pack/kernel with MR = 32 (email, 10 Nov) | the MR = 32 fp32 results don't predict fp16 32x8 (half the register bytes per row, different LMUL) |
| 7 | n = 1 | left out (3 shapes) | GEMV (`gemv_rvv_fp32`) | correct to leave out: no tile is used there |
| 8 | thread binding | `OMP_PROC_BIND=close`, cores 0-7 | `taskset -c 0-7`, no binding | changes how much a core-0 interruption costs (see Part B4) |

## A3. Scope against his setup

| item | his | ours | match? |
|---|---|---|---|
| chip | SpacemiT K1 in an OrangePi (3 GB RAM, email) | SpacemiT K1 / X60 in BPI-F3 `jbpi` (`provenance.txt`: `Spacemit(R) X60`, 8 harts, governor `performance`) | yes, same chip; board, RAM and clock settings not checked on his side |
| threads / cores | `TH=8`, `taskset -c 0-7` (`gemm.bin`) | 8 threads, cores 0-7 | yes |
| shapes | 53 rows, `gemm_study/yolo.dat` | 50 rows (`datasets/yolo26_k1.dat`, n = 1 left out); the other 50 match row for row (Part B1) | yes |
| weight of each shape | real inference runs each layer once per image; time per layer unknown | each shape counted once | unknown (stated in #63 §5) |
| metric | whole YOLO inference time ("barrera del medio segundo") | GEMM-only GOPs | partial: GEMM-only, as asked |

## A4. Was the answer sent?

Not visible from this checkout. The Gmail account connected here has no copy of the thread (searched `subject:Optimizaciones Yolo`, last 30 days). `.scratch/reports/issue-63/README.md` is the only answer text. It gives the per-n winners without saying that (a) they come from our GEMM, (b) his library has none of those tiles, (c) most per-n winners are inside launch noise (Part B4).

## A5. Fixes made on this branch

- `.scratch/reports/issue-63/README.md`: a "Read this before using the tile list" block: R1/R2, the kernels that exist in his library, the MR coupling with im2col, and the noise in the per-n winners. Numbers unchanged.
- `.scratch/issues/0041` (#58) and `0046` (#63): a line linking this audit.

## A6. Left open (needs the user)

- **Which reading Héctor meant.** If R1: the direct test is #63 Q7 / #79 part D. It needs **no new kernel code**: his library already has 32x8, 16x16 and 8x16 fp32. Rebuild `gemm.x` with the `uk` choice in `gemm_B3A2C0` changed (n = 4 → 32x4 is not in his `uk.c`; n ≤ 64 → 32x8; n ≥ 80 → 16x16), keeping `YOLO_MR` coherent with the kernel, and time it A-B-B-A against stock at 8 threads on an idle board. Local patch only, never pushed to his repo.
