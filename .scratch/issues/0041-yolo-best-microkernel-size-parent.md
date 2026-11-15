# Parent: best microkernel size for the YOLO26 GEMM shapes on the K1, one core first, then 8 threads (Héctor's request)

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/58

**Role / Label:** `needs-triage`. Parent issue, opened 2026-11-15 for review.
**Status:** 📝 Open. Nothing built or measured.
**Target:** SpacemiT X60, 8 cores, VLEN 256. That is the K1 chip in Héctor's OrangePi, and the same chip as our `jbpi`/`jbpi2` (`model name: Spacemit(R) X60`, 8 cores, in every `provenance.txt`).
**Where code runs (user, 2026-11-15):** when C code can run on the RISC-V board (`jbpi`/`jbpi2`), run it there directly. No simulation, emulation (e.g. qemu) or host build as a stand-in: the board is right there, and we want to see how the code performs on the RISC-V machine. This is not a rule to force every step onto the board; Python tools (generator, reports, tests) stay on the host. Board runs follow the house rules (#39) and the check for other users.
**Shapes:** the 53 YOLO26 GEMM shapes from Héctor (m, n, k after im2col): `HectorYolo/yolo26_k1_matrix_sizes.csv`. They match `gemm_study/yolo.dat` in his repo row for row.
**Primary sources:**
- `HectorYolo/Re: Optimizaciones Yolo sobre RISC-V.pdf`: the email thread, 10-13 Nov 2026
- https://github.com/martineh/GEMMLib_yolo (commit `b292c92`, 13 Nov 2026): `src/GEMM/gemm_blis.c`, routine `gemm_B3A2C0`
- `tests/pure_xdsl_blis_rvv/generate_pure_blis_project.py`, `config.py`, `templates/driver_benchmark.cpp`
- #56 (`.scratch/issues/0040-parallel-gemm-design.md`): parallel GEMM, opened 2026-11-12, waiting on "which program"

---

## 1. The request

Héctor (13 Nov): *"Jie, comentamos ayer que sería interesante que observaras los tamaños de micro-kernels que mejor rinden para la versión paralela de la GEMM con 8 hilos."*

He wants the microkernel sizes (MR x NR) that run fastest on the YOLO GEMM shapes, in the **parallel GEMM with 8 threads**. Adrián wants to know if a size other than the two Héctor uses today would help.

**Plan: grow our design (the xDSL Project Generator) with two new features, and answer Héctor with it:**
1. **Microkernel size search** (S2 #60): the design times a list of tiles on every shape and reports the best per shape.
2. **Parallel GEMM** (S5 #56): the design's macrokernel runs on several threads.

Both are built **in our design and on our code**. Héctor's GEMMLib is used only:
- as something to read, to learn how he splits the loop over threads (S4 #62)
- as the program we compare against on the same board (S6 #63)

We do not run our kernels inside his GEMM, and we do not copy his GEMM into our design.

**Rule for both features: nothing that runs today changes.** Each one is opt-in. With the feature off (one tile per row, one thread), the generated code is byte-identical to today, so every result since #39 stays valid. A test checks this for each feature.

## 2. Where each side stands today

| | Héctor's GEMMLib | Our xDSL design |
|---|---|---|
| Microkernel | hand-written: fp32 16x8 (n > 4), 16x4 (n = 4); fp16 32x8 and 32x4 in progress | generated for any MR x NR |
| n = 1 | GEMV (`gemv_rvv_fp32`) | no GEMV: runs the GEMM with a fringe tile |
| Blocking | fixed: MC=256, KC=256, NC=n | per shape, from the cache model (`get_optim_mc_nc_kc`) |
| Threads | OpenMP, loop 5 (`ir`) split across threads | **one thread only** |
| Size search | none: sizes are fixed by `#define` | **partly**: see §3 |
| Timing | `gemm.bin`, `TIMIN=2.0`, `TH` threads | reference protocol, `--tmin`, pinned to one core |

## 3. Can our design search for the best size today?

**Partly.**
- **Search mode exists.** A `config.json` row with fewer than 8 fields (no `iterations`) turns it on (`config.py:54`). The generator then builds every tile from 1x1 up to the largest MR x NR in the row (`generate_pure_blis_project.py:330-445`). The driver times every tile on every shape and prints the fastest (`templates/driver_benchmark.cpp:1335-1350`, "best variant").
- **Problems with using it as-is:**
  - It builds every tile, not a chosen list. A maximum of 32x16 means 512 tiles x 2 beta modes, and every one runs on every shape.
  - It ranks by kernel GFLOPS (`ker_gflops`), not by the whole-call time we quote.
  - It has not run on the board in recent work. Every recent sweep (#32, #48, #51) used one pinned row per tile, and only `test_config.py` touches `is_pinned = False`. Whether it still builds with the current driver (`--pack-b`, `--zero-pack-lines`, per-rep log) is not checked.
- **What works today:** one pinned row per tile (as in `config_issue48_sweep.json`, 25 tiles). Each tile then runs every shape of the dataset. Nothing picks the best tile per shape across those rows.
- **So:** S2 #60 adds a proper search to our design, built on the pinned path so a tile times the same alone or in a search.

## 4. Sub-issues

| # | Sub-issue | Kind | Needs the board | Blocked by |
|---|---|---|---|---|
| S1 #59 | YOLO shape set: dataset file and candidate tile list ([[0042-yolo-shape-set]]) | task | no | none |
| S2 #60 | **Our design:** add microkernel size search (list of tiles → best tile per shape) + report ([[0043-yolo-size-search-tooling]]) | spec + build | for the correctness check | S1 |
| S3 #61 | One-core sweep of the candidate tiles on the YOLO shapes ([[0044-yolo-one-core-sweep]]) | board run | yes | S1, S2 |
| S4 #62 | Read Héctor's parallel `gemm_B3A2C0` beside our macrokernel, and run his `gemm.bin` on the board ([[0045-read-hector-parallel-gemm]]) | code read + board run | yes, his C code | none |
| S5 #56 | **Our design:** make the generated GEMM parallel, opt-in `--threads N`, default 1 ([[0040-parallel-gemm-design]]) | spec + build | yes, for the check | S4 |
| S6 #63 | 8-thread sweep, and our GEMM against Héctor's GEMMLib on his shapes ([[0046-yolo-eight-thread-sweep]]) | board run | yes | S3, S5 |

S1, S2 and S4 can start now and do not depend on each other except S2 on S1.

**Status 2026-11-16:** S1 #59 and S2 #60 closed. S3 #61 run on `jbpi` (uninstrumented, reference protocol, `--tmin 2`, core 4, 3 launches): best single tile **24x8**, 4.26 GOPs by mean time / 4.25 by median rep over the 50 shapes, against 3.79 / 3.79 for 16x8 and 3.44 / 3.44 for 16x4. Per n and the report: [[0044-yolo-one-core-sweep]] §5a, `tests/pure_xdsl_blis_rvv/logs/yolo_sweep_2026-11-15_205445/tile_search_report.md`.

## 5. Out of scope for now (open questions)

1. **fp16.** Héctor is moving the product to fp16 (MR=32). Our generator is fp32/int8 only. Should fp16 tiles be a follow-up issue?
2. **GEMV for n = 1** (3 shapes, about 0.01% of the FLOPs). These shapes stay on Héctor's GEMV. We skip them.
3. **Packing inside im2col.** Héctor's phase 3 packs A during im2col (`gemm_prepack_B3A2C0`). Our GEMM always packs A itself. Our numbers therefore include an A pack that his pre-packed path skips, and every comparison must say so.
4. **Call counts.** The CSV lists 53 *distinct* shapes. It does not say how often each one runs per image. Ask Héctor for the count per shape, or a time per shape from his profile, before ranking shapes by weight.
5. **Core 0.** 8 threads on an 8-core K1 must use core 0, which house rule #1 (issue #39) forbids. S5 and S6 must settle this (§S6).

## 6. Closure criteria

- [ ] S1-S6 closed.
- [ ] One table, per YOLO shape: best tile on one core and on 8 threads, with GOPs by mean time and by median rep, beside Héctor's 16x8/16x4.
- [ ] The answer sent to Héctor and Adrián: which sizes to use, by shape group, with the board, build, protocol and cores named.
- [ ] Every number from a committed log.
