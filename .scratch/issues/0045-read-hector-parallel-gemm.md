# S4: How Héctor's GEMM runs on 8 threads: read `gemm_B3A2C0` beside our macrokernel

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/62

**Role / Label:** `needs-triage`. Sub-issue of #58 ([[0041-yolo-best-microkernel-size-parent]]). Code read, plus board runs of his C code on `jbpi`/`jbpi2` (X60, the K1's chip). Changed 2026-11-15 (user): his C code can run on the board, so we run it there directly rather than guess, simulate or emulate (parent rule).
**Status:** 📝 Open. A first read was done on 2026-11-15 (§2). The side-by-side with our code is not done.
**Primary sources:**
- https://github.com/martineh/GEMMLib_yolo @ `b292c92`:
  - `src/GEMM/gemm_blis.c:831-950` (`gemm_B3A2C0`)
  - `:950-` (`gemm_prepack_B3A2C0`)
  - `:191-203` (`UKERNEL` fringe macro)
  - `:780-825` (entry point: im2col, GEMV for n = 1)
  - `src/GEMM/gemm_blis.h:21-32`
  - `Makefile` (`-fopenmp -march=rv64gcv_zfh_zvfh`)
  - `gemm.bin` (`TH`, `TIMIN=2.0`)
- ours: the generated macrokernel in `generate_pure_blis_project.py` (macro loop emission), `templates/driver_benchmark.cpp`

---

## 1. Goal

We learn from his code. We do not run our kernels in it or copy it into our design. His code is C, so we also build it with his `Makefile` and run his `gemm.bin` on the RISC-V board, to see how his thread split performs on the X60, not only how it reads. Board runs follow the house rules (#39): check other users first; a thread count that uses core 0 follows #63's core-0 decision.

Write down exactly how Héctor splits the GEMM over threads, so S5 (#56) can copy what works and avoid what does not.

## 2. First read (2026-11-15, his commit `b292c92`)

- **Threads:** OpenMP, one `#pragma omp parallel` around the whole loop nest. The thread count comes from the caller (`TH` in `gemm.bin`).
- **Loop order:** `jc` (NC = n, so a single pass) → `pc` (KC = 256) → `ic` (MC = 256) → `jr` (NR) → `ir` (MR). This is the same B3A2C0 order as ours.
- **Split:** `loop = LOOP_IR` (loop 5). Each `ic` block of MC = 256 rows is cut into `ceil(mc / MR / threads)` microtiles per thread. With MR = 16 and 8 threads, each thread gets 2 microtiles (32 rows) per block.
- **Packing B:** every thread packs its own slice of Bc, `its_nc * NR` columns, into a shared Bc. Then a barrier.
- **Packing A:** every thread packs only its own rows of Ac, then runs the microkernel on them. There is no barrier between the A pack and the compute, because each thread reads only what it packed.
- **Barriers:** one after the B pack (per `pc`), one after each `ic` block.
- **Fringe:** a per-thread `Ctmp_th` buffer (`MR*NR`). A partial tile computes into it, and then the thread adds it to C with a plain C loop.
- **Tile choice:** fixed at compile time.
  - fp32: 16x8, or 16x4 when n == 4
  - fp16 (`NQ_FP32` not set): 32x8 and 32x4
  - n == 1 goes to `gemv_rvv_fp32` before any of this.
- **Blocking:** fixed `#define` (MC = KC = 256). No cache model.

## 2b. Decisions (grilling, 2026-11-16; agent's recommended answers, accepted by the user's standing prompt)

Facts found first: both boards have `gcc-14` (his `Makefile` sets `CC := gcc-14`) and two L2 clusters, cores 0-3 and 4-7 (`/sys/.../cache/index2/shared_cpu_list`). His `gemm.bin` pins with `taskset -c 0` for 1 thread and `0-7` for 8, sets no `OMP_PROC_BIND`, and his driver (`src/driver_gemm.c:186-199`) reports the mean time per call over a `TIMIN` loop, with the check (`TEST=T`) outside the timed loop. MC, KC and the split loop are `#define`s (`gemm_blis.h:20-32`); the driver's MC/NC/KC arguments do not reach `gemm_rvv`.

1. **His code is not edited.** A read-only clone at `b292c92` in the scratchpad, copied to the board as `~/hector_gemmlib_b292c92` (the boards share one NFS home), built there with his `Makefile` and `Makefile.inc` as they are (`gcc-14 -O3 -fopenmp`, `NQ_FP32`, `K1`): `make all install`, since `gemm.bin` preloads `install/lib/libGEMM.so`.
2. **His run command, with our cores.** The `gemm.bin` command line as it is, with `TIMIN=2.0`, `TEST=T`, `gemm_study/yolo.dat` (53 rows, 3 of them n = 1 on his GEMV), except the core list: 1 thread on core 4 (his script uses core 0; house rule 1), 8 threads on cores 0-7 (core 0 allowed for the 8-thread run only, user 2026-11-16: an exception to house rule 1, named with every 8-thread number). No `OMP_PROC_BIND`, as he runs it.
3. **Launch order 1T, 8T, 8T, 1T, 1T, 8T** (A-B-B-A-A-B, house rule 2): 3 launches each.
4. **Board, per launch** (user, 2026-11-16): read both `jbpi` and `jbpi2` before every launch; run on the idle one, the same board as the last launch when both are idle. A busy board gets a side note (user, command, CPU, core) in the logs and here. Both busy: wait 10 min and read again; never start under load.
5. **One metric for his numbers.** His driver gives the mean time per call only, so his GOPs are "by mean time"; "by median rep" does not exist for them. We do not change his driver.
6. **§3.2-3.4 are answered with the approved runs only** (1 and 8 threads): the per-shape 8-thread speedup shows where threads sit idle. An MC sweep needs his `#define`s changed, and a 4 + 4 cluster split needs 4-thread runs: both are listed as optional follow-ups, not run.
7. **A runner, `run_hector_gemm.sh`**, that syncs, builds, reads the boards, runs the launches and keeps the logs in `logs/hector_gemm_<stamp>/`; a copy goes to `HectorYolo/`.

## 3. Questions to answer in the side-by-side

1. Where in our generated macrokernel does the same `ir` split go? Our Ac/Bc are allocated by the driver: does each thread need its own Ac slice?
2. With MC = 256 and 8 threads, each thread gets only 32 rows per block, and there is a barrier per block. Is MC = 256 a deliberate choice for 8 threads, or left over from one-thread tuning? Our cache model picks mc for one core.
3. The K1 has two clusters of 4 cores, each with its own L2 (to confirm on the board from `/sys`). His split ignores that. Does a split by `jr` or by `ic`, or 4 + 4 threads per cluster, do better?
4. His B pack splits by `its_nc * NR` columns. With small n (16, 32), several threads get no columns. Does that leave threads idle at the barrier?
5. What do we need from his timing harness (`gemm.bin`, `driver_gemm.c`) to compare fairly in S6: thread pinning (`OMP_PROC_BIND`?), first-touch of buffers, `TIMIN`?

## 4. Output

A short note in this issue: the table above checked against our loop nest, his `gemm.bin` numbers from the board, the answers to §3, and a list of "copy / change / skip" items handed to S5 (#56).

## 5. Closure criteria

- [ ] Side-by-side with our macrokernel written, with `file:line` on both sides.
- [ ] His `gemm.bin` built and run on the board (1 and 8 threads at least), with board, build, `TIMIN`, threads and cores named; logs committed.
- [ ] §3 answered, with a board measurement where a question is about speed or idle threads (§3.2-3.4).
- [ ] Copy / change / skip list posted on #56.
