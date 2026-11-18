# S4: How Héctor's GEMM runs on 8 threads: read `gemm_B3A2C0` beside our macrokernel

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/62

**Rerun #94 (2026-11-18):** report [PDF](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/yolo-gemm-at-1-and-8-threads-our-vector-packer-our-scalar-packer-and-hectors/.scratch/reports/issue-94/report/issue94_report.pdf) / [LaTeX](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/yolo-gemm-at-1-and-8-threads-our-vector-packer-our-scalar-packer-and-hectors/.scratch/reports/issue-94/report/issue94_report.tex), [README](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/yolo-gemm-at-1-and-8-threads-our-vector-packer-our-scalar-packer-and-hectors/.scratch/reports/issue-94/README.md). V (ours, vector packer), S (ours, scalar packer) and H (Héctor's `gemm.bin`) mixed in one series on jbpi, tile 24x8, 1T core 4 and 8T cores 0-7 (all bound), 3 launches each. GOPs by mean time, 1T / 8T: V 4.27 / 19.17, S 4.05 / 18.52, H 5.10 / 18.99.
**Role / Label:** `needs-triage`. Sub-issue of #58 ([[0041-yolo-best-microkernel-size-parent]]). Code read, plus board runs of his C code on `jbpi`/`jbpi2` (X60, the K1's chip). Changed 2026-11-15 (user): his C code can run on the board, so we run it there directly rather than guess, simulate or emulate (parent rule).
**Status:** ✅ Closed 2026-11-16 (merged in PR #66, with #65). His `gemm.bin` ran on `jbpi` and `jbpi2`, 3 launches per thread count on each board (§6.1). The side-by-side and the §3 answers are in §6.3-6.4. The copy / change / skip list (§6.6) is posted on #56. Report and figures: `.scratch/reports/issue-62/`.
**Report:** [`.scratch/reports/issue-62-report.md`](../reports/issue-62-report.md) (engineering report: method, changes, verification, results, discussion, per-shape appendix; audited by a second agent, findings fixed; copy in `.scratch/reports/issue-62/report/`).
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

- [x] Side-by-side with our macrokernel written, with `file:line` on both sides (§6.3).
- [x] His `gemm.bin` built and run on the board (1 and 8 threads at least), with board, build, `TIMIN`, threads and cores named; logs committed (§6.1).
- [x] §3 answered, with a board measurement where a question is about speed or idle threads (§3.2-3.4) (§6.4). §3.3's split alternatives are not measured (§2b decision 6).
- [x] Copy / change / skip list posted on #56 (§6.6).

---

## 6. Results (2026-11-16)

### 6.1 His `gemm.bin` on the board

Runner: `tests/pure_xdsl_blis_rvv/run_hector_gemm.sh` (tests: `test_run_hector_gemm.py`), report and figure: `hector_gemm_report.py`. His tree at `b292c92`, unedited, built on the board with his `Makefile` / `Makefile.inc` (`gcc-14 -O3 -fopenmp -march=rv64gcv_zfh_zvfh`, `NQ_FP32`, `K1`), his `gemm.bin` command line with `TIMIN=2.0`, `TEST=T`, `gemm_study/yolo.dat` (53 rows). 1 thread on core 4; 8 threads on cores 0-7 (**core 0 included**: the user's exception, 8-thread runs only). No `OMP_PROC_BIND`, as he runs it. Order 1T 8T 8T 1T 1T 8T, 3 launches each. His check printed `OK` on every row of every launch. His driver gives the mean time per call only, so every number below is **GOPs by mean time**; there is no "by median rep".

Two runner bugs were fixed before the series that counts (both ours, not his code):
- `make -j4` let `gemm.x` link against a half-written `libGEMM.so` (`undefined reference to gemm_rvv`). Now `make clean && make && make install`, serial, as his README says.
- `gemm.x` NEEDs `libGEMM.so` by name, and his `LD_PRELOAD` alone does not satisfy the loader (exit 127). The runner adds `LD_LIBRARY_PATH=install/lib`, the same file he preloads. His `gemm.bin` presumably runs in a shell that already has that path.

One pass over the dataset (each row once, mean time per call averaged over the 3 launches):

| board | series | 1 thread (core 4) | 8 threads (cores 0-7) | speedup | side notes |
|---|---|---|---|---|---|
| jbpi | `logs/hector_gemm_2026-11-16_111457/` | 583.5 ms, 5.10 GOPs | 136.0 ms, 21.87 GOPs | 4.29x | none |
| jbpi2 | `logs/hector_gemm_2026-11-16_113052/` | 565.2 ms, 5.26 GOPs | 126.2 ms, 23.56 GOPs | 4.48x | none |

Per shape, both boards: `.scratch/reports/issue-62/` (`README.md`, and a report `.md` and figure per series). The two boards are reported side by side, never averaged.

### 6.2 Speedup by `n` (one pass over the rows with that `n`, GOPs)

| n | rows | threads with a B slice | jbpi 1T | jbpi 8T | jbpi speedup | jbpi2 1T | jbpi2 8T | jbpi2 speedup |
|---|---|---|---|---|---|---|---|---|
| 1 (his GEMV) | 3 | - | 2.43 | 6.47 | 2.66x | 2.42 | 6.78 | 2.80x |
| 4 (16x4) | 3 | 1 | 2.44 | 8.14 | 3.34x | 2.50 | 7.58 | 3.03x |
| 8 | 1 | 1 | 2.40 | 6.95 | 2.90x | 2.45 | 7.08 | 2.90x |
| 16 | 10 | 2 | 2.83 | 9.05 | 3.20x | 3.02 | 11.56 | 3.83x |
| 32 | 6 | 4 | 5.12 | 18.84 | 3.68x | 5.18 | 19.35 | 3.74x |
| 64 | 10 | 8 | 5.53 | 27.53 | 4.98x | 5.76 | 27.59 | 4.79x |
| 80 | 6 | 5 | 7.61 | 32.56 | 4.28x | 7.58 | 32.24 | 4.25x |
| 128 | 9 | 8 | 6.14 | 33.39 | 5.43x | 6.26 | 34.71 | 5.55x |
| 256 | 5 | 8 | 5.99 | 29.75 | 4.97x | 6.04 | 29.67 | 4.91x |

Single shapes (1T → 8T GOPs, speedup):

| shape | jbpi | jbpi2 |
|---|---|---|
| 76800x16x27 (300 `ic` blocks) | 4.29 → 4.61, 1.07x | 4.51 → 9.79, 2.17x |
| 1200x16x144 | 2.75 → 25.14, 9.14x | 2.70 → 26.14, 9.68x |
| 300x1x9 (his GEMV) | 2.13 → 1.32, 0.62x | 2.13 → 1.37, 0.64x |

76800x16x27 differs most between the boards; these runs do not show why. 1200x16x144 goes above 8x because its 1-thread run is slow (2.7 GOPs); the cause was not measured.

### 6.3 Side by side: his loop nest and ours

His: `src/GEMM/gemm_blis.c` @ `b292c92`. Ours: `tests/pure_xdsl_blis_rvv/generate_pure_blis_project.py` (the emitted `gemm_blis_generic`) and `templates/driver_benchmark.cpp`.

| step | his (`gemm_blis.c`) | ours |
|---|---|---|
| threads | `#pragma omp parallel` around the whole nest, `:873` | none: no OpenMP anywhere in the generator or driver |
| `jc` | `:881`, NC = n (one pass) | `:553`, `nc` from the cache model (`driver_benchmark.cpp:893`, 1536 for 16x8) |
| `pc` | `:888`, KC = 256 (`gemm_blis.h:25`) | `:556`, `kc` from the cache model (256 for 16x8) |
| B pack | every thread packs `its_nc * NR` columns into a shared Bc, `:883`, `:896`; barrier `:902` | one call packs all of `cur_nc`, `:644` |
| `ic` | `:904`, MC = 256 | `:648`, `mc` from the cache model (384 for 16x8) |
| A pack | every thread packs its own `its_mc * MR` rows, `:907`, `:919`; no barrier after it | one call packs all of `cur_mc`, `:654` |
| `jr` / `ir` | `:924` / `:926`, split on `ir` (`LOOP_IR`) from `ir_init = th_id * its_mc * MR` | `:664` / `:667`, all tiles |
| fringe | per-thread `Ctmp_th` (`MR*NR`), `UKERNEL` macro `:191-201`, plain C add-back | stack `tmp_C` per tile (pinned mode), `:623`, copy back |
| barrier per `ic` | `:937` | - |
| buffers | one `malloc` of `BUFFER_SIZE` floats on first call, Bc = Ac + MC*KC, `:783`, `:789` | driver allocates Ac, Bc per shape, zeroed, `driver_benchmark.cpp:928-929` |
| counters | - | `pack_time`, `kernel_time`, `*pack_bytes` accumulate through shared pointers, `:645`, `:655` |
| `n = 1` | `gemv_rvv_fp32`, `:821` | runs the GEMM path |

### 6.4 Answers to §3

1. **Where the `ir` split goes in ours, and Ac.** Inside `gemm_blis_generic`: the B pack at `:644` is split by column slices with a barrier after it, the A pack at `:654` by row slices, and `ir` at `:667` starts at the thread's first row. One Ac buffer is enough, as in his code: each thread packs and reads only its own slice (`Ac + kc * its_mc * MR * th_id`), so no per-thread Ac allocation is needed. Our fringe buffer is already per-thread (stack, `:623`). The counters at `:645`/`:655` and the timers are shared and would race: per-thread, or thread 0 only.
2. **MC = 256 with 8 threads.** It is a fixed `#define` (`gemm_blis.h:23`), not tied to the thread count: with MR = 16 each thread gets 2 microtiles (32 rows) per block, and there is a barrier per block (`:937`). Where blocks are many and the work per block is small, 8 threads barely help: 76800x16x27 (300 blocks, k = 27) gets 1.07x on jbpi and 2.17x on jbpi2. The runs do not separate the barrier from the other costs. A partial last block also idles threads: m = 300 leaves 44 rows, so 3 of 8 threads work. The barrier at `:937` is not needed for correctness of Ac (each thread reads only what it packed); only one barrier per `pc` is needed before Bc is packed again. Our cache model's `mc` (384 for 16x8) is for one core and does not know the thread count.
3. **Two L2 clusters.** Confirmed on both boards: L2 shared by cores 0-3 and 4-7 (`/sys/devices/system/cpu/cpu*/cache/index2/shared_cpu_list`). His split ignores them, and no `OMP_PROC_BIND` is set. Not measured here: a `jr` split, an `ic` split or 4 + 4 threads per cluster need his `#define`s changed or 4-thread runs (§2b decision 6); listed as optional follow-ups.
4. **Idle threads at the B pack.** Yes for small `n`: `its_nc = ceil(n / NR / 8)`, so n = 4, 8 give one thread a B slice, n = 16 two, n = 32 four, n = 80 five. The speedup by `n` (§6.2) follows on both boards: 2.9-3.8x for n ≤ 32, 4.3-5.6x for n ≥ 64. Part of that is also the A side and the per-block barrier, which the runs do not separate.
5. **His harness, for a fair S6 comparison.** Mean time per call over a `TIMIN` loop (`driver_gemm.c:186-199`), no warm-up, check outside the timed loop (`:201-202`): the same shape as our reference protocol. No thread pinning (`OMP_PROC_BIND` unset; `taskset` only limits the set); his `gemm.bin` puts 1 thread on core 0. Buffers: one `malloc` on the first call, reused (`gemm_blis.c:783`), so they are warm after the first rep. GOPs from `2mnk`, the same as ours.

### 6.5 Tests

Full test suite on this branch (`dc984ccb`): **583 passed, 0 failed, 0 skipped**, 15 min 38 s; the two board-side test files ran on the board. Report: `.scratch/reports/issue-62/full_test_suite_2026-11-16.md`.

### 6.6 Copy / change / skip for S5 (#56)

- **Copy:** the `ir` split with each thread packing its own A rows and no barrier between its A pack and its compute; the column-sliced B pack with one barrier after it; one shared Ac/Bc buffer, sliced by thread.
- **Change:** drop the barrier per `ic` block (keep one per `pc`, before Bc is packed again); split B so small `n` (4-32) does not leave most threads idle at the B pack; give `mc` a thread-aware choice instead of a fixed 256; make the pack/kernel counters per-thread.
- **Skip:** the fixed `#define` blocking (we keep the cache model); his GEMV path (out of scope for the GEMM design); running 1 thread on core 0.
- **Test:** pin threads (`OMP_PROC_BIND`/`OMP_PLACES`) and try 4 + 4 per L2 cluster, measured on the board.
