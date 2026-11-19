# Issue #56: Make the design parallel for the GEMM test, and test it on one specific program

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/56

**Combined YOLO report (all findings, read first):** `.scratch/reports/yolo-multithread/report/yolo_report_v2.pdf` ([LaTeX](../reports/yolo-multithread/report/yolo_report_v2.tex)); index of every YOLO report, newest first: `.scratch/reports/yolo-multithread/README.md`.
**Rerun #94 (2026-11-18):** report [PDF](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issue-94/report/issue94_report.pdf) / [LaTeX](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issue-94/report/issue94_report.tex), [README](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issue-94/README.md). V (ours, vector packer), S (ours, scalar packer) and H (Héctor's `gemm.bin`) mixed in one series on jbpi, tile 24x8, 1T core 4 and 8T cores 0-7 (all bound), 3 launches each. GOPs by mean time, 1T / 8T: V 4.27 / 19.17, S 4.05 / 18.52, H 5.10 / 18.99.
**Audit #79 (2026-11-17):** brief [PDF](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issue-79/issue-79-audit-brief.pdf) / [LaTeX](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issue-79/issue-79-audit-brief.tex), reports in `.scratch/reports/issue-79/`. Engine (#56): host tests and code read PASS; all 19 tiles at 8 threads PASS on the board (950/950).

**Role / Label:** `needs-triage` — opened 2026-11-12 at the user's request. A general new direction, not a spec yet.
**Status:** ✅ Done 2026-11-16. Built (§6), board run on `jbpi`: 6 of 6 launches ok, every row PASS (§7). Report: `.scratch/reports/issue-56/`.
**Target:** to be decided (board, core count and which program).

---

## 1. Direction

Make the design run in parallel (several cores) for the GEMM test, and test it on one specific program.

## 2. Open points (to settle before a spec)

- Which specific program to test on (to be named by the user).
- How the work is split across cores, and how that fits the house rules of #39 (never core 0, saved logs, idle board).

---

## 3. Update 2026-11-15: the program is YOLO26 (Héctor)

- **Program:** the YOLO26 GEMM shapes on the K1 (Héctor, email of 13 Nov 2026). This issue is now **S5** of the YOLO parent #58 [[0041-yolo-best-microkernel-size-parent]].
- **Starting point for the split:** Héctor's `gemm_B3A2C0` (OpenMP, loop 5 `ir` split, B packed in slices by every thread, A packed per thread, two barriers). Read in S4 [[0045-read-hector-parallel-gemm]]. S4's copy / change / skip list lands here before a spec is written.
- **Built in our design.** The generated macrokernel gets the thread split. Héctor's code is only read (S4) and compared against (S6), never copied in.
- **Rule: opt-in, nothing that runs today changes.** A new `--threads N` (default 1) on `compile_macro_bench.sh` and the generator. With 1 thread the generated code is byte-identical to today (test), so every result since #39 stays valid. The search from S2 #60 runs with any thread count.
- **Still open:** the core-0 rule for 8 threads (S6 [[0046-yolo-eight-thread-sweep]] §2), and whether the cache model needs a per-thread `mc`.
- **Blocked by:** #62 (S4). Unblocked 2026-11-16: S4's list is in §4.

---

## 4. From S4 (#62, 2026-11-16): copy / change / skip

Source: [[0045-read-hector-parallel-gemm]] §6. Héctor's `gemm_B3A2C0` (unedited, `b292c92`) on the YOLO26 shapes, one pass over 53 shapes, GOPs by mean time, 3 launches per thread count per board:
- **jbpi:** 5.10 GOPs at 1 thread (core 4), 21.87 at 8 (cores 0-7), 4.29x.
- **jbpi2:** 5.26 and 23.56, 4.48x.

Small `n` scales worst: 2.9-3.8x for n ≤ 32, 4.3-5.6x for n ≥ 64. Report and figures: `.scratch/reports/issue-62/`; full engineering report: [`.scratch/reports/issue-62-report.md`](../reports/issue-62-report.md).

- **Copy:**
  - the `ir` split: each thread packs its own A rows, with no barrier between that pack and its compute;
  - the column-sliced B pack, with one barrier after it;
  - one shared Ac/Bc buffer, sliced by thread (no per-thread allocation).
- **Change:**
  - drop his barrier per `ic` block and keep one per `pc`, before Bc is packed again;
  - split B so small `n` (4-32) does not leave most threads idle at the B pack (his gives a slice to 1-4 of 8 threads there);
  - choose `mc` with the thread count in mind, instead of his fixed 256 or our one-core cache model;
  - make the pack/kernel counters (`pack_time`, `kernel_time`, `*pack_bytes`) per-thread.
- **Skip:**
  - his fixed `#define` blocking (we keep the cache model);
  - his GEMV path for n = 1;
  - 1 thread on core 0.
- **Test on the board:** pin threads (`OMP_PROC_BIND`/`OMP_PLACES`; he sets none) and try 4 + 4 threads per L2 cluster (cores 0-3 and 4-7 each share an L2).

---

## 5. Decisions (grilling, 2026-11-16; agent's recommended answers, accepted by the user's standing prompt)

1. **Opt-in.** `--threads N` (default 1) on `generate_pure_blis_project.py` and `compile_macro_bench.sh`. With 1 thread the generated code is byte-identical to today (test).
2. **OpenMP.** One `omp parallel` region per GEMM call; `-fopenmp` only when N > 1.
3. **Split.** Every thread runs the `jc`/`pc` loops. The B pack is split over (panel x k-chunk) units, so small `n` still feeds all threads; one barrier after it. Each thread packs and computes its own `mr` panels of every `mc` block (the `ir` split) in one shared Ac/Bc; no barrier per full `ic` block. One extra barrier before a short last `ic` block, whose panel split differs (else a thread overwrites Ac rows another thread still reads; the host test fails without it). One barrier before the next `pc` repacks Bc.
4. **Blocking.** The one-core cache model's `mc/nc/kc`, unchanged. On YOLO26 it gives `mc` = 300-4096 (#61 run folders), at least 12 panels of 24 rows. A per-thread `mc` is left for later.
5. **Refused with N > 1:** `--instr` (its counters are not thread-safe), `--scalar-pack`, `--pack-b` other than `row`, `--zero-pack-lines`.
6. **Cores.** N > 1 runs on cores 0..N-1 with `OMP_PROC_BIND=close OMP_PLACES=cores`, the cores of Héctor's #62 runs. The user's core-0 exception (#62) is extended to these runs; `provenance.txt` records threads and cores. The 4 + 4 per-cluster test goes to S6 (#63).
7. **Test program.** YOLO26 (`datasets/yolo26_k1.dat`), tile 24x8 (#61's best single tile). 1 thread on core 4 and 8 threads on cores 0-7, order 1T 8T 8T 1T 1T 8T (3 launches each), `--tmin 2`, idle `jbpi`, uninstrumented, reference protocol, `--no-nc-overrides`, every row checked for correctness. Both GOPs metrics, beside Héctor's #62 jbpi numbers.
8. **Output.** `.scratch/reports/issue-56/`, laid out like `.scratch/reports/issue-62/`.

## 6. Built (2026-11-16)

- `generate_pure_blis_project.py --threads N` (`render_parallel_nest`), `compile_macro_bench.sh --threads N` (also refuses `--no-ref-protocol`: its warm-up scalar-pack pass would get the vector packers), `run_parallel_gemm.sh`, `config_parallel_gemm.json`, `parallel_gemm_report.py`.
- Tests: `test_parallel_gemm.py` (byte-identical 1-thread output; 3- and 8-thread dispatch on the host with OpenMP, 20 reps per case; removing either barrier fails it), `test_parallel_gemm_report.py`. Full suite: 604 passed.
- Board run: §7.

## 7. Board run (2026-11-16, jbpi)

Series `logs/parallel_gemm_2026-11-16_193132/`. Tile 24x8 on `datasets/yolo26_k1.dat` (50 shapes), uninstrumented, reference protocol, `--tmin 2`, `--no-nc-overrides`. 1 thread on core 4, 8 threads on cores 0-7. Order 1T 8T 8T 1T 1T 8T (run folders `run_2026-11-16_193618`, `run_2026-11-16_194037`, `run_2026-11-16_194504`, `run_2026-11-16_194944`, `run_2026-11-16_195424`, `run_2026-11-16_195855`). 50 of 50 PASS in every launch; no other user on the board.

One pass over the 50 shapes:

| | 1 thread | 8 threads | speedup |
|---|---|---|---|
| ours, by mean time | 698.6 ms · 4.26 GOPs | 152.1 ms · 19.55 GOPs | 4.59x |
| ours, by median rep | 698.2 ms · 4.26 GOPs | 148.3 ms · 20.05 GOPs | 4.71x |
| Héctor (#62 jbpi, not interleaved), by mean time | 583.3 ms · 5.10 GOPs | 135.7 ms · 21.91 GOPs | 4.30x |

- We scale better at n ≤ 32 (3.6-4.6x against 2.9-3.7x; the k-split B pack), but our 1 thread is 1.4-2.2x slower there.
- n ≥ 64 is mixed: level at 80, behind at 64 and 128, ahead at 256. Our 8 threads match or beat his on 25 of 50 shapes.
- At 8 threads the median rep runs well above the mean on the m = 300/1200, n ≥ 64 shapes (e.g. Y34 49.81 against 37.11): some slow reps. Not looked into.

Report: `.scratch/reports/issue-56/README.md`, engineering report `.scratch/reports/issue-56/report/issue-56-report.md`. Next: S6 #63 is unblocked.

## Closure criteria

- [x] `--threads N`, opt-in, 1 thread byte-identical (test).
- [x] Tested on one program (YOLO26), correct on every shape, logs committed.
- [x] Report in `.scratch/reports/issue-56/`.
