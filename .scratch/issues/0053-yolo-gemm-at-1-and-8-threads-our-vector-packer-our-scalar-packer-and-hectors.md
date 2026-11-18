# YOLO GEMM at 1 and 8 threads: our vector packer, our scalar packer and Héctor's

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/94

**Combined YOLO report (all findings, read first):** `.scratch/reports/yolo-multithread/report/yolo_report.pdf` ([LaTeX](../reports/yolo-multithread/report/yolo_report.tex)); index of every YOLO report, newest first: `.scratch/reports/yolo-multithread/README.md`.

**Role / Label:** `needs-triage`. Opened 2026-11-18 at the user's request. Board runs on `jbpi`.
**Status:** open. Board series done 2026-11-18 (18 of 18 launches ok, every row PASS). Report written and linked (§7). Code review done (§7.1). Waiting for the user to close; then PR.
**Report:** [PDF](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/yolo-gemm-at-1-and-8-threads-our-vector-packer-our-scalar-packer-and-hectors/.scratch/reports/issue-94/report/issue94_report.pdf) / [LaTeX](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/yolo-gemm-at-1-and-8-threads-our-vector-packer-our-scalar-packer-and-hectors/.scratch/reports/issue-94/report/issue94_report.tex), [README](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/yolo-gemm-at-1-and-8-threads-our-vector-packer-our-scalar-packer-and-hectors/.scratch/reports/issue-94/README.md); folder `.scratch/reports/issue-94/`.
**Final report (the #91 report with §1.2 redone from this rerun):** [PDF](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/yolo-gemm-at-1-and-8-threads-our-vector-packer-our-scalar-packer-and-hectors/.scratch/reports/issue-94/final/issue94_final_report.pdf) / [LaTeX](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/yolo-gemm-at-1-and-8-threads-our-vector-packer-our-scalar-packer-and-hectors/.scratch/reports/issue-94/final/issue94_final_report.tex).
**Branch:** `yolo-gemm-at-1-and-8-threads-our-vector-packer-our-scalar-packer-and-hectors`, cut from `pure-xdsl`.
**Related:** #56 [[0040-parallel-gemm-design]] (ours, vector packer), #62 [[0045-read-hector-parallel-gemm]] (Héctor's `gemm.bin`), #83 [[0050-scalar-packing-rerun-of-yolo-and-paper-sweeps]] (scalar packer).

---

## 1. Question

Rerun the #56/#62 comparison with our scalar packer added. Three setups:

1. **V**: ours, vector packer (as #56).
2. **S**: ours, scalar packer (`--scalar-pack`, works at 8 threads since #83).
3. **H**: Héctor's `gemm.bin` at `b292c92`, unchanged (as #62).

Each at 1 thread (core 4) and 8 threads (cores 0-7).

Why: #83 compared the scalar packer to Héctor only at 8 threads, with the best tile per shape (#63 setup). It never covered the #56 setup: one fixed tile, one pass, 1 and 8 threads. And #56 compared to Héctor's #62 runs from another day, not mixed in one series.

## 2. Setup (same as #56)

- Tile 24x8, the 50 YOLO26 shapes of `datasets/yolo26_k1.dat` (no n = 1 rows), board `jbpi`.
- Ours: uninstrumented build (`INSTRUMENTED=0`), reference protocol, `--tmin 2`, `--no-nc-overrides`, every row checked (`config_parallel_gemm.json`).
- 8 threads: `OMP_PROC_BIND=close OMP_PLACES=cores`, cores 0-7 (core 0 by the user's 8-thread exception, #56 §5).
- Héctor: `TIMIN=2.0`, `TEST=T`, 8T bound like ours (§2.1), his tiles (16x8; 16x4 at n = 4), his `gemm_study/yolo.dat` (53 rows; the 3 n = 1 rows go to his GEMV and are left out of the comparison, as in #56).
- All three setups mixed in one series, 3 launches per setup per thread count, the order rotated between rounds.

### 2.1 Same run settings on both sides (user, 2026-11-18)

User: when comparing with Héctor's design, use the same run settings (time per shape, cores, flags) as far as
possible; the algorithms stay each side's own. Checked flag by flag:

| setting | ours | Héctor | in this series |
|---|---|---|---|
| time per shape | reps until `--tmin 2`, mean per call, check after timing | `TIMIN=2.0`, the same loop | same |
| cores | 1T core 4, 8T cores 0-7 | the same | same |
| 8T thread binding | `OMP_PROC_BIND=close OMP_PLACES=cores` | none in #62 (only `taskset`) | **same now**: his 8T launches get the same binding (`run_hector_gemm.sh --omp-bind`, env only, his code unchanged) |
| compiler | g++ 13.2 `-O2` (generated kernels) | gcc-14 `-O3` (his Makefile) | **differs**, kept: matching it changes our build and breaks the link to #56 and every earlier run. Offered as a separate run. |

The first start (11:52) was stopped by me during C1, before any timed launch, to add the binding; it restarts from C1.

## 3. Order

Three rounds. Each round runs one block per thread count; thread counts go 1 8 8 1 1 8 over the rounds (as #56). Inside a block the setup order rotates, so each setup takes each place once:

| round | first block | second block |
|---|---|---|
| 1 | 1T: V S H | 8T: V S H |
| 2 | 8T: S H V | 1T: S H V |
| 3 | 1T: H V S | 8T: H V S |

18 launches. Before each launch the board is read; while another user's job is busy, it waits (never kills).

## 4. Correctness first

On `jbpi`, before any timing: tile 24x8, scalar packer, scalar oracle on, on `datasets/yolo_correctness_small.dat` (as #83 C1/C2):

- **C1:** 1 thread, core 4.
- **C2:** 8 threads, cores 0-7.

Every row must PASS, or the series stops before timing. Their times are not quoted.

## 5. Plan for the code (smallest change)

- `run_parallel_gemm.sh`: add `--threads 1|8` (run only that thread count) and `--scalar-pack` (passed to `compile_macro_bench.sh`).
- New `run_issue94.sh`, like `run_issue83.sh` / `run_issue63.sh`: C1, C2, then the 18 launches by calling `run_parallel_gemm.sh --launches 1` and `run_hector_gemm.sh --launches 1`. Writes the plan and order into its log folder.
- New `config_issue94_correctness.json`: 24x8 on `yolo_correctness_small.dat`, oracle on.
- Host test `test_run_issue94.py` (`--dry-run`: order and refusals).
- The vector path and Héctor's code stay as they are.

## 6. Output

`.scratch/reports/issue-94/`: logs and a report (PDF + LaTeX) with the tables of #91 §1.2:

- whole pass ms and GOPs at 1T and 8T, speedup 8T/1T;
- the by-n table;
- a column group per setup (V, S, H);
- both GOPs by mean time and by median rep (Héctor: by mean time only, his driver gives no per-rep time);
- glossary first, pgfplots horizontal bars with values, one chart per figure.

Linked from this issue and #56, #62, #83.

## 7. Results (2026-11-18)

Series `issue94_2026-11-18_115554`, jbpi, 12:00-13:09. The board was idle before every launch (no waits, no side notes).

- **Correctness:** C1 4 of 4 PASS, C2 4 of 4 PASS. Each of our 12 timed launches: 50 of 50 PASS. Each of Héctor's 6: 53 of 53 OK in his own check.
- **One pass, GOPs by mean time, mean of 3 launches.** Ours: uninstrumented build, reference protocol, `--tmin 2`. Héctor's: `TIMIN=2.0`.

| setup | 1T (core 4) ms | GOPs | 8T (cores 0-7, bound) ms | GOPs | speedup |
|---|---|---|---|---|---|
| V: ours, vector packer | 697.2 | 4.27 | 155.2 | 19.17 | 4.49x |
| S: ours, scalar packer | 733.9 | 4.05 | 160.6 | 18.52 | 4.57x |
| H: Héctor's gemm.bin | 583.4 | 5.10 | 156.6 | 18.99 | 3.73x |

- **GOPs by median rep** (ours only): V 4.27 / 20.06, S 4.05 / 19.48 at 1T / 8T.
- **Launch ranges:**
  - 1T: V 4.25-4.29, S 4.05-4.05, H 5.08-5.12.
  - 8T: V 19.14-19.18, S 18.47-18.54, H 18.75-19.27. V's range lies inside H's.
- **Earlier runs:** #56 V 4.26 / 19.55; #62 H 5.10 / 21.87, his 8T then without thread binding.
- **Without Y00** (as #91 §1.4 quotes ours against Héctor's), 8T: V 19.59, S 18.96, H 22.98 GOPs (148.4, 153.3, 126.5 ms).
  - Héctor's Y00 took 29.5-30.4 ms in each of his 8T launches; at 1T it took 15.3-15.6 ms, and at 8T in #62 (no binding) 13.7-15.3 ms.
  - So the whole-pass tie at 8T comes from Y00.
- **By n:** report §3.
  - 8T: V is highest at n = 4, 16 and 256; H at n = 8, 32, 64, 80 and 128.
  - S is below V at every n at 8T, and at every n except 256 at 1T.

### 7.1 Code review (against `pure-xdsl`)

Two reviews: standards and spec.

- **Fixed:** `run_parallel_gemm.sh` printed "Issue #56" in its plan whatever `--issue` said. It now prints the given number. Test added.
- **Checked, no change:**
  - Héctor's plan prints `TIMIN=2` where the spec says `2.0`. His driver reads the value with `atof`, so both are the same.
  - His 1T launches get `--omp-bind`. `run_hector_gemm.sh` sets the binding only when there is more than 1 thread.
- **Left as is (judgement calls):**
  - `busy_others`/`wait_idle` are copied from `run_issue83.sh`; this follows the house pattern, one script per issue.
  - The steps are `"V 8"` strings.
  - The "earlier runs" line quotes #56/#62 from committed logs. It is context for a rerun, not a new analysis.
  - The launch-range column and the per-shape appendix follow the #91 report.

### 7.2 Final report (user, 2026-11-18)

The user asked for one final report: the #91 report with its opening comparison against Héctor (§1.2, from #56/#62) replaced by this rerun.

- `final/issue94_final_report.pdf` (24 pages).
- §1.2 has the V/S/H tables (whole pass, without Y00, by n by mean time and by median rep) and the two by-n charts.
- The answer, glossary (V/S/H, thread binding), Part 4, correctness, limits (Y00) and runs list are updated.
- Parts 2 and 3 are #91's, unchanged. The #91 slides are not updated.
- #91's own report is left as committed; its issue gets a pointer here.

### 7.3 Why ours trails at 1T (#95, 2026-11-18)

- #95 found the main cause: our A packer reads A panel by panel, which costs a column jump per read. Héctor's packer reads it column by column.
- Report: [PDF](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/why-our-yolo-gemm-is-slower-than-hectors/.scratch/reports/issue-95/report/issue95_report.pdf) / [LaTeX](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/why-our-yolo-gemm-is-slower-than-hectors/.scratch/reports/issue-95/report/issue95_report.tex).
- The fix is pending in #96.
- The final report's Part 4 has a short section on it.

## 8. Closure

- [x] Driver and host test, tests passing
- [x] C1, C2 all PASS
- [x] 18 launches done, logs committed
- [x] Report written and linked (#94, #56, #62, #83)
- [x] Final report (#91 with §1.2 from this rerun), linked from #91 and #94
- [x] Code review done, findings fixed (§7.1; host suite 646 passed)
