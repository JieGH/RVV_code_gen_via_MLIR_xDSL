# YOLO GEMM rerun: best single tile per thread count (1T 24x8, 8T 16x12) against Héctor's unchanged GEMM, all 53 layers

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/109

**Combined YOLO report (all findings, read first):** `.scratch/reports/yolo-multithread/report/yolo_report_v2.pdf`; index: `.scratch/reports/yolo-multithread/README.md`.

**Role / Label:** `needs-triage`. Opened 2026-11-18 at the user's request. Board runs on `jbpi`.
**Status:** closed 2026-11-19. Re-search (§6b): 1T 24x8, 8T 32x7 (tie with 16x12, mean winner). Head-to-head series `issue94_2026-11-19_093036` on `jbpi`: 18 of 18 steps ok. Results in report v2 §1.2–§1.3: [PDF](../reports/yolo-multithread/report/yolo_report_v2.pdf), [LaTeX](../reports/yolo-multithread/report/yolo_report_v2.tex).
**Branch:** `yolo-gemm-rerun-best-single-tile-per-thread-count-against-hectors-unchanged-gemm`, cut from `pure-xdsl`.
**Supersedes:** #108 [[0056-time-the-n-1-yolo-layers-ours-against-hectors-gemv]] (stopped by the user, 2026-11-18 23:36; §8).
**Replaces in the report:** §1.2 of the combined report, which today comes from #94 [[0053-yolo-gemm-at-1-and-8-threads-our-vector-packer-our-scalar-packer-and-hectors]].
**Related:** #58 (parent, Héctor's request), #61 (1T tile sweep), #63 (8T tile sweep), #56 (our parallel GEMM), #62 (Héctor's GEMM), #79 (audit), #82 (his GEMM with other tiles, still open).

---

## 1. Why: the flaw in §1.2 today

§1.2 compares our GEMM with Héctor's at 1 thread (1T) and 8 threads (8T). Ours runs **tile 24x8 at both thread counts**. But 24x8 is the best single tile at **1T only**:

| | best single tile | ms per pass (50 layers) | 24x8 rank | 24x8 ms per pass |
|---|---|---|---|---|
| 1T (#61) | 24x8 | 698.6 | 1st of 19 | 698.6 |
| 8T (#63) | **16x12** | **145.6** | **10th of 19** | 158.7 |

How it happened: #61 (1T sweep) finished first. #56 built the parallel GEMM and needed one tile before the 8T sweep existed, so it took 24x8. #94 kept 24x8 to stay comparable with #56. So the 8T column of §1.2 runs ours at a tile picked for 1T, about 9% slower than our own 8T best single tile.

"Best single tile" = the one tile with the shortest pass: each of our 19 tiles runs all 50 layers; per tile, add up the mean time over the 50 layers; shortest sum wins. Charts: combined report §1.5 (Figures 1 and 2).

## 1b. Recheck of the best-tile claim (2026-11-19)

Recomputed from the raw CSVs of the 6 sweep run folders (#61: `run_2026-11-15_220501`, `_231549`, `run_2026-11-16_002638`; #63: `run_2026-11-16_222551`, `_234245`, `run_2026-11-17_005337`) with a separate script, not `search_report.py`. Pass = sum over the 50 shapes of the 3-launch mean of `RVV_Time_ms`.

- **1T: 24x8 holds.** 698.9 ms against 703.3 for 24x9; 24x8 wins each of the 3 launches.
- **8T: 16x12 is not a clear winner.** 16x12 145.7, 32x7 145.9, 16x14 146.2, 16x10 146.9, 16x15 147.7, 32x6 148.4, 16x8 149.8 ms: 7 tiles within 2.8%. 32x7 wins launches 2 and 3; 16x12 wins launch 1. Gaps under 3% are inside the noise (house rules #39).
- So the user asked for a re-search before the series (§6b).

## 2. The rule (user, 2026-11-18)

**One best tile per condition.** Each thread count runs ours at the one tile that won at that thread count. Héctor's design runs unchanged.

- No tile per layer: one tile for all layers at a given thread count.
- No change to Héctor's code, tiles, blocking or flags.

## 3. Setups

| setup | 1T (core 4) | 8T (cores 0–7) |
|---|---|---|
| **V**: ours, vector packer | 24x8 | 32x7 (§6b) |
| **S**: ours, scalar packer (`--scalar-pack`) | 24x8 | 32x7 (§6b) |
| **H**: Héctor's `gemm.bin` at `b292c92`, unchanged | his tiles: 16x8 (n > 4), 16x4 (n = 4), GEMV (n = 1) | the same |

S is in by default, as in #94 (decision §10.1).

## 4. Shapes: all 53 YOLO26 layers

- The 50 layers of `datasets/yolo26_k1.dat` (n ≥ 4), as #94.
- Plus the 3 n = 1 layers of `datasets/yolo26_k1_n1.dat` (added in #108): Y06 4800×1×9, Y21 1200×1×9, Y35 300×1×9.
- Ours runs n = 1 as a GEMM with an n-edge tile (our design has no GEMV). Héctor runs n = 1 on his GEMV. Both are each design's own way; the rule leaves them as they are.
- Héctor's `gemm.bin` always runs all 53 rows of his `gemm_study/yolo.dat`; his rows match ours by layer ID and m, n, k (as in #94's `make_data.py`).
- Report: one pass over the 50 layers (comparable with #94) and one pass over all 53.

## 5. Run settings (the same as #94)

- Board `jbpi`. Idle-board check before every launch; wait 10 min while another user's job is busy. Never kill other users' work.
- Ours: uninstrumented build (`INSTRUMENTED=0`), reference protocol, `--tmin 2`, `--no-nc-overrides`, every row checked against the scalar oracle in every timed launch.
- Héctor: `TIMIN=2.0`, `TEST=T`, his own check (rows marked OK).
- 1T on core 4. 8T on cores 0–7 (core 0 by the user's 8-thread exception, #56 §5) with `OMP_PROC_BIND=close OMP_PLACES=cores` on both sides.
- Compilers stay as they are: ours g++ 13.2 `-O2`, his gcc-14 `-O3` (noted in the report, as in #94).
- GOPs by mean time only (user, 2026-11-18).

## 6. Order

1. **Correctness first** (any FAIL stops everything before timing):
   - C1: tile 24x8, 1T core 4, `--scalar-pack`, oracle on, `--tmin 0.2`.
   - C2: tile 32x7, 8T cores 0–7, `--scalar-pack`, oracle on, `--tmin 0.2`.
   - Shapes: the #83/#94 small YOLO-like set (`yolo_correctness_small.dat`) **plus the 3 n = 1 rows**. 16x12 has never run an n = 1 shape; 24x8 has (#108, PASS at 1T and 8T).
2. **18 timed launches**, one series, as #94: 3 rounds; each round a 1T block and an 8T block (1 8, then 8 1, then 1 8); setup order rotated per round (V S H, S H V, H V S). 3 launches per setup per thread count.
3. Héctor's build: by his first launch, then reused (`--no-build`).

Time: #94 took 69 min for 18 launches over 50 shapes; the 3 n = 1 rows add 3 short rows per launch. Expect about 75 min.

## 6b. Re-search of the best single tile (user, 2026-11-19; before §6)

- 1T: the top 4 of #61 (24x8, 24x9, 32x7, 24x6), 4 launches (each tile starts one launch).
- 8T: the top 7 of #63 + 24x8, 6 launches.
- All 53 layers (`datasets/yolo26_k1_53.dat`), every row checked by the oracle, `run_yolo_sweep.sh --tmin 2 --issue 109`, `jbpi`, idle board, same settings as #61/#63.
- Pick: the shortest pass (sum of the launch-mean times over 53 layers). If the top tiles are still within the launch range, the report calls them a tie and uses the mean winner.
- The winners go into the §6 series in place of 24x8 / 16x12.

### §6b results (2026-11-19, `jbpi`)

Series `yolo_sweep_2026-11-19_002730` (1T, 4 launches, 00:27–01:42) and `yolo_sweep_2026-11-19_014357` (8T, 6 launches, 01:43–05:45); copies in `.scratch/reports/issue-109/`. Every launch exit 0, all 53 rows PASS in every launch. Uninstrumented, reference protocol, `--tmin 2`, `--no-nc-overrides`, vector packer. (An earlier 1T try, `yolo_sweep_2026-11-19_000802`, died partway through launch 1 when the session ended; not used.)

Pass = sum over the 53 layers of `RVV_Time_ms`; mean and range over the launches.

| threads | tile | ms per pass (mean) | range over launches | launches won |
|---|---|---|---|---|
| 1T | **24x8** | **697.7** | 695.6–700.2 | 4 of 4 |
| 1T | 24x9 | 708.2 | 705.9–709.5 | 0 |
| 1T | 32x7 | 730.1 | 727.9–732.3 | 0 |
| 1T | 24x6 | 730.2 | 728.4–732.0 | 0 |
| 8T | **32x7** | **140.7** | 139.9–141.4 | 5 of 6 |
| 8T | 16x12 | 141.6 | 140.3–142.8 | 1 |
| 8T | 16x10 | 142.4 | 141.5–143.7 | 0 |
| 8T | 32x6 | 142.6 | 142.0–143.7 | 0 |
| 8T | 16x8 | 142.6 | 141.2–143.8 | 0 |
| 8T | 16x15 | 143.1 | 142.1–144.4 | 0 |
| 8T | 16x14 | 143.7 | 142.3–144.8 | 0 |
| 8T | 24x8 | 156.2 | 155.3–157.4 | 0 |

- **1T: 24x8**, a clear winner whose range does not overlap 24x9's.
- **8T: 32x7** has the shortest mean pass and wins 5 of 6 launches. Its range overlaps 16x12's, so by the §6b rule it is a tie and the mean winner is used: **32x7**. The top 7 tiles sit within 2.1%. 24x8 is about 11% slower than 32x7 at 8T.

## 7. Code changes (smallest change)

Build on the #108 changes (`run_parallel_gemm.sh --config`, `run_issue94.sh --config/--issue`), which are uncommitted on the #108 branch and move to this branch.

1. **Configs** (new):
   - `config_yolo53_24x8.json`: tile 24x8, dataset `yolo26_k1_53.dat`, every row checked.
   - `config_yolo53_32x7.json`: the same with tile 32x7 (§6b winner).
   - `config_yolo53_correctness_24x8.json`, `config_yolo53_correctness_32x7.json`: `yolo_correctness_small.dat` + `yolo26_k1_n1.dat`, oracle on, 5 iterations.
2. **One 53-row dataset** `datasets/yolo26_k1_53.dat` (the 50 + the 3 n = 1 rows, Héctor's layer order) instead of two datasets per config, so `run_parallel_gemm.sh` and `run_yolo_sweep.sh` count 53 shapes without a change. `run_parallel_gemm.sh` prints the config's tile in its plan (was fixed text "24x8").
3. **`run_issue94.sh`**: one config per thread count: `--config-1t FILE` and `--config-8t FILE` (ours only), and the matching correctness configs for C1 and C2 (`--check-1t`, `--check-8t`). Defaults stay as today, so `run_issue94.sh` with no new flags still runs #94 exactly.
4. **Host test** (`test_run_issue94.py`): the 1T steps get the 1T config, the 8T steps the 8T config, Héctor's steps get neither; C1/C2 use their own configs; the shape count sums both datasets.
5. **Report data**: `issue-94/report/make_data.py` reads the new series as it read #94 (already works for a 3-shape series after the #108 Y00 guard). Speedup column relabelled (§9).

No change to the generator, the kernels, the driver or Héctor's code.

## 8. What happens to #108

- Stopped by the user on 2026-11-18 at 23:36, during step 15 of 18 (S 1T, mid-build). Steps 1–14 finished, exit 0, every row of ours PASS.
- Its 8T rows ran ours at 24x8, which breaks the rule in §2. Its logs stay in `.scratch/reports/issue-108/` and `tests/pure_xdsl_blis_rvv/logs/`; nothing from it goes into the report.
- Kept from #108: the dataset `yolo26_k1_n1.dat`, the `--config`/`--issue` runner flags and their test, and the knowledge that 24x8 handles n = 1 (PASS).
- #108 gets closed as superseded by this issue once this one is opened.

## 9. Report changes (combined YOLO report)

- **§1.2 rebuilt from this series.** Opening: "Ours at the best single tile of each thread count: 24x8 at 1T, 16x12 at 8T (§1.5). Héctor's code unchanged."
- Tables: one pass over 50 layers and over 53 layers; without Y00 (as now); by n, now with an n = 1 row.
- Per-layer charts: all 53 layers in Héctor's layer order; Y06, Y21, Y35 no longer empty. Bar labels show the tile used (24x8 at 1T, 16x12 at 8T; Héctor's 16x8 / 16x4 / GEMV).
- **Speedup 8T over 1T** for ours now changes tile between the columns (24x8 → 16x12). Label it "8T (16x12) over 1T (24x8)" and say so in the note.
- **#94 stays** in the report as the earlier fixed-24x8 run (one short paragraph and its pass table, clearly labelled), so the effect of the tile choice at 8T is visible.
- §1.1 table: the n = 1 row says "timed (§1.2)" instead of "not timed".
- Glossary: add "best single tile" (the rule in §1).
- README index and every linked issue file get the new report link (house rule).

## 10. Decisions for the user

1. **Keep S (scalar packer)?** **Yes** (user, 2026-11-19): 18 launches, about 75 min.
2. **Re-check the best single tile?** **Yes, focused re-search** (user, 2026-11-19), §6b, after the recheck in §1b.
3. **Report file:** **new `yolo_report_v2.{tex,pdf}`**; the old one renamed `yolo_report_v1_old.{tex,pdf}` with an "old, superseded by v2" banner (user, 2026-11-19). v2 reworks the best-tile part and checks that the rest of the report agrees with itself.

## 11. Output

- Logs: one folder per launch, auto-copied to `.scratch/reports/issue-<N>/` (house rule); series plan and order in `issue94_<stamp>/` (`plan.txt`, `steps.tsv`).
- Report data: `.scratch/reports/issue-<N>/report/data/` via `issue-94/report/make_data.py <series> --out`.
- Combined report rebuilt (`make_data.py`, then `tectonic -X compile yolo_report_v2.tex`); PDF + LaTeX links here and on GitHub.

## 12. House rules that apply

- No `pytest` or `--dry-run` while the series runs (they rewrite `pj_gen/`); test before, restore `generated/` after.
- Quote only committed logs; name build, protocol, cores and tile with every number.
- Wait on the series PID with `kill -0`, not `pgrep`.
- Board busy → wait, never kill another user's job.

## 13. Checklist

- [x] Decisions §10 confirmed by the user
- [x] Best-tile claim rechecked (§1b)
- [x] Re-search 1T and 8T (§6b): 1T 24x8, 8T 32x7 (tie with 16x12, mean winner)
- [x] Branch cut; #108 changes carried over
- [x] Configs (§7.1), runner changes (§7.2–7.3), host test (§7.4); all host tests pass
- [x] Dry run shows 1T → 24x8, 8T → 32x7 (after §6b), C1/C2 on their own configs, 53 shapes
- [x] Board idle; C1 and C2 all PASS
- [x] 18 of 18 launches ok, every row of ours PASS, Héctor's 53 of 53 OK
- [x] Report §1.2 rebuilt (§9); PDF compiled; links in this issue and on GitHub
- [x] #108 closed as superseded
- [x] Results committed; PR issue branch → `pure-xdsl`, then `pure-xdsl` → `main` after close
