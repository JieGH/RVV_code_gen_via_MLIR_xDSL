# Stale crontab entries call cron_bench_run.sh at its old path

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/140

**Role / Label:** `needs-triage`. Drafted 2026-11-21 at the user's request, after #137 moved `cron_bench_run.sh` into `runners/`.
**Branch:** stale-crontab-entries-old-cron-path, cut from pure-xdsl
**Status:** closed 2026-11-21.

---

## 1. Problem

The user's crontab on this host still holds the four entries of the 19-20 Sep 2026 overnight sweep (#26):

```
0 23 19 9 *              .../tests/pure_xdsl_blis_rvv/cron_bench_run.sh jbpi
30 23 19 9 *             .../tests/pure_xdsl_blis_rvv/cron_bench_run.sh jbpi2
0 0,1,2,3,4,5,6 20 9 *   .../tests/pure_xdsl_blis_rvv/cron_bench_run.sh jbpi
30 0,1,2,3,4,5,6 20 9 *  .../tests/pure_xdsl_blis_rvv/cron_bench_run.sh jbpi2
```

- The sweep they scheduled is long finished.
- Cron has no year field, so they fire again on 19-20 Sep 2027: 16 unattended benchmark runs on jbpi and jbpi2.
- Since #137 the script lives at `tests/pure_xdsl_blis_rvv/runners/cron_bench_run.sh`, so each entry would now fail with "not found". Cron mails the error, or drops it if no mail is set up.

## 2. What to do

Remove the four entries and their comment line: they belong to a finished one-off sweep, and today's runners schedule themselves (`--start-at`, `--after-pid`). (The other option, pointing them at `runners/cron_bench_run.sh`, only matters if the same sweep is wanted next September.)

The crontab sits outside the repo. When drafted, the fix was left to the user; the user then asked for it (`/implement issue 140`).

## 3. Out of scope

- Changing `cron_bench_run.sh` itself.

## 4. Closure criteria

- [x] `crontab -l` has no entry that calls `cron_bench_run.sh` at its old path (`tests/pure_xdsl_blis_rvv/cron_bench_run.sh`).

## 5. Result (2026-11-21)

- Took the recommended option: removed the four entries and their comment line. They were the whole crontab, so it is now empty.
- Backup before the change: `.scratch/reports/issue-140/crontab.before`. To undo: `crontab .scratch/reports/issue-140/crontab.before`.
- Check: `crontab -l | grep cron_bench_run.sh` finds nothing.
- Board test of the moved script: `runners/cron_bench_run.sh jbpi`, started as cron would (`env -i`, `PATH=/usr/bin:/bin`, from `/`), took the lock, timed all 10 BERT and Square layers of `config.json` in 8 minutes and exited 0 (`logs/cron_jbpi_20261121_204041.log`; not a number to quote: the board was not checked idle for timing).
- 0071's "left for the user" note now points here.

### Added at the user's request: README example commands

- `README.md`: "Run a model" (BERT B1-B5, Square S1-S5) and "Feature examples" (one command per data type, packing method, forced or swept tile, threads).
- New `configs/benchmark/config_bert.json`, `configs/benchmark/config_square.json`, `datasets/square_s1_s5.dat`.
- New `tests/test_readme_commands.py`: the bench accepts every README board command (`--dry-run`) and both model configs generate.
- Every README command ran on jbpi as written: see the table below.

### README commands on jbpi (2026-11-21, core 4, correctness only)

| # | Command | Exit | Minutes | Run | Rows |
|---|---|---|---|---|---|
| 1 | BERT B1-B5 | 0 | 7 | `run_2026-11-21_205653` | 5 timed (check off) |
| 2 | Square S1-S5 | 0 | 3 | `run_2026-11-21_210019` | 5 timed (check off) |
| 3 | fp32 AXPY | 0 | 6 | `run_2026-11-21_210631` | 9/9 PASS |
| 4 | fp32 DOT | 0 | 2 | `run_2026-11-21_210831` | 3/3 PASS |
| 5 | int8 | 0 | 2 | `run_2026-11-21_211026` | 3/3 PASS |
| 6 | f16f16f32 `--scalar-pack` | 0 | 9 | `run_2026-11-21_211923` | 12/12 PASS |
| 7 | `--scalar-pack` | 0 | 6 | `run_2026-11-21_212540` | 9/9 PASS |
| 8 | `--pack-b col-g4` | 0 | 6 | `run_2026-11-21_213155` | 9/9 PASS |
| 9 | `--zero-pack-lines` | 0 | 6 | `run_2026-11-21_213811` | 9/9 PASS |
| 10 | `--pad-ldb 16` | 0 | 6 | `run_2026-11-21_214426` | 9/9 PASS |
| 11 | forced tile | 0 | 6 | `run_2026-11-21_215042` | 9/9 PASS |
| 12 | swept tile (search mode) | 0 | 23 | `run_2026-11-21_220847` | 1056/1056 PASS |
| 13 | `--threads 8` | 0 | 6 | `run_2026-11-21_221912` | 9/9 PASS |

All 13 exit 0; every checked row says PASS. The swept-tile run's 121 MB per-rep log is gitignored (over GitHub's 100 MB limit); its verdicts are in `all_raw_*.txt`.
