# Cron runs on jbpi failed "Stale file handle" because jbpi and jbpi2 share one directory

**Role / Label:** `done` — cause found from the logs and read-only probes, fix written. Run overnight, see Status.
**Status:** ✅ confirmed by the 19–20 Sep overnight schedule: all 16 runs (8 per board) finished with no errors. Each run took about 7 min (cron start 23:00:01, run folder written 23:06). This shows the queue works, not that the diagnosis is the only possible cause; the failing case was never reproduced on purpose.
**Target:** `jbpi` and `jbpi2` (BananaPi, VLEN=256), reached via `fej`.
**Primary Sources:**
- `tests/pure_xdsl_blis_rvv/logs/cron_jbpi_20260918_{00,01,02,03,04}0001.log` (all five failed), `cron_jbpi2_20260918_*` (all five passed), `cron_jbpi_20260917_224037.log` (manual run, "Directory not empty").
- `tests/pure_xdsl_blis_rvv/logs/report_2026-09-18/HANDOFF.md`, which called this "a remote filesystem fault on that board".
- `tests/pure_xdsl_blis_rvv/cron_bench_run.sh`, `compile_macro_bench.sh` lines 296–301, `generate_pure_blis_project.py` line 771.

---

## 1. The symptom

`jbpi` failed 5 of 5 scheduled runs on 2026-09-18 on the board, at the first build step that reached the second file:

```
Fatal error: can't create gemm_b1_xdsl.o: Stale file handle
bash: line 1: cd: fp32_AXPY_FP32_Pack_vlen256_MR32NR6: No such file or directory
```

`jbpi2` passed all five, each run about 40 min. The earlier manual run failed differently: `rm: cannot remove '.../MR32NR6/logs': Directory not empty`. The handoff read this as a fault on the `jbpi` board.

## 2. The cause

It is not the board. **`jbpi`, `jbpi2` and `fej` see one directory.** Read-only probe on 2026-09-19: on all three, `~/xdsl_rvv_microkernel/tests/pure_xdsl_blis_rvv/pj_gen` is inode `145555544` under `/nfs/gap/jlei/`, exported by `fej` (`192.168.1.1:/data`).

`compile_macro_bench.sh` does `rm -rf .../pj_gen` on the board, syncs a fresh copy, then builds in it. Cron started `jbpi` at :00 and `jbpi2` at :01, so `jbpi2` deleted the tree while `jbpi` was compiling in it. The NFS client on `jbpi` then held handles to files that no longer existed: "Stale file handle", then "No such file or directory". `jbpi2` ran second and alone, so it always passed. The manual run on 09-17 hit the same thing from the other side (`rm` racing with another writer).

The local side is shared the same way: the generator wipes local `pj_gen` (`generate_pure_blis_project.py:771`) and both cron jobs use it.

Not proven by a reproduction. It fits the five failures, the five passes, the one-minute offset and the shared inode. It was not tried with the two runs started apart.

## 3. The fix

`cron_bench_run.sh` takes a lock (`flock`, file `logs/.cron_bench.lock`) before it does anything, waits up to 3 h, then runs. Two entries fired at the same minute now queue instead of colliding. The lock is local, which is enough because both cron entries start on this machine.

Checked: two `flock` calls on one file, the second waited for the first (2 s in the test). `bash -n` passes. **The wrapper was not run against a board.**

Schedule replaced. The old crontab entries were for 18 Sep only (`0 0-4 18 9 *`) and would not fire again until 2027. New: the boards take turns, one run every 30 min (about 10 min run with `--tmin 10`, then 20 min idle). `jbpi` at :00 and `jbpi2` at :30, from 23:00 on 19 Sep to 06:30 on 20 Sep: 16 runs, 8 per board, the last one ends about 06:40. The wrapper now passes the default run command flags (`--no-nc-overrides --cpu 0 --tmin 10`). The 09-18 cron runs took about 40 min each because they were an older protocol; a `--tmin 10` run ended 22:37 on 09-19 after roughly 9 min (estimated from the log times, the start was not logged). If a run takes longer than 30 min, the next one waits on the lock and the times drift, but nothing collides.

## 4. What this does not establish

- Whether `jbpi` and `jbpi2` differ in speed. There is still no cross-board comparison, only jbpi2 run to run. This run should give the first one.
- A run touched by hand at the same time as cron can still collide. Manual runs do not take the lock.
- Running the two boards in parallel would need a separate workspace per board on the board side and a separate local `pj_gen`. Not done; the queue costs wall time but not correctness.
