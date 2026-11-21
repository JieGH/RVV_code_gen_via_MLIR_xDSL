# Stale crontab entries call cron_bench_run.sh at its old path

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/140

**Role / Label:** `needs-triage`. Drafted 2026-11-21 at the user's request, after #137 moved `cron_bench_run.sh` into `runners/`.
**Status:** fixed 2026-11-21 (entries removed); close after merge.

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

## 2. What to do (pick one)

- **Remove the four entries and their comment line.** Recommended: they belong to a finished one-off sweep, and today's runners schedule themselves (`--start-at`, `--after-pid`).
- Or point them at `runners/cron_bench_run.sh`, if the user wants the same sweep again next September. Then also check the date fields.

The crontab sits outside the repo; only the user edits it (`crontab -e`). Look first: `crontab -l`.

## 3. Out of scope

- Changing `cron_bench_run.sh` itself.

## 4. Closure criteria

- [x] `crontab -l` names no path under `tests/pure_xdsl_blis_rvv/` that does not exist.

## 5. Result (2026-11-21)

- Took the recommended option: removed the four entries and their comment line. They were the whole crontab, so it is now empty.
- Backup before the change: `.scratch/reports/issue-140/crontab.before`. To undo: `crontab .scratch/reports/issue-140/crontab.before`.
- Check: `crontab -l | grep -o '/[^ ]*tests/pure_xdsl_blis_rvv/[^ ]*'` finds no path, so none is missing.
- No repo code changed.
