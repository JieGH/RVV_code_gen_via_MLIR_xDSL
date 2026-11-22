# Per-rep logs fill most of the repo

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/143

**Role / Label:** `needs-triage`. Drafted 2026-11-22 at the user's request, from a walk through the project's structure. Audited the same day by a subagent; its findings are folded in.
**Branch:** per-rep-logs-fill-the-repo, cut from pure-xdsl
**Status:** implemented 2026-11-22 on branch `per-rep-logs-fill-the-repo`; open until the user closes it. Option chosen: C (see §5).

---

## 1. Problem

Board runs commit every file they write under `tests/pure_xdsl_blis_rvv/logs/`. One kind of file, the per-rep log `reps_*.txt.gz`, takes up most of the repo.

| What | Size |
|---|---|
| All tracked files (working tree) | 874 MB |
| `reps_*` files, working tree | 716 MB (82%) |
| - under `tests/pure_xdsl_blis_rvv/logs/` | 1,898 files |
| - under `.scratch/reports/` (copies made for reports) | 130 files |
| All blobs in git history (compressed, on disk) | 778 MB |
| `reps_*` blobs in git history (compressed, on disk) | 694 MB (89%) |

Measured 2026-11-22: working tree with `git ls-files -z | du -ch --files0-from=-`; history with `git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectsize:disk) %(rest)'`.

- Every other log kind is small: `raw_*` 12 MB, `perf_*` 13 MB, `summary_*` 14 MB, `cpu_times_*` 15 MB, `all_raw*` 7 MB, `all_summary*` 9 MB.
- One reps file has already gone over GitHub's 100 MB file limit: #137's search-mode run, 120 MB. `.gitignore:251-252` lists it by name and it is kept on this host only. Longer search runs will hit the limit again. The largest tracked reps file today is 3.2 MB.
- `logs/` has 18,489 tracked files. The count slows `git status`, grep and IDE indexing, but bytes are the bigger problem.
- `.pid` files from scheduled runs: 12 are tracked (e.g. `logs/issue51_rerun.pid`) and 3 more show up untracked in every `git status`.

### Why the reps files can't just be dropped

The reps files are inputs, not leftovers. Almost every reader finds them through one function, `readers/reps_report.py:190` `_reps_files()` (`rglob("reps_*.txt*")`). Readers:

- `readers/reps_report.py`: compares runs across launches (house rule from #39).
- `readers/slow_state_survey.py`
- `readers/search_report.py` (exits with an error when it finds no reps files)
- `readers/parallel_gemm_report.py`
- `ref_baseline/ref_per_rep.py`. It also writes `reps_ref_*.txt.gz` (54 tracked), the reference-design per-rep files.
- `experiments/issue46`, `issue48`, `issue55` report scripts.
- `.scratch/reports/`: `issue-54-figs/make_figs.py`, `issue-79/rep_tails.py`, `issue-94` and `issue-95` `make_data.py`, `yolo-multithread/report/metric_gap.py`.

Writers: `compile_macro_bench.sh:483-484` (gzips each reps file), plus `runners/run_yolo_sweep.sh`, `runners/run_parallel_gemm.sh` and `ref_baseline/run_ref_baseline.sh`.

Some readers need only per-launch stats. `rep_tails.py` and `slow_state_survey.py` need the full rep-by-rep series.

The boards never read reps files from git: `compile_macro_bench.sh` copies only `pj_gen/` and `datasets/` to the board. This is from reading the script; the boards themselves were not checked.

House rule: quote only committed logs. Any fix must keep every quoted number traceable to a file that other people can fetch. `tests/test_house_rules.py:37-52` checks this only loosely: it passes when a cited `run_*` folder has *any* tracked file.

## 2. What to do

Decide where the reps files live. Options:

- **A. Git LFS for `reps_*.txt.gz`.** Readers and paths stay the same, and the files stay "committed".
  - git-lfs isn't installed on this host, and no doc or script mentions it.
  - Needs a new `.gitattributes`.
  - Uses GitHub's LFS storage and bandwidth quota (check the current limits for this account).
  - Every clone and every worktree downloads the LFS files again. Two-board series use one worktree per board.
  - Stops new growth only. The 694 MB already committed stays unless `git lfs migrate` rewrites history.
- **B. Ignore `reps_*` in git, and upload each run's reps files as a GitHub Release asset** (one `.tar` per run, named after the run folder). Each run folder gets a committed `reps_archive.txt` that points to its asset. The repo stays small, and each file can be up to 2 GB. Readers need a fetch step; put it in `_reps_files()` so the readers above get it at once.
- **C. Ignore `reps_*` in git and keep the files on the NFS home only.** Simplest, but it breaks the committed-logs rule for everything that reads reps files, and it depends on one disk.
- **D. Commit less data.** Commit a per-launch summary (case, n, mean, slow share, first slow rep) and keep the full series outside git (B or C). Before choosing D, check which readers really need the full rep-by-rep rows (at least `rep_tails.py` and `slow_state_survey.py` do).

Open question for the user: A, B, C or D.

Whichever option is picked, also:

- Untrack the 12 tracked `.pid` files (`git rm --cached`) and ignore `tests/pure_xdsl_blis_rvv/logs/*.pid`.
- With B, C or D: use a narrow path pattern (e.g. `tests/pure_xdsl_blis_rvv/logs/**/reps_*.txt.gz`), not a bare `reps_*`. Decide on purpose whether `reps_ref_*` and the `.scratch/reports/` copies are covered.
- With B, C or D: make the post-run step in `compile_macro_bench.sh` print where the reps files went, so a run never drops them without a message.
- Replace the two per-file reps entries in `.gitignore:251-252` with the new rule.
- Reword the "committed" lines in the subproject `README.md` (line 35) and `docs/agents/execution.md:14` if the rule changes.
- With B: make `test_house_rules.py` check that each cited run has its reps data or its `reps_archive.txt`.

## 3. Out of scope

- Rewriting git history to remove the 694 MB of reps blobs already committed. That needs a force push, and every clone and board worktree would need a fresh clone. Separate decision, separate issue if wanted. Without it, this issue stops growth only; existing clones stay at their current size.
- Cutting the small files `compile_macro_bench.sh` writes per run (raw, perf, summary, cpu_times).
- Moving or renaming `logs/` or the run folders.

## 4. Closure criteria

- [ ] A new run adds no `reps_*` blob to normal git history (A: the blob goes to LFS; B, C, D: the file is ignored). Check: `git check-ignore -v` on a new reps path (B, C, D), or `git lfs ls-files` (A).
- [ ] `git ls-files '*.pid'` is empty and `git check-ignore` matches a new `logs/x.pid`.
- [ ] `readers/reps_report.py`, `readers/slow_state_survey.py`, `readers/search_report.py` and `ref_baseline/ref_per_rep.py` give the same output as before on `logs/run_2026-11-17_194103` (old run) and on one run made after the change.
- [ ] A fresh clone in a temporary folder can run `readers/reps_report.py` on both runs (A: after `git lfs pull`; B: after the fetch step). Under C this is expected to fail for the new run; say so in the result.
- [ ] `tests/test_house_rules.py` covers the new rule.
- [ ] Host tests pass (`python -m pytest` from inside `tests/pure_xdsl_blis_rvv`).

## 5. Result (2026-11-22)

**Choice.** The user picked option C: "these files should not be on github, its private to dev". Coverage: "Both covered", so `reps_ref_*` and the copies under `.scratch/reports/` follow the same rule. The user's words cover files already in the repo, so those were untracked too, not only new ones.

**Done:**
- `.gitignore`: `reps_*.txt`, `reps_*.txt.gz` (anywhere in the repo, as "Both covered" asks; the §2 narrow-path idea did not fit that choice) and `tests/pure_xdsl_blis_rvv/logs/*.pid`. These replace the two #137 single-file entries.
- Untracked with `git rm --cached`: 2,024 per-rep logs and 12 pid files. The files stay on this machine's disk. No cited run folder is left with zero tracked files.
- `tests/test_house_rules.py`: `test_no_per_rep_log_or_pid_file_is_committed` and `test_new_per_rep_logs_and_pid_files_are_ignored` (a reps log, a `reps_ref_*`, a `.scratch/reports` copy, a pid). Both failed before the change and pass after.
- `compile_macro_bench.sh`: after the fetch it prints where the per-rep logs are and that git ignores them (only when `--save-reps` is on).
- Docs: rule 3 in `docs/agents/execution.md` now names the exception and what stays checkable from a clone; also `docs/metrics.md`, `docs/user_guide.md`, the subproject `README.md`, `readers/README.md`.

**Checks:**
- Full host suite: 754 passed (`python -m pytest` inside `tests/pure_xdsl_blis_rvv`, 17 min 55 s); `ref_baseline/generated/` restored after.
- `reps_report.py logs/run_2026-11-17_194103` (old run) reads the now-untracked local files and prints the usual table.
- Pyright on `test_house_rules.py`: 0 errors.
- Code review (standards + spec subagents): stale lines in `metrics.md` and `user_guide.md`, the unconditional echo and the rule-3 wording were fixed. Kept: the test's regex repeats the `.gitignore` patterns; `git ls-files -ci --exclude-standard` cannot replace it, because the repo already tracks ignored files (`.scratch/reports/issue-62/*.log`).

**Limits (as option C expected):**
- A fresh clone has no per-rep logs. `reps_report.py`, `search_report.py`, `slow_state_survey.py` and `ref_per_rep.py` work only on the dev machine. Launch spread and slow-rep share can be checked there only. Summary GOPs (`all_*`, `summary_*`) stay committed.
- The 694 MB of per-rep logs already in git history are still on GitHub. Removing them needs a history rewrite and a force push (out of scope, §3).
- A checkout that pulls this change (e.g. a board worktree) deletes its own copies of the untracked files; this checkout keeps them.

### Closure criteria

- [x] New runs add no `reps_*` blob (ignored; tested).
- [x] `git ls-files '*.pid'` is empty; new `logs/*.pid` ignored (tested).
- [x] Readers work on the old run here. A new run comes with the next board run.
- [x] Fresh clone: expected to fail for the per-rep readers under C, as noted above.
- [x] `test_house_rules.py` covers the rule.
- [x] Host tests pass.
