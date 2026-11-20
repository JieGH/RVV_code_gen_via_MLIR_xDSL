---
description: Work through every open sub-issue of the architecture parent issue, one at a time, with no manual steps
argument-hint: "[parent issue number, default 115]"
---

Work through the sub-issues of parent issue #${ARGUMENTS:-115} until none are left or one is blocked. Do not ask the user questions. Where a spec says "settled in grilling", decide yourself, pick the smallest design that meets the closure criteria, and write the decision into the issue.

## State lives on GitHub, not in your head

Context is compacted at 35% usage. After a compaction, re-derive where you are from these commands. Never rely on memory of earlier turns.

- Sub-issues: `gh api repos/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/<parent>/sub_issues --jq '.[] | {number,title,state}'`
- Blockers: `gh api repos/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/<n>/dependencies/blocked_by --jq '.[] | {number,state}'`
- Local spec: `.scratch/issues/00NN-*.md` (find the file with `grep -l "issues/<n>" .scratch/issues/*.md`)
- Progress: your last comment on the issue, plus `git log` on its branch.

## Loop

1. **Pick the next issue.** Open sub-issues with no open blocker, in this order: Strong, Worth exploring, Speculative (strength is in each spec). If none is free but some are open, stop and report which blockers remain. If none are open, go to "Finish".
2. **Read** the issue's local spec file in full. Do not start if its Status says a board series is running or it is `wontfix`.
3. **Branch** from `pure-xdsl`, named after the issue title in kebab-case, no number. Never commit this work on `pure-xdsl` or `main`.
4. **Build** what the spec's section "What to build" says, and nothing from "Out of scope". Record each design decision as one line in the spec file under a new section "Decisions". Keep `--dry-run` output, log names and log columns identical unless the spec says otherwise.
5. **Test.**
   - Host tests: from `tests/pure_xdsl_blis_rvv`, `python -m pytest`. Check first that no board series is running (`pgrep` is wrong here: use the saved PID files in `logs/*.pid` with `kill -0`). If one runs, wait for it, never test under it.
   - Afterwards restore `generated/` with `git checkout`. Never run `git reset --hard` or `git clean`.
   - Where the spec's closure criteria demand a diff of old and new output, produce it and save it in the issue's report folder under `.scratch/reports/issue-<N>/`.
6. **Board correctness run**, only when the closure criteria list one (#118, #119). Follow `docs/agents/execution.md`. Check other users first. A correctness-only run may use a busy board. Never kill another user's process. Quote no timing. If the board is unreachable, stop this issue, comment why, and move to the next free one.
7. **Close out.**
   - Tick each closure box in the spec file that is true. Leave unproven ones unticked and say why.
   - Sync the spec to GitHub: `gh issue edit <n> --body-file <spec file>`.
   - Commit with `--no-verify` (pyright strict fails on old generator code; the user confirmed this). End the message with the attribution trailer Claude Code currently gives you.
   - Comment on the parent issue: one line with the sub-issue number, what changed, and the test result.
8. **Close and merge only if every closure box is ticked and host pytest passes.** Then close the issue with `gh issue close`, open a PR from the issue branch into `pure-xdsl`, merge it, then PR `pure-xdsl` into `main` and merge it. If any box is unticked or any test fails, leave the issue open, leave the branch pushed, and say so in the parent comment. Do not merge a failure.
9. Go back to step 1.

## Finish

- Tick the parent's closure boxes that are true and sync it to GitHub.
- Final report to the user, short: per sub-issue, state (closed / open and why), branch, commit, test result, and anything not checked.

## Stop and report at once if

- A test fails twice for the same cause.
- A change would alter a log name, a log column or a `--dry-run` plan that the spec says must stay the same.
- You would have to touch `../ext/`, `../llvm-project/` or `../xxxxxdsl-template/`. They are read-only.
- You are about to delete a closed issue's script.
