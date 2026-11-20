# Experiments as plan files run by one series runner

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/119

**Parent:** #115, codebase architecture optimization.
**Role / Label:** `needs-triage`. Spec, 2026-11-20. Strength: **Worth exploring**.
**Status:** open. Not started. Blocked by #116 (board access module).
**Branch:** cut from `pure-xdsl` when work starts, named after this title.

---

## 1. Problem

Each experiment since #46 adds four files: `run_issueNN.sh` (230-527 lines), `config_issueNN_*.json`, `issueNN_report.py`, `test_run_issueNN.py`. Today: 11 run scripts, 22 issue configs, 14 run-script tests. New scripts start as a copy of the last one (`run_issue54_ref.sh:86`: "as run_issue55.sh").

The house rules for timing runs live in those copies: A-B-B-A order, three launches, gaps under 3% inside one launch, core 4, idle board, rotated row order (#61). A new experiment can drop one without anyone noticing.

The general runners (`run_yolo_sweep.sh`, `run_paper_sweep.sh`, `run_ref_sweep.sh`) already do most of this for one kind of series each.

## 2. What to build

- A **plan file** format: the configs, their A-B-B-A order, launches, threads, cores, packer, protocol flags, boards.
- One **series runner** that carries out any plan, using the Board access module (#116), and writes `plan.txt`, `runs.tsv` and the logs as today.
- New experiments add a plan file and, if needed, a report. No new run script.

Grilling decides: plan file format (JSON like configs, or shell-sourced), whether the general runners become plan files too, and which house rules the runner refuses to break.

## 3. Out of scope

- Converting closed issues' scripts. They stay, so their logs keep pointing at what ran.
- Report scripts (#117 gives them one reader; a shared report is a separate question).

## 4. Tests

- Runner `--dry-run` on sample plans: printed plan, written configs, order of launches.
- Refusals: plan that breaks a house rule (core 0, one launch for a quoted number, missing idle gate).
- One existing experiment (e.g. #94's plan) written as a plan file gives the same `--dry-run` output as its script.

## 5. Closure criteria

- [ ] One recent experiment reproduced as a plan file with an identical `--dry-run`.
- [ ] Runner documented in `docs/agents/execution.md`.
- [ ] Host pytest passes.
- [ ] One correctness series through the runner on jbpi (busy board allowed).
