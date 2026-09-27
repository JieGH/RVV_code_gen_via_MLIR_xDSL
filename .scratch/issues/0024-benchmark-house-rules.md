# Issue #39: Benchmark house rules: no core 0, several launches for small gaps, keep every quoted log

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/39

**Role / Label:** `needs-triage` — from the #31-#34 audit report ([issues-31-34-brief.pdf](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-31-34-brief.pdf)), next step 5.
**Performance results per test:** Appendix A of the audit report, rows 2, 3, 13 and 21 (core-0 penalty; launch-to-launch spread).
**Brief:** [issues-35-36-39-brief.pdf](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-35-36-39-brief.pdf) ([LaTeX source](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-35-36-39-brief.tex)) — results of #35, #36 and #39, 2026-09-27.
**Status:** ✅ Implemented 2026-09-27 (see §Done).

## Problem
The #31-#34 audit found conclusions resting on setups the project already knew were weak:
- #32 ran all 32 launches on core 0, the same day #31 found core 0 runs 5-10% slower.
- #31 and #33 quote numbers from runs whose logs were never saved (kernel swaps, reference-side runs).
- #33 reads differences of 0.3-1.6% from single launches, while the same binary moves up to 1.7% between launches in the same hour.

## Proposed rules (not written yet)
1. Never pin to core 0; say which core in every result.
2. One launch is not a result: when a difference is under ~3%, compare inside one launch and repeat over several.
3. Every quoted number must come from a saved log under `tests/pure_xdsl_blis_rvv/logs/`.
4. Reconsider the stored default run command (`--cpu 0`).

Where: `docs/agents/execution.md` and `docs/metrics.md`.

## Done (2026-09-27)
- `docs/agents/execution.md`: new section **"House rules for a benchmark result"** with the four rules
  (never core 0; one launch is not a result under ~3%, compare inside one launch and repeat over ≥ 3 launches in
  A-B-B-A order; every quoted number from a committed log under `tests/pure_xdsl_blis_rvv/logs/`; name build,
  protocol and core). Example commands (reference, head-to-head, correctness) now say `--cpu 4`.
- `docs/metrics.md`: the new per-rep log (`--save-reps`, see below) is documented.
- `README.md`: example commands use `--cpu 4`; the `--cpu` row says never core 0 and points to the rules.
- `run_pack_swap.sh`: default core 0 → **4**. The two overnight sweep scripts (`run_scalar_pack_sweep.sh`,
  `run_ref_sweep.sh`) keep core 0 on purpose, so a rerun stays comparable with their 19-20 Sep core-0 runs; the
  rules say to pass `--cpu 4` for new work.
- Rule 4 (the stored default run command): now `--no-nc-overrides --cpu 4 --tmin 10`.
- Supporting tool for rule 2: `compile_macro_bench.sh --save-reps` writes every timed rep to `reps_*.txt`, and
  `reps_report.py` summarises a set of launches (mean over launches, lowest and highest launch). First used for
  #35/#36, which follow these rules (core 4, all settings inside one launch, 3 launches per packer, logs committed).
