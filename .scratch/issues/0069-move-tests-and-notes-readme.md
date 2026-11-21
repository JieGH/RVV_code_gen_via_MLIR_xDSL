# Move general tests into tests/, old notes into notes/, and write the folder README

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/131

**Parent:** #127, organize `tests/pure_xdsl_blis_rvv` by feature.
**Role / Label:** `needs-triage`. Spec, 2026-11-20.
**Status:** closed 2026-11-21. 35 tests in `tests/`, 6 notes in `notes/`, folder `README.md` with the feature map.
**Branch:** `organize-test-folder-by-feature`.

---

## 1. Problem

35 general `test_*.py` files sit among the tools they test. Six old notes (`PLAN.md`, `LLM_AUDIT_GUIDANCE_2026-07-11.md`, three `xdsl_*_report.md`, a `debug perfromance ... .md` export) are cited by no file. The folder has no README saying which feature lives where.

## 2. What to build

- `git mv` every `test_*.py` left at the top level into `tests/`; fix each one's `HERE` to point at the project folder. Tests that import another test's helpers (`from test_pad_zeroing import ...`, `from test_ref_protocol import ...`) keep working because both move together.
- `git mv` the six notes into `notes/`.
- Write `README.md` for the folder: the parent's §2 table (feature -> entry point -> configs -> tests) with the new paths, the layout tree, and how to run the host tests (`.venv/bin/python -m pytest` from this folder).
- Update `docs/agents/execution.md` and any other doc that names a moved test.

## 3. Out of scope

- Splitting `tests/` further by feature. The README table does that job without more moves.

## 4. Tests

- Full host suite passes, same test IDs as the baseline up to the folder.
- Snapshot compare clean.

## 5. Closure criteria

- [ ] Top level holds only the generator, bench, general runners and readers, `config.json`, `README.md` and the folders.
- [ ] Snapshot compare clean; full host suite passes.
- [ ] README reviewed by the user (left open for the user; not blocking).

## 6. Results (2026-11-21)

- Moves by a Sonnet subagent, checked here. Each moved test's `HERE` is `parents[1]` (still the project folder); tests that import another test's helpers (`test_pad_zeroing`, `test_ref_protocol`) add `tests/` to `sys.path`. `test_house_rules.py` also scans `notes/*.md`, so the moved notes stay in its cited-run-folder check. `pyproject.toml` ruff ignores and `docs/agents/execution.md` follow the new test paths.
- `README.md` (written here): layout, feature -> entry point -> configs -> tests, the pinned/search row rule, how to run the host tests.
- **Snapshot compare against the #128 baseline:** generated C and `--dry-run` overview identical for all 45 configs.
- **Full host suite:** 681 passed (baseline 674, same IDs, + 7 feature smoke tests), 16.6 min. `ref_baseline/generated/` restored.
- Top level now: 31 files (generator, bench, board module, general runners and readers, `config.json`, `README.md`) and 11 folders, from 152 files.
