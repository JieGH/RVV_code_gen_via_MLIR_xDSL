# Move general tests into tests/, old notes into notes/, and write the folder README

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/131

**Parent:** #127, organize `tests/pure_xdsl_blis_rvv` by feature.
**Role / Label:** `needs-triage`. Spec, 2026-11-20.
**Status:** open. Blocked by #129 and #130.
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
- [ ] README reviewed by the user.
