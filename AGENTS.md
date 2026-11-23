# Agent Guidelines for RVV_code_gen_via_MLIR_xDSL

## Workspace Context
- **Active Repository:** This repository (`RVV_code_gen_via_MLIR_xDSL`) is the primary active project (`JieGH/RVV_code_gen_via_MLIR_xDSL.git`).
- **Reference Directories:** Sibling folders outside this repository (`../ext/`, `../llvm-project/`, `../xxxxxdsl-template/`) are strictly read-only reference sources. Never modify, commit, or delete them.
- **Active Subproject:** `tests/pure_xdsl_blis_rvv/` is the primary active subproject (the "Project Generator" paradigm for RISC-V Vector microkernels).
- **Working Tree Integrity:** Working tree changes are live and uncommitted. Never run destructive git commands (`git reset --hard`, `git clean -fd`, etc.).
- **Passive & Inspection Autonomy:** Prefer native read tools (`list_dir`, `view_file`, `grep_search`, `find_by_name`) for zero-prompt inspection. All non-mutating inspection operations (`ls`, `cat`, `git status`, `git diff`, inspecting logs) are pre-approved and should execute autonomously as long as no files are modified. Never alter existing code without explicit user alignment.
- **Permission & Risk Transparency:** Whenever requesting permission or proposing a non-trivial command/script, always explain: (1) what the operation (`op`) actually does in plain English, and (2) what the associated risks are (e.g. file changes, remote board side-effects, git mutations, or zero risk).

## Agent skills

### Issue tracker
GitHub Issues via `gh` CLI for `JieGH/RVV_code_gen_via_MLIR_xDSL`. See `docs/agents/issue-tracker.md`.

### Triage labels
Standard 5-role vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs
Single-context layout (`CONTEXT.md` and `docs/adr/`). See `docs/agents/domain.md`.

### Execution
Running the pipeline against the RISC-V board (direct vs. jump-host, entry-point scripts, reference command). See `docs/agents/execution.md`.

### Board connection
The real user, host, port and board names live in `docs/agents/board.local.md` (gitignored, local only). Read it before any board run and use it to fill the `<...>` placeholders in README.md. If it is missing, ask the user for the values.
