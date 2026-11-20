# Codebase architecture optimization: deepen the board, log and protocol modules of the Project Generator harness

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/115

**Role / Label:** `needs-triage`. Parent issue. Opened 2026-11-20 at the user's request, from an architecture review (`/improve-codebase-architecture`).
**Status:** open 2026-11-20. Specs written for the five sub-issues; nothing implemented. No board runs.
**Branch:** `codebase-architecture-optimization`, cut from `pure-xdsl`. Each sub-issue gets its own branch when work starts.
**Scope:** `tests/pure_xdsl_blis_rvv/`, the files that changed most since 2026-09: `templates/driver_benchmark.cpp`, `compile_macro_bench.sh`, the `run_*.sh` scripts, the report scripts, `ref_baseline/`.
**Review page:** https://claude.ai/artifact/4WVzd5pKFzRehFt5PnKZJ6 (before/after diagrams for all five sub-issues; private until shared from its Share menu).

---

## 1. Why

Most new experiments since #46 copy the last `run_issueNN.sh` and its report script. The copies have drifted. A house rule (idle board, other users, core 4, log naming) now has to be fixed in up to 20 files, and tests check each copy separately.

Words used here, from the `/codebase-design` vocabulary:

- **Module**: anything with an interface and an implementation (a shell function file, a Python file, the driver).
- **Interface**: everything a caller must know to use it: arguments, outputs, error cases, ordering.
- **Deep** module: a lot of behaviour behind a small interface. **Shallow**: the interface is about as big as what it does.
- **Seam**: the place where the interface sits; behaviour can change behind it without editing the callers.
- **Locality**: a fix lands in one place. **Leverage**: one module serves many callers.
- **Deletion test**: if deleting a module just moves its code into callers, it was shallow; if the code comes back N times, it earned its place.

## 2. Findings (counts by grep, 2026-11-20, checked by hand)

| # | Finding | Evidence |
|---|---------|----------|
| 1 | Board access copied into run scripts | `board_ps()` defined in 7 scripts, all different; `StrictHostKeyChecking=no` in 17 scripts; 4 separate idle gates; 7 per-script `*_PS_CMD` test hooks |
| 2 | Driver log name and columns re-spelled by reports | suffix built at `driver_benchmark.cpp:711` (6 parts); re-spelled in `compare_pack_swap.py:34`, `report_scalar_sweep.py:39-40`, `compare_ref_sweep.py:9`, `compare_ref_baseline.py:56` (default name lacks `_noinstr`, so it misses the default build); 7 different row loaders |
| 3 | Protocol decided in three modules | default protocol in driver and `compile_macro_bench.sh:175-180`; `--tmin` refusal in `driver_benchmark.cpp:694` and `compile_macro_bench.sh:182`; `_refproto` suffix in driver, `templates/Makefile:48`, and the shell glob at `compile_macro_bench.sh:256` |
| 4 | One experiment = 4 files | 11 `run_issueNN.sh` (230-527 lines), 22 `config_issue*.json`, `issueNN_report.py`, 14 `test_run_*.py` |
| 5 | Four Reference Design patchers | `ref_pack_fix.py`, `repair_ref_kernels.py`, `ref_per_rep.py`, `fringe.py` patch C by string match |

## 3. Sub-issues

| Order | Sub-issue | Strength | Depends on |
|-------|-----------|----------|------------|
| 1 | Board access module (#116) | Strong | none |
| 2 | Raw-log reader (#117) | Strong | none |
| 3 | Driver owns the protocol and its log name (#118) | Worth exploring | pairs with #117 |
| 4 | Experiments as plan files for one series runner (#119) | Worth exploring | #116 |
| 5 | One Reference Design patch step (#120) | Speculative | none |

Top pick: **#116**. It removes the most copied code, `board_pin.sh` is already sourced by 17 scripts, and it holds the rules that decide whether a timing number counts.

## 4. Rules for every sub-issue

- **Behaviour stays the same.** Every `--dry-run` plan, log name and log column stays byte-identical unless the sub-issue says otherwise. Old logs must stay readable.
- **Closed issues keep their scripts runnable.** Their logs and reports cite them. Moving their shared code into a module is fine; deleting the scripts is not.
- **No tests during a board series.** Run `python -m pytest` from `tests/pure_xdsl_blis_rvv` only when no series is running; restore `generated/` with `git checkout` after.
- **No timing runs needed** for refactors that do not touch the driver or the generated C. #118 changes the driver: it needs one correctness run on the board (idle not required) before merge.
- **Pre-commit:** `--no-verify` (pyright strict fails on old generator code).

## 5. Closure

- [ ] Each sub-issue closed or marked `wontfix` with a reason.
- [ ] If a sub-issue is rejected for a lasting reason, record an ADR in `docs/adr/` so later reviews skip it.
