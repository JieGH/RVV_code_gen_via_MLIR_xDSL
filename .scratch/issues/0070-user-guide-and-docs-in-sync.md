# User guide for the high-level operations, and every README in step with the current code

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/134

**Role / Label:** `needs-triage`. Opened 2026-11-21 at the user's request, after #127 reorganized `tests/pure_xdsl_blis_rvv/`.
**Status:** closed 2026-11-21. `docs/user_guide.md` written; root README rewritten; other docs fixed against the code.
**Branch:** `user-guide-and-docs-in-sync`, cut from `pure-xdsl`.

---

## 1. Problem

The docs disagree with the code and with each other. Found on 2026-11-21 in the root `README.md` alone:

- "Our KerGOPs figure excludes packing": packing has been inside `KerGOPs_*` since #13 (`docs/metrics.md:59`).
- Board needs "GCC 14 or newer": the generated Makefile uses `g++` (13 on the board); only f16f16f32 projects use `g++-14` (`templates/Makefile:21-24`).
- Tests: "145 tests, of which 139 run on the host": the suite is 681 tests (7 `*_on_board`), about 17 min.
- The layout table lacks `configs/`, `experiments/`, `tests/` (#127), and the root README names no feature beyond the single benchmark and Tile Search: no data types, packing comparison, threads/YOLO, hand-over kernels.

There is no guide that walks a user through the high-level operations end to end.

## 2. What to build

1. **`docs/user_guide.md`**: one guide for the high-level operations, each with the command, what it writes, and how to read the result: set up; generate without a board; correctness pass; a benchmark run; head-to-head against the Reference Design; Tile Search; packing comparison; data types; threads and YOLO; reading results; house rules for a quoted number; host tests; adding a new experiment; troubleshooting.
2. **Root `README.md`**: fix the stale claims, add the new layout, point to the guide.
3. **Every other user-facing doc** (`CONTEXT.md`, `docs/*.md`, `tests/pure_xdsl_blis_rvv/{README,configs/README,experiments/README,ref_baseline/README}.md`): checked against the code; stale claims fixed or listed.

## 3. Out of scope

- `.scratch/issues/`, `logs/`, `notes/`: history, not rewritten.
- `.agents/skills/`: agent tooling, not user docs.

## 4. Tests

- Every command in the guide: `--dry-run` where it has one, or run on the host; board commands named, not run (one correctness pass is enough, as #127).
- Every path the docs name exists (script check).
- Host suite passes (docs only; `test_house_rules.py` scans docs for run folders).

## 5. Closure criteria

- [x] `docs/user_guide.md` written; linked from the root README and the folder README.
- [x] Root README and the other docs agree with the code; no dead path.
- [x] Host suite passes.

## 6. Results (2026-11-21)

- **`docs/user_guide.md`** (new): 15 sections, one per operation: set up, generate, write a config, correctness, benchmark, head-to-head, Tile Search, packing, data types, threads/YOLO, reading results, house rules, tests, new experiment, troubleshooting. Linked from both READMEs.
- **Root `README.md`**: feature table, new layout, quick start; fixed KerGOPs (packing included since #13), board compiler (`g++` 13; `g++-14` for f16f16f32), test count and command (681; 7 `*_on_board`).
- **Audit** of 16 docs by a Sonnet subagent; each claim used was checked here against the code. Fixed:
  - `ref_baseline/README.md` §1: said our GOPs exclude packing and `e2e` = pack + compute; both include packing since #13, `e2e` = `2mnk / RVV_Time_ms`.
  - `docs/agents/execution.md`: stale `config_correctness.json` path; "tracked `pj_gen/`" (it is gitignored); core-0 enforcement worded for scripts that take `--cpu`; `run_yolo_sweep.sh` `--tmin` default is 10; compiler line.
  - `docs/metrics.md`: `KerGOPs_*` denominator per build; `INSTRUMENTED=1` is the default of plain `make`, not of the script; default log name `_noinstr_refproto`; per-rep logs on by default; `skip_ref_run` true for `config.json` only.
  - `CONTEXT.md`: `../ext/RVV_ukernels_benchmark` path. `docs/dependencies_setup.md`: where to set `--mlir-translate`.
  - Marked historical: `docs/ROADMAP.md` (2026-09-16), `docs/int8_integration_report.md`, `docs/mac_method_comparison.md`. `docs/dev_note.md` is gitignored (a private note): left as it was.
  - `tests/pure_xdsl_blis_rvv/README.md`: missing scripts and tests, test count.
- **Path check** over the 10 user docs: no dead repo path (remaining hits are generated files, `../ext/`, Héctor's repo, a Linux kernel path).
- **Commands** in the guide checked by `--dry-run` (board reads faked): `run_yolo_sweep.sh`, `run_pack_swap.sh`, `run_parallel_gemm.sh`, `run_head_to_head.sh`, f16f16f32 correctness with `--scalar-pack`; refusals of `--threads 8 --cpu 4` and `--instr --tmin` as stated; `gen_handover_ukernels.py --out` writes its 10 files. Found and documented: `run_paper_sweep.sh` requires `--issue N`.
- **Full host suite:** 681 passed. `ref_baseline/generated/` restored.
- Left as is: `docs/index.md`, `docs/reference.md` and `mkdocs.yml` are the unused xdsl-template site scaffold.
