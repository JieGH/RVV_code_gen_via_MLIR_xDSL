# One issue's files are spread over five places

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/146

**Role / Label:** `needs-triage`. Drafted 2026-11-22 at the user's request, point 3 of the project-structure walk-through (point 1 was #143). Audited the same day by a subagent; its findings are folded in.
**Branch:** issue-files-spread-over-five-places, cut from pure-xdsl
**Status:** closed 2026-11-22. Options done: A + B + D (see §5).

---

## 1. Problem

The work for one issue can sit in five places:

| Place | Naming | Example for #85 |
|---|---|---|
| `.scratch/issues/` | **local** number `NNNN-slug.md` | `0051-when-does-vector-packing-pay-off.md` |
| `tests/pure_xdsl_blis_rvv/experiments/` | GitHub number `issueNN/` | `experiments/issue85/` |
| `tests/pure_xdsl_blis_rvv/datasets/` | `issueNN_*.dat`, flat with every workload | `issue85_sweep.dat`, `issue85_tile.dat`, `issue85_correctness_small.dat` |
| `tests/pure_xdsl_blis_rvv/logs/` | mostly `run_<stamp>` (628 folders), no issue in the name | cited from the issue text |
| `.scratch/reports/` | GitHub number, several styles | `issue-85/`, `issue-54-brief.tex`, `issues-31-34-brief.tex`, `yolo-multithread/` |

Four of the five places already use the GitHub number. The two real problems are narrower than "five places":

### 1a. Two number spaces

- Issue files use a local number; everything else uses the GitHub number. `0051` is #85, `0035` is #51, `0073` is #143. All 58 linked files have a local number that differs from their GitHub number.
- The only bridge is the `GitHub:` line inside each file. 18 of the 76 issue files have none (local 0001-0014, 0025 twice, 0029, 0039).
- Some numbers hold several files: 0020 (3 files), 0025 (5), 0026 (3), 0039 (2: spec and progress log), mixing `.md`, `.c` and `.txt`.

### 1b. `datasets/` mixes workloads, one-off inputs and unused files

50 files:

- **Shared workloads named after an old issue** (a reader of `config_bert.json` must look up #2 and #26 to learn they are BERT layers):

| Dataset | Named in (outside `.scratch` and `logs`) |
|---|---|
| `issue2_bert_16x15.dat` | `config.json`, `config_bert.json`, `config_pack_swap_b2b3b4.json`, `config_dim_sweep_b3.json`, `experiments/issue36` config |
| `issue2_bert_32x6.dat` | `config.json`, `config_bert.json`, `config_b1b5_nr6_first.json`, `config_b1b5_nr7_first.json` |
| `issue2_square_bert_32x7.dat` | `config.json`, `config_dim_sweep_b4.json`, `experiments/issue37` config |
| `issue26_b4.dat` | `config_bert.json`, `config_pack_swap_b2b3b4.json`, `config_dim_sweep_b4.json`, `config_dim_sweep_16x16_24x10.json` |
| `issue46_ref_b3.dat` | experiments #46, #48, #49, #50, #55 (configs, run scripts, `test_run_issue55.py:401`), `ref_baseline/cases_issue46.tsv`, `docs/agents/execution.md` |

- **One-issue inputs:** the other 16 `issueNN_*` files. 14 are used only inside their own `experiments/issueNN/`; `issue26_b3` and `issue26_s1` are used by `configs/benchmark/config_pack_swap.json`.
- **Unused:** 11 files are named nowhere outside `datasets/`: `gnn`, `googlenet`, `googlenet.blis`, `gpt2_large`, `resnet50`, `resnet50.blis`, `vgg16`, `vgg16.blis`, `square_small`, `tiny_test`, `trail`.
- **Default run:** with no `datasets` list, the generated Makefile runs every `datasets/*.dat` (`templates/Makefile:86`), one-off and unused files included.

### How dataset names work (checked)

- Configs name datasets by path relative to `datasets/`; the generator writes `../../datasets/<name>` into the Makefile (`generate_pure_blis_project.py:911-913`), and result files are named by `basename`.
- `compile_macro_bench.sh` copies the whole `datasets/` folder to the board with `rsync`, subfolders included.
- So a sub-path such as `experiments/issue85_sweep.dat` works today, and files in a subfolder drop out of the default `*.dat` run.
- The 7 saved `logs/*/config.json` files name none of the 5 shared datasets. Old names in other log files are history text, not rerun inputs.

## 2. What to do

Options; they can be combined.

- **A. Rename the 5 shared workloads** to what they are (names to agree, e.g. `issue26_b4.dat` → `bert_b4.dat`). Edit the 23 files in the table (configs, run scripts, `test_run_issue55.py:401`, `ref_baseline/cases_issue46.tsv`, `docs/agents/execution.md`), a comment in `ref_baseline/cases.tsv`, and the `# issueNN` header comments inside the datasets. `.scratch` text (26 files) keeps the old names as history; a `datasets/README.md` maps old → new.
- **B. Move one-issue and unused datasets into subfolders:** `datasets/experiments/issueNN_*.dat` (configs name `experiments/<file>`) and `datasets/unused/` (or delete the 11 unused ones). Works with today's generator and board copy. Side effect: they leave the default `*.dat` run, which then covers only real workloads.
- **C. Rename issue files to their GitHub number.** Risky: local and GitHub numbers overlap for different issues (local 0051 is #85; renaming 0035 → 0051 hits the current 0051). Needs one rename pass from a full old → new table, then every link rewritten in one go: 58 link strings in 13 `.scratch` files and 5 memory notes. The 18 files without a GitHub issue and the multi-file numbers need a rule.
- **D. Generated index:** a script builds `.scratch/issues/README.md` from each file's `GitHub:` line and the folders: GitHub number, local file(s), `experiments/issueNN/`, its datasets, `.scratch/reports/issue-NN*`. Files with no `GitHub:` line show as "local only"; all files under one local number group in one row. Nothing moves, no links break.

Recommendation: A + B + D. C costs the most and its job is mostly done by D.

### Decision (2026-11-22)

The user chose the recommendation: **A + B + D**. C is dropped.

Proposed names for A (from each file's header; the user can change them before work starts):

| Old | New | Holds |
|---|---|---|
| `issue2_bert_16x15.dat` | `bert_b2_b3.dat` | B2, B3 (tile 16x15) |
| `issue2_bert_32x6.dat` | `bert_b1_b5.dat` | B1, B5 (tile 32x6) |
| `issue2_square_bert_32x7.dat` | `square_bert_b4.dat` | S1-S5 and B4 (tile 32x7) |
| `issue26_b4.dat` | `bert_b4.dat` | B4 alone |
| `issue46_ref_b3.dat` | `b3_controls.dat` | B3 with controls B2, B3_K256, R24 |

For B: move the 11 unused datasets to `datasets/unused/` rather than delete them (easy to undo).

## 3. Out of scope

- Renaming or moving `logs/run_*` folders: issue files and reports cite them by name, and `test_house_rules.py` checks those names.
- Editing old issue text in `.scratch/` or on GitHub to the new dataset names.
- Changing how datasets reach the board.

## 4. Closure criteria

- [x] (A, B) `grep -r` for each old dataset name finds nothing outside `.scratch/`, `logs/` and the `datasets/README.md` map.
- [x] (A, B) `config.json`, `configs/benchmark/config_bert.json`, `configs/benchmark/config_pack_swap.json` and one moved-dataset experiment config generate (`generate_pure_blis_project.py`), and each generated Makefile names dataset files that exist. Restore `ref_baseline/generated/` with `git checkout` after any `--dry-run`.
- [x] (D) Changed while building it: the index is built from the folders that exist, so "named but missing" can't happen. Instead `make_index.py --check` exits non-zero when the committed index is stale, and `test_issue_index.py` runs it (§5).
- [x] Host tests pass (`python -m pytest` from inside `tests/pure_xdsl_blis_rvv`), `ref_baseline/generated/` restored after.

## 5. Result (2026-11-22)

**A. Renamed** the 5 shared workloads as in the decision table (`bert_b2_b3`, `bert_b1_b5`, `square_bert_b4`, `bert_b4`, `b3_controls`).

**B. Moved** 16 one-issue datasets to `datasets/experiments/` and the 11 unused ones to `datasets/unused/` (kept, not deleted).
- 47 configs, scripts, tests, docs and dataset files now use the new names. A second grep found 3 `${HERE}/datasets/<old>` forms the first pass missed (`run_issue51.sh`, `probe_issue95.sh`, `ref_baseline/cases.tsv`); fixed.
- New `datasets/README.md`: what goes where, and old → new names. `.scratch/` and `logs/` keep the old names as history.
- A config with no dataset list now runs only the top-level workloads; subfolders are left out.

**D. Index.** `.scratch/issues/make_index.py` writes `.scratch/issues/README.md`: one row per GitHub issue (61), with title, local issue file(s), `experiments/issueNN/`, one-issue datasets and reports. Files that share a local number join that number's GitHub issue (0025, 0039 sub-files). 15 local numbers have no `GitHub:` line and are listed as local only. Titles were not used to guess GitHub numbers: several say "(local)" and #7 appears twice. Report names `issues-A-B` count as a range (#31-#34), names with 3 or more numbers as a list.

**Docs:** root `README.md`, `docs/user_guide.md`, `docs/agents/issue-tracker.md`, `tests/pure_xdsl_blis_rvv/README.md`, `experiments/README.md`.

**Tests (new):**
- `test_layout_and_scripts.py`: every one of the 50 configs names datasets that exist; no `issue*.dat` at the top of `datasets/`, and `datasets/README.md` exists.
- `test_issue_index.py`: the committed index is up to date (`--check`); the #85 row links `0051-...md`, `experiments/issue85/`, `issue85_sweep.dat` and `reports/issue-85/`; local-only files are listed; a two-number report name covers the whole range.

**Checks:**
- Full host suite: 808 passed (17 min 53 s). It ran before the last index fix and the README edits; the affected files ran again after: `test_layout_and_scripts.py`, `test_issue_index.py`, `test_readme_commands.py`, 115 passed. `ref_baseline/generated/` restored.
- Generator on `config.json`, `configs/benchmark/config_bert.json`, `configs/benchmark/config_pack_swap.json` and `experiments/issue85/config_issue85_sweep.json`: every dataset path in the generated Makefiles exists.
- Pyright on `make_index.py` and `test_issue_index.py`: 0 errors.
- Code review (standards + spec subagents): fixed crashes and bogus rows on odd names (strict patterns, issue numbers up to 3 digits), each file read once, ranges for `issues-A-B`. Kept: the #85 test uses fixed names on purpose (a known-good example).

**Known limit:** `test_issue_index.py` fails after any new issue file, experiment folder, one-issue dataset or report, or a changed issue title, until `make_index.py` runs again. That is the reminder the index needs.
