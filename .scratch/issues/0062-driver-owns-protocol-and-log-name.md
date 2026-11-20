# Driver owns the protocol and its log name; the shell and Makefile stop working them out again

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/118

**Parent:** #115, codebase architecture optimization.
**Role / Label:** `needs-triage`. Spec, 2026-11-20. Strength: **Worth exploring**.
**Status:** open. **Analysed 2026-11-20, not implemented: the design below does not hold against the code** (section 6). Needs a decision before any work.
**Branch:** cut from `pure-xdsl` when work starts, named after this title.
**Pairs with:** #117 (raw-log reader).

---

## 1. Problem

Which protocol a run uses (Reference Protocol or Warm-up Protocol) is decided in three modules:

| Decision | driver_benchmark.cpp | compile_macro_bench.sh | templates/Makefile |
|----------|---------------------|------------------------|--------------------|
| Default protocol per build | :576 | :175-180 | |
| `--tmin` only with the Reference Protocol | :694 | :182 | |
| `_refproto` in the log name | :711 | :256 (glob), :277, :295 | :48 |

The `--tmin` refusal text exists twice; `test_ref_protocol.py:674+` tests only the shell copy. Adding a flag that changes the log name means editing the driver's argv parser (:591-644, one hand-written branch per flag), its usage text, the shell, the Makefile and the reports.

## 2. What to build

- The **driver** alone decides the protocol, refuses bad flag sets, names its log, and **prints the result path** on one fixed line.
- **`compile_macro_bench.sh`** passes protocol flags through and reads that line to find the log. It keeps its own checks only where it must refuse before building (decided in grilling).
- **`templates/Makefile`** drops the suffix logic.

Open question for grilling: should the shell keep the early `--tmin` refusal (fast fail before a board build) and the driver keep the authoritative one? If so, one test checks both say the same.

## 3. Out of scope

- Changing either protocol's timing.
- Changing log names. Old and new runs must write the same names.
- A general argv library for the driver (only if grilling finds the parser is the real cost).

## 4. Tests

- Driver tests (host build, as `test_ref_protocol.py` already does): default per build, `--tmin` refusal, printed result path matches the file written, for each suffix part.
- Shell tests: the flag is passed through; the log is found from the printed line.

## 5. Closure criteria

- [ ] `_refproto` appears in one source file (driver) plus the reader from #117.
- [ ] Log names for every flag combination unchanged (table in this issue).
- [ ] Host pytest passes.
- [ ] One correctness run on jbpi (`compile_macro_bench.sh`, uninstrumented, Reference Protocol, `--cpu 4`), PASS; busy board allowed, no timing quoted.

## 6. Findings (2026-11-20, agent, before writing any code)

The spec's design ("the driver prints its result path; the shell and Makefile stop working it out") does not work as written:

1. **The Makefile needs the name before the driver runs.** `templates/Makefile:92` does `./$(TARGET) $$dataset $(TEST_ARGS) | tee "results_..$(RESULT_SUFFIX).csv"`. The shell opens that file first, so the driver's output cannot name it.
2. **The shell's name is for display and happens on the host.** `BUILD_LOGS` (`compile_macro_bench.sh:255-275`) feeds the `--dry-run` banner, "Logs" lines at :329, :478, :535. A dry run must not need the board's binary, so the host cannot ask the driver. The spec says `--dry-run` output must stay identical.
3. **The Makefile suffix is shorter than the driver's.** It covers `_noinstr _refproto _scalarpack _zeroline _no_nc_override`; the driver adds `_padldb<N>` and `_packb-<name>`. Letting the driver name the `results_*.csv` would rename result files of padldb and packb runs, a change the spec forbids.
4. **The `--tmin` refusal is already tested on both sides:** driver `test_ref_protocol.py:374`, shell `:700`. The spec's "the test checks only the shell copy" is wrong.

What is still true: the protocol default, the `--tmin` rule and the log suffix are each written in two or three places (driver, shell, Makefile), and a new flag that changes the name means editing all of them.

Options for the user (none done):
- **A. Leave as is.** The copies are small, each has a test, and the shell and Makefile run before the driver exists. Close as `wontfix`; record an ADR so later reviews skip it.
- **B. Driver `--print-log-suffix` for the Makefile only.** Deletes about 25 Makefile lines; renames `results_*.csv` for padldb/packb runs (check that nothing reads those names first); needs a board run.
- **C. One table of suffix parts** (a small data file) read by the driver build, the Makefile and the shell. Removes drift without moving the decision; larger change.

Closure criteria 1-3 of the first draft are not met and not attempted.
