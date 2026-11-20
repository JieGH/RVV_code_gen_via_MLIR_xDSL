# Driver owns the protocol and its log name; the shell and Makefile stop working them out again

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/118

**Parent:** #115, codebase architecture optimization.
**Role / Label:** `needs-triage`. Spec, 2026-11-20. Strength: **Worth exploring**.
**Status:** open. Not started. Needs one board correctness run before merge.
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
