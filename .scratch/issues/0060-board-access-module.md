# Board access module: move ssh, the other-users check, the idle gate and series naming into board_pin.sh

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/116

**Parent:** #115, codebase architecture optimization.
**Role / Label:** `needs-triage`. Spec, 2026-11-20. Strength: **Strong**.
**Status:** open. Not started. No board runs needed (no timing change).
**Branch:** cut from `pure-xdsl` when work starts, named after this title.

---

## 1. Problem

Every board script reaches the board through its own copy of the same code:

- `board_ps()` is defined in 7 scripts (`run_issue46/48/49/50/51.sh`, `run_yolo_sweep.sh`, `run_parallel_gemm.sh`). No two copies are the same.
- The two-step ssh (`sshpass ... ssh $REMOTE_HOST "ssh $board '...'"`) also appears as `hop()` (`run_issue112_handover.sh:29`), `on_board` (`run_hector_gemm.sh:92`, `probe_issue95.sh:28`), `board_ssh` (`run_issue55.sh:201`), and inline in `run_issue54_ref.sh:87`, `run_issue94.sh:68`, `ref_baseline/run_direct_check.sh:65`. `StrictHostKeyChecking=no` appears in 17 scripts.
- The idle-board gate has 4 versions: `run_issue55.sh:262/288`, `run_issue54_ref.sh:119/144` (comment: "as run_issue55.sh"), `run_issue83.sh:49`, `run_issue94.sh:73`.
- Each script has its own test hook to fake ssh (`YOLO_SWEEP_PS_CMD`, `ISSUE50_PS_CMD`, ...; 7 scripts), each tested separately.

`board_pin.sh` is already sourced by 17 scripts, but it covers only pinning, provenance and `OTHER_USERS_AWK`.

**Cost today:** a change to the idle-board rule (house rule since 2026-10-22) or to the jump host means editing up to 20 files, and a missed copy runs with the old rule.

## 2. What to build

Deepen `board_pin.sh` into the **Board access module**. Scripts call it; they no longer carry their own copies.

It covers:

1. **Run a command on a board**, direct or through the jump host (`--direct` vs default, as `docs/agents/execution.md`).
2. **Copy files to and from a board** (rsync over the same route).
3. **Read the board's processes** and **name other users' busy jobs** (today's `board_ps` + `busy`, using `OTHER_USERS_AWK`).
4. **Idle gate**: wait until two readings a set gap apart both show no other user; give up after a set time. One version, from `run_issue55.sh` (the most complete).
5. **Series folder naming**: `logs/<name>_<YYYY-mm-dd_HHMMSS>`.
6. **One fake-ssh hook** for every test (one environment variable), replacing the 7 per-script hooks.

The exact function names and arguments are settled in grilling, before code.

## 3. Out of scope

- Changing any rule (idle gap, wait time, core 4, other-users threshold). Same behaviour, one place.
- Deleting scripts of closed issues. They switch to the module but stay.
- Plan files and a series runner (that is #119).

## 4. Tests

- One test file for the module (extend `test_board_pin.py`): direct and jump-host command lines, the busy-user parse, idle gate pass / wait / give-up with the fake hook.
- Every `test_run_*.py` that sets a `*_PS_CMD` hook switches to the shared hook. Their assertions on `--dry-run` output stay unchanged.

## 5. Closure criteria

- [ ] No `board_ps`, `on_board`, `hop`, `board_ssh` or idle-gate copy left outside `board_pin.sh` (grep).
- [ ] `StrictHostKeyChecking=no` appears only in `board_pin.sh` (and docs).
- [ ] Every `--dry-run` plan byte-identical to before (diff saved in this issue).
- [ ] Full host pytest passes (run with no board series active).
- [ ] One real launch of `compile_macro_bench.sh` through the module to jbpi, correctness only (busy board allowed).
