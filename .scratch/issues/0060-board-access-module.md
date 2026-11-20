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

## 5. Decisions (2026-11-20, by the agent, no grilling)

- **The seven `board_ps` copies were identical** apart from the name of the test hook variable (`ISSUE46_PS_CMD`, ..., `YOLO_SWEEP_PS_CMD`). The spec's "all different" was too strong.
- **Names:** every function in the module has the prefix `board_`: `board_ssh`, `board_rsync`, `board_ps`, `board_busy`, `board_wait_idle`, `board_top_read`, `board_top_summary`, `board_top_trim`, `board_one_reading`, `board_idle_gate`. Callers pass the board name as an argument (the #54_ref copy used a global).
- **Two kinds of idle check stay two functions, not one.** `board_busy` reads `ps` (46-51, yolo, parallel). `board_idle_gate` reads `top` (#54_ref, #55) and records readings in `${SERIES}`. `board_wait_idle` is the #83/#94 form (jump host, user `jlei`, polls every 10 min). Merging them would change thresholds and logs.
- **Test hooks:** one pair, `BOARD_PS_CMD` and `BOARD_TOP_CMD`, replaces the seven per-script `*_PS_CMD` and `ISSUE55_TOP_CMD`. Tests were renamed; their assertions did not change.
- **Series folder naming left alone.** It is one `date +%Y-%m-%d_%H%M%S` call; a helper's interface would be as big as the line (deletion test: nothing concentrates).
- **Not moved, because their connection form differs and no board was used to check a change:**
  - `run_hector_gemm.sh` (`on_board`, `busy`: login shell, key login, stdin passed through, no host-key option, user `USER_ON_BOARD`);
  - `run_issue112_handover.sh` (`hop`, `board`);
  - `probe_issue95.sh` (`on_board`, user fixed to `jlei`);
  - `ref_baseline/run_direct_check.sh` (`ssh_board`, `rsync_board`, host `fej`);
  - `run_issue83_pack_split.sh`, `run_issue85.sh` (2 ssh options each);
  - `compile_macro_bench.sh` and `ref_baseline/run_ref_baseline.sh` (the entry points; 7 and 5 ssh options).

## 6. Result

- Host tests before: 222 passed, 2 failed, 3 errors (14 min 28 s). After: **233 passed, 2 failed, 3 errors**. The 5 problems are the same, all in `test_parallel_gemm.py`, present before this change. The 11 new passes are `test_board_access.py`.
- Removed from the run scripts: 7 `board_ps`, 2 `busy`, 2 `busy_others` + 2 `wait_idle`, `board_ssh` (x2), `board_rsync`, `top_read`, `top_summary` (x2), `trim_sample`, `one_reading` (x2), `idle_gate` (x2). 22 files, +207 / -325 lines before the new test file.

## 7. Closure criteria

- [ ] No `board_ps`, `on_board`, `hop`, `board_ssh` or idle-gate copy left outside `board_pin.sh`. **Not met:** `on_board` (hector, probe_issue95), `hop` (#112), `ssh_board` (direct_check) remain; see section 5.
- [ ] `StrictHostKeyChecking=no` appears only in `board_pin.sh`. **Not met:** still in `compile_macro_bench.sh`, `run_ref_baseline.sh`, `run_direct_check.sh`, `run_issue112_handover.sh`, `run_issue83_pack_split.sh`, `run_issue85.sh`.
- [ ] Every `--dry-run` plan byte-identical to before. **Not proven by a diff.** The existing dry-run tests of the changed scripts pass with only the hook names changed.
- [x] Full host pytest passes. The 2 failures + 3 errors in `test_parallel_gemm.py` were the conda Python lacking `xdsl`, not code (found 2026-11-20 while on #117): with `.venv/bin/python`, `test_parallel_gemm.py` and nine other `xdsl`-dependent files pass, 147 of 147. Tests of the changed scripts: 233 passed in the conda Python.
- [ ] One real launch through the module to jbpi, correctness only. **Not done:** no board run in this session.
