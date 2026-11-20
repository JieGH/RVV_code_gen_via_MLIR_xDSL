# Board access module: move ssh, the other-users check, the idle gate and series naming into board_pin.sh

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/116

**Parent:** #115, codebase architecture optimization.
**Role / Label:** `needs-triage`. Spec, 2026-11-20. Strength: **Strong**.
**Status:** open. Module and 17 scripts done, tested, smoke-tested on jbpi (branch `board-access-module`). One closure box open: the two entry points keep their own ssh (section 5).
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
- **Second step (2026-11-20, same day): the remaining scripts moved too.** `board_ssh` got a third test hook, `BOARD_SSH_CMD` (`<hook> <board> <command...>`, stdin passed through); `board_ps` now calls `board_ssh`. Converted: `run_hector_gemm.sh` (its `HECTOR_GEMM_ON_BOARD` hook is now `BOARD_SSH_CMD`), `run_issue112_handover.sh`, `probe_issue95.sh`, `ref_baseline/run_direct_check.sh`, `run_issue83_pack_split.sh`, `run_issue85.sh` (the last two lost their `busy_others` copy and use `board_wait_idle`, its message to stderr because `launch` prints a path on stdout).
- **Route differences accepted.** The key-login scripts (hector, probe95, direct_check) now go through `sshpass -p ""` and add `StrictHostKeyChecking=no`; `board_rsync` is `rsync -av`, so direct_check's and #112's rsync output is more verbose. Checked on jbpi: key login works through both.
- **Entry points moved too (third step, user said "finish 116").** `compile_macro_bench.sh` and `ref_baseline/run_ref_baseline.sh` keep thin wrappers `ssh_board`/`rsync_board` over `board_ssh`/`board_rsync`. The jump route's `-t` on both hops stays: `BOARD_SSH_TTY=1` (module switch) for `ssh_board`; `ssh_board_no_tty` does not set it. `RSYNC_TARGET` is now the module's `@T@` placeholder; messages use `RSYNC_LABEL`. Accepted: `ConnectTimeout=15` is added, and `run_ref_baseline.sh`'s rsync is `-av` instead of `-a`.
- **Idle wait 10 min -> 5 min (user, 2026-11-20).** `board_wait_idle` polls every 5 min and says so. The `--wait-min` defaults of `run_hector_gemm.sh` and `run_yolo_sweep.sh` (10) were not changed. One plain `ssh fej "rm -rf ..."` line in `run_direct_check.sh` is not a board call.

## 6. Result

- Host tests of the run scripts and the module (`test_board_access.py`, `test_run_*.py`, `test_*issue*.py`): **199 passed** after the second step, plus 1 new hook test (12 in `test_board_access.py`). The first step's full run was 233 passed; the 2 failures and 3 errors there were the missing `xdsl` in the conda Python, not code.
- Dry-run plans, old copy vs new, byte-identical (times masked): `run_issue83_pack_split.sh`, `run_issue85.sh`. `run_hector_gemm.sh`'s plan is covered by its test file (passes). `run_issue112_handover.sh` and `probe_issue95.sh` have no `--dry-run`; `run_direct_check.sh` was not diffed.
- Entry-point run (jbpi, quiet, core 4): `compile_macro_bench.sh --jump --config config_correctness.json --no-nc-overrides --cpu 4 --tmin 0.2`, exit 0, every row PASS, synced and fetched through the module (`logs/run_2026-11-20_215259`, correctness only, no timing quoted).
- Full host suite in `.venv/bin/python`: 672 passed, 1 failed (a double space my tty change left in the remote command; fixed, 13 module tests pass). Not re-run in full after that fix and the 5-minute change.
- Real board (jbpi, no other user, 2026-11-20), through the jump host with key login and empty `REMOTE_PASS`: `board_ssh jbpi hostname` -> `jbpi`; `board_ps`, `board_busy` (empty); `board_rsync` of a scratch folder to `~/issue116_smoke/`, `board_ssh` read it back and removed it. Correctness only, no benchmark launch.

## 7. Closure criteria

- [x] No `board_ps`, `hop`, `ssh_board` or idle-gate copy of the ssh/idle logic left outside `board_pin.sh`. Remaining one-line wrappers (`ssh_board`, `rsync_board`, `on_board`) only call the module.
- [ ] `StrictHostKeyChecking=no` appears only in `board_pin.sh`. **One line left:** `ssh -o StrictHostKeyChecking=no fej "rm -rf ..."` in `run_direct_check.sh`. It is a call to the jump host, not a board, so it is not in the module's scope.
- [x] `--dry-run` plans unchanged: diffed for #83 pack-split and #85, test-covered for hector (section 6).
- [x] Host tests pass (section 6).
- [x] One real launch through the module to jbpi, correctness only (section 6).
