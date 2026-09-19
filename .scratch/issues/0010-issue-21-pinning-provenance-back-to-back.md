# Issue #21 (local): pin both sides to one core, record the conditions, run back to back

**Role / Label:** `done` — implemented and checked on the host and by read-only probes of the board; GitHub #21 closed. **The issue's own verification (B1 three times pinned) was not run.**
**Status:** 🟡 the mechanism ships; whether it tightens the spread is unmeasured.
**Target:** `jbpi` (BananaPi M2 Pro, 8 cores, VLEN=256), reached via `fej`.
**Primary Sources:**
- GitHub #21, child of #11. Ranked "Tier 1, the largest hole" in [[0009-issue-11-midpoint-the-result-flipped]] §5.
- Commits `b8f6b27` (the work) and `95f7302` (the default command), on `feat/combined`.
- `tests/pure_xdsl_blis_rvv/board_pin.sh`, `run_head_to_head.sh`, `test_board_pin.py`; edits to `compile_macro_bench.sh`, `ref_baseline/run_ref_baseline.sh`, `ref_baseline/compare_ref_baseline.py`; `docs/agents/execution.md`.

---

## 1. What the issue asked for, and what was done

| the issue said | done |
|---|---|
| `taskset -c N` on the board command in both scripts | Yes. Ours: `taskset -c N make run` (the driver inherits it). Reference: `taskset -c N ./run_gemm.sh` (it starts `test_gemm.x` directly). Only the timed step is pinned, not the build. |
| capture `scaling_governor`, `/proc/cpuinfo`, `gcc --version` into `provenance.txt` on both sides | Yes, plus `g++ --version` (the driver is built with `g++`, `model_level.c` with `gcc`), the affinity `taskset` reports under the pin, host/kernel, load average and a timestamp. Ours writes `<run dir>/provenance.txt`; the reference block goes into its existing `provenance.txt`. A probe the board cannot answer reads `unavailable`. |
| run both sides back to back in one session | `run_head_to_head.sh`: ours (`--no-nc-overrides`), then the reference (`--bracket --no-compare`), then the comparison on exactly that pair. |
| verify: B1 three times pinned, spread tighter than cross-day | **Not run.** Needs board time. |

Beyond the issue, on purpose:
- `compare_ref_baseline.py` prints a `Gap` and a `Pinning` line above the table and a `WARNING` when the runs are more than 1 h apart, a side was not pinned, the cores differ, or a run predates provenance (reported as *pinning unknown*, not guessed). Without it, a bad pair looks the same as a good one. On the real 09-19 / 09-18 pair it says 26.1 h apart and pinning unknown.
- `--cpu N`, `--no-pin` (to reproduce the pre-#21 numbers), a check before syncing that the board can pin to the core, and `--no-compare` on the reference script (the wrapper runs the comparison itself).

## 2. Which core

The script default is **core 4**, one constant (`BENCH_CPU_DEFAULT` in `board_pin.sh`) read by all three scripts, so the two sides cannot get different defaults. Read-only probes of `jbpi` on 2026-09-19:
- `taskset` is installed; governor is already `performance` on all 8 cores.
- Every device interrupt in `/proc/interrupts` is delivered to CPU0, which is why the default avoids it.
- All cores were ~100% idle. Load average sits at 2.0 with nothing on the CPUs (uninterruptible tasks, not a busy core).
- The core layout (clusters, shared caches) was **not** checked: 4 is a pick among the cores that take no interrupts, not a topology-based one.

**The default command uses `--cpu 0`** (user's choice, `95f7302`). That is the interrupt core, so expect a wider spread there than on core 4. If the spread on core 0 is no better than the old unpinned runs, that is the first thing to change.

## 3. How it was checked

- 35 new tests in `test_board_pin.py`, none needing the board; 125 pass in `tests/`.
- The provenance probe was run on the real board through the jump host: affinity list `4`, both compilers, governor, cpuinfo all came back.
- `run_ref_baseline.sh` was run end to end against stub `ssh`/`rsync`: the command sent is `taskset -c 4 ./run_gemm.sh …`.
- `compile_macro_bench.sh` was **not** run end to end against stubs (it deletes and regenerates the tracked `pj_gen/`). Its pinned line, `${FORCE_NC_ENV}${PIN_PREFIX}make run …`, is checked by reading.
- Code review (two agents). One claim rejected (the reference script's ssh does not force a tty). Fixed: the truncated `-h` output, the three-times-repeated flag checks (now `pin_check`), probe fallbacks that the comment promised but the code did not have, `g++`, a test tying the shell format to the Python parser, an unfriendly abort in the wrapper.

## 4. What this does not establish

- **The pin reaches the timed process** is argued from the code (children inherit affinity; the driver has no thread or affinity code) and the probe shows `taskset` works on that core. Nothing records the affinity of the process that did the timing. If that matters, the driver should log its own `sched_getaffinity`.
- **The comparison checks the requested pin**, not the affinity line.
- **The gap** is between the two run-dir stamps: ours is stamped when its logs are fetched (end of run), the reference when its script starts. It bounds the separation from below; the midpoints of the two measurements are further apart by about half of each run.
- **One pair cannot cancel drift**, only bound it. The wrapper always runs ours first. The stdin pipe of the probe through the two-hop jump path was tested by hand once, not in a test.
- `cron_bench_run.sh` is now pinned to core 4 as a side effect of pinning being the default.
- The old `run_*` and `refbase_*` dirs have no provenance; they will always read *pinning unknown*.

## 5. Next

1. Run the default command once (`docs/agents/execution.md`, "Head-to-head"): both sides on core 0, one pair, no `WARNING` lines expected.
2. The issue's check: the same command three times (there is no B1-only filter), compare the B1 row across the three runs against the cross-day spread in [[0009-issue-11-midpoint-the-result-flipped]]. Repeat on core 4 if core 0 is noisy.
3. Then #11's write-up can quote a number that was taken under stated conditions.
