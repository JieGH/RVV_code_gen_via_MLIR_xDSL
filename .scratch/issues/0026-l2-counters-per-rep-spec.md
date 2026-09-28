# Spec: L2 and prefetch counters per rep, fast reps against slow reps (for #41)

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/42

**Engineering report (#31-#44, all issues together):** [issues-31-43-engineering-report.pdf](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-31-43-engineering-report.pdf) ([LaTeX source](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-31-43-engineering-report.tex)), 2026-09-28.

**Status:** ✅ Implemented 2026-09-27 (see "Done" at the end); results written into #41 §8.
**Brief:** [issue-42-brief.pdf](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issue-42-brief.pdf) (what was built, the checks, the review, the first results).
**Parent:** #41 ([[0025-b3-slow-state-root-cause]]). This spec covers #41's plan steps 1-3 (§4). Steps 4
(`ldb` padding, tied to #38) and 5 (repeat on `jbpi2`) stay in #41.
**House rules:** #39's four rules, as enforced in code since f87ee43 (see [[0024-benchmark-house-rules]], "Enforced in
code and tests"), apply to everything here. Where this spec touches a rule it says so.

## Problem Statement

B3 (64×384×384) runs at two speeds. Every run starts fast and then, after 0.1-0.5 s (vector packer) or 0.5-8 s
(scalar packer), drops into a state about half as fast. In that state both the packing and the microkernel slow
down while the clock stays the same, so the core is waiting on memory (#36). The main suspect is the shared L2
(512 KiB, cores 4-7) or the hardware prefetcher, but today there is no way to see either one:

- The counter group the instrumented build opens holds only generic events. On the X60, every generic L2
  (`LL-*`), cache-reference, cache-miss and prefetch event opens without error but always reads 0 (#41 §3). Only
  `cycles` and `L1D-read-miss` really count.
- The per-rep log (`--save-reps`) gives each rep's time and its pack/compute cycles, but no cache counts. So a
  fast rep cannot be put next to a slow rep to see what changed in the memory system.
- Nothing records what the other cores in the same L2 cluster were doing during a launch, so a noisy neighbour
  on cores 5-7 cannot be ruled in or out.

## Solution

1. Find the X60's own (raw) event codes for L2 access, L2 miss, L2 refill/writeback and the prefetcher, and
   check on the board that each one counts sensibly: few L2 misses on a 64 KiB stream, about one per 64 B line
   on a 4 MiB stream.
2. Let the instrumented build open a list of raw events next to its usual counter group, and write each raw
   event's pack and compute count for every rep into the per-rep log.
3. Let `reps_report.py` put fast reps and slow reps side by side for every counter column, per case, so "slow
   reps have N× the L2 misses of fast reps for the same work" (or "they don't") can be read straight off.
4. Record each core's busy share across a launch in the run's provenance, so a core-2 against core-4 run (and
   what ran on cores 5-7) can be judged.

With this, #41's question "is the slow state in the L2 / prefetcher?" gets a yes or no from logged numbers.

## User Stories

1. As the researcher, I want a written table of the X60 raw event codes for L2 access, L2 miss, L2 refill,
   L2 writeback and prefetch, with where each code came from, so that I don't have to find them again.
2. As the researcher, I want each code in that table marked as "checked on the board" or "does not count",
   so that I only trust events that really count.
3. As the researcher, I want an on-board check that streams 64 KiB and 4 MiB and asserts each chosen raw event
   gives sensible counts, so that a firmware or kernel update that breaks the events is caught.
4. As the researcher, I want the on-board check to skip cleanly when `jbpi` is unreachable, so that the host
   test run stays green offline.
5. As the researcher, I want to pass a list of raw events to the driver by name and code, so that I can try
   candidate codes without regenerating the project.
6. As the researcher, I want a short named preset for the checked X60 L2 events, so that the usual run does
   not need a long list of hex codes on the command line.
7. As the researcher, I want `compile_macro_bench.sh` to pass the raw event list to the driver, so that I can
   run it through the usual entry point with the house rules (core 4, `--tmin 10`, several launches).
7a. As the researcher, I want the raw counts to land in the per-rep log that `compile_macro_bench.sh` now writes
   by default (house rule 2), so that asking for raw events needs no extra flag.
7b. As the researcher, I want `--raw-events` together with `--no-save-reps` refused while parsing arguments,
   with a message saying why, so that I never pay for counters that go nowhere.
8. As the researcher, I want the per-rep log to gain a pack column and a compute column for each raw event,
   so that I can see whether the extra misses are in packing, in the microkernel, or both (both slow down in
   #36).
9. As the researcher, I want the new columns named after the event names I gave, so that the log is readable
   without looking up codes.
10. As the researcher, I want the per-rep log's existing columns to stay the same and in the same order, so
    that `reps_report.py` and older logs keep working.
11. As the researcher, I want the raw event columns to read `nan` when the raw group could not be opened, so
    that a missing count is never mistaken for zero.
12. As the researcher, I want a rep's raw counts to read `nan` when its counter window was shared out
    (multiplexed), so that undercounts never reach the log.
13. As the researcher, I want the run overview to say which raw events were asked for and whether they opened,
    so that a launch log tells me what it measured.
14. As the researcher, I want the uninstrumented build to refuse or clearly ignore the raw event list, so that
    a speed number never carries counter-read cost (house rule: speed numbers come from `INSTRUMENTED=0`).
15. As the researcher, I want the usual 10-event counter group and its per-case perf log to keep working
    unchanged when raw events are added, so that earlier results stay comparable.
16. As the researcher, I want the raw events kept in their own group, so that a raw event the board rejects
    does not take the usual counters down with it.
17. As the researcher, I want `reps_report.py` to show, per case, the mean per rep of each counter column over
    fast reps and over slow reps, and their ratio, so that the answer to "do slow reps miss more in L2?" is one
    line.
18. As the researcher, I want the fast/slow split in that comparison to be the same split `reps_report.py`
    already uses for its slow-rep share, so that the two views agree.
19. As the researcher, I want rep 0 left out of the fast/slow comparison, as it is left out of the split, so
    that the cold rep does not bias the fast group.
19a. As the researcher, I want the fast/slow comparison to pool each case's reps over all the launches given
    and to carry the same "fewer than 3 launches" WARNING the launch table already prints (house rule 2), so
    that a one-launch comparison is never read as a result.
20. As the researcher, I want cases with one speed to show "one speed, no slow reps" in the comparison instead
    of a meaningless ratio, so that controls like B2 and Reuse24 read cleanly.
21. As the researcher, I want the comparison to cover the existing `Pack_Cycles` / `Compute_Cycles` columns as
    well as the new raw ones, so that cycles and misses sit side by side.
22. As the researcher, I want `reps_report.py` to handle per-rep logs with and without raw columns (and gzipped
    ones), so that old and new logs can be read in one call.
23. As the researcher, I want each launch's provenance to record every core's busy share over the launch, from
    the kernel's per-core time counts before and after, so that I can see whether cores 5-7 were busy while B3
    ran on core 4.
24. As the researcher, I want that busy-share record for cores 0-3 too, so that a core-2 run against a core-4
    run can be judged the same way. (The comparison core is 2, never 0: scripts refuse `--cpu 0` without
    `--allow-core-0`, house rule 1, and this work has no reason to pass it.)
24a. As the researcher, I want `provenance.txt` to also record the raw event list asked for, and
    `reps_report.py`'s per-run conditions line (house rule 4) to show it and the other cores' busy share, so
    that every comparison says what was counted and how quiet the cluster was.
25. As the researcher, I want the busy-share record to be skipped (with a line saying so) on a dry run, so that
    `--dry-run` stays side-effect free.
26. As the researcher, I want a config for the #41 cases (B3 both packers, B3 with K=256, B2 and Reuse24 as
    controls), so that the fast/slow comparison run is one command.
27. As the researcher, I want the results of that run committed as logs and written up in #41, naming build,
    protocol and core, so that the numbers can be quoted under the house rules. `test_house_rules.py` fails if
    the write-up cites a `run_*` folder that is not committed (house rule 3).
28. As a future agent, I want the on-board check and the code table to say which kernel and firmware they were
    checked on, so that I know when to check them again.

## Implementation Decisions

- **Code discovery is a research step with a written output.** Sources, in order: the SpacemiT K1 / X60 manuals,
  the vendor kernel's PMU driver and device tree (`riscv,event-to-mhpmcounters` / `riscv,raw-event-to-mhpmcounters`),
  and the OpenSBI PMU tables in the vendor firmware. The output is a table (name, raw code, source, checked yes/no,
  64 KiB and 4 MiB counts per 64 B line), kept in #41 and next to the preset in the counter header.
- **Raw events open as `PERF_TYPE_RAW`** with the same settings as the usual group (user space only, group
  read with enabled/running times).
- **Separate group.** The raw events form a second counter group beside the usual 10-event group, of variable
  size (up to a small fixed maximum, set by the number of X60 hardware counters). If the raw group cannot open,
  only the raw columns go to `nan`; the usual group is not touched.
- **Same windows.** The existing pack and compute region windows read both groups, so every raw event gets a
  pack count and a compute count per rep. The reads stay outside the wall-clock timer, as today. The "reject a
  window that was shared out" rule applies to the raw group on its own.
- **Driver interface.** A new driver flag `--raw-events <list>`, where each item is `name=0xCODE` or the preset
  name for the checked X60 L2 events. Instrumented build only; the uninstrumented build rejects the flag with a
  message saying why. `compile_macro_bench.sh` gains the same flag and passes it through, and names it in the
  run overview. Since the per-rep log is on by default (house rule 2), the flag adds no other switch; the script
  refuses `--raw-events` with `--no-save-reps` while parsing arguments, next to its other refusals
  (`pin_check` and friends), before anything runs.
- **Per-rep log schema.** The existing 14 columns are unchanged. When raw events are asked for, each event adds
  `Pack_<name>` and `Compute_<name>` at the end, in the order given. Without raw events the file is exactly as
  today. The log name does not change (the instrumented build's name already differs from the uninstrumented one).
- **Per-case perf log.** Unchanged in this spec. Raw events are per-rep only.
- **`reps_report.py`.** A new fast/slow table per case: for each counter column present (the two cycle columns
  and any raw ones), mean per rep over fast reps, over slow reps, and slow/fast ratio. It uses the existing
  two-means split and leaves rep 0 out, pools reps across the launches given, and warns for cases seen in fewer
  than 3 launches exactly as the launch table does. One-speed cases print as such. The existing tables, the
  per-run conditions lines and their core-0 WARNING are unchanged apart from the additions below.
- **Busy share.** `compile_macro_bench.sh` reads the board's per-core time counts at the start and end of a
  launch and writes each core's busy share into `provenance.txt`, together with the raw event list asked for.
  Not per case. `reps_report.py`'s conditions line (`run_conditions`) reads both and prints them after board,
  build, protocol and core.
- **Config.** One new config for #41's cases (B3 vector and scalar packers, B3 with K=256, B2, Reuse24), run
  with the house rules: core 4 (and core 2 for the cluster comparison; never core 0), `--tmin 10`,
  `--no-nc-overrides`, at least 3 launches per setting. The packers need separate launches, so they alternate
  A-B-B-A-A-B (rule 2); the cases (B3, B3 K=256, B2, Reuse24) go inside one launch. The instrumented build is
  used, so these runs find where time goes and are never quoted as speed (rule 4).

## Testing Decisions

- Good tests here check what a user of the tool sees: the flags it takes, the columns and values in the logs,
  the lines in the overview and the report. They do not check how the header stores its state.
- **Driver seam** (prior art: `test_ref_protocol.py`, the real driver built on the host against the stub
  kernel). Covers: `--raw-events` parsing (names, codes, the preset, bad input), the new column names and their
  order, `nan` in the raw columns when the group cannot open (the host's `perf_event_paranoid` usually blocks
  it, which is the case to test), the overview line, the uninstrumented build refusing the flag, and the
  per-rep log being exactly as today without the flag. Script pass-through is tested with `--dry-run`, as for
  `--save-reps` (whose default-on behaviour is already covered there).
- **Counter header seam** (prior art: `test_perf_counters.py`'s host harness with fake kernel read buffers).
  Covers: decoding a raw group read of any size up to the maximum, per-window deltas, and a shared-out window
  rejected for the raw group without touching the usual group.
- **Report seam** (prior art: `test_reps_report.py`, functions over hand-made rows and the command line over
  files, including its hand-made `provenance.txt` for the conditions lines). Covers: the fast/slow means and
  ratio per column, rep 0 left out, pooling across launches and the under-3-launches WARNING, one-speed cases,
  logs with and without raw columns, gzipped logs, and the raw event list and busy share in the conditions
  line (absent from old provenance files: printed as `unknown`, no failure).
- **On-board check** (prior art: the on-board test in `test_perf_counters.py`, skips when `jbpi` is not
  reachable through `fej`). Replaces the one-off probe: opens the chosen raw events, streams 64 KiB and 4 MiB,
  and asserts L2 misses per 64 B line are low on the first and about one on the second, and that the group is
  never shared out.
- The busy-share record is tested at the script level with `--dry-run` (it says it is skipped) and with a fake
  per-core time file, following `test_board_pin.py`. The `--raw-events` + `--no-save-reps` refusal is tested the
  same way as the core-0 refusal there: non-zero exit, a message saying why, and nothing run.
- **House rules stay green:** `test_house_rules.py` must pass after this work; any doc line added for the new
  flag must not show `--cpu 0` without `--allow-core-0`, and any run folder cited in #41 or #42 must be committed.
- Tests are written first (red, then green), one behaviour at a time.

## Out of Scope

- Changing the packer, the blocking, or the cache model to avoid the slow state. That waits until the cause is
  known (#41 §6).
- `ldb` / `lda` / `ldc` padding (#41 step 4, #38).
- Repeating the runs on `jbpi2` (#41 step 5).
- Raw events in the per-case perf log or in the summary.
- dTLB misses per rep (only if the L2 turns out not to be the cause, #41 §5).
- Fixing the usual group's other generic events if they also read 0 on the X60 (worth a separate issue if the
  on-board check shows it).

## Further Notes

- If the SBI firmware does not pass raw events through, step 1's written result is "no raw L2 events on this
  firmware". The per-rep plumbing is still worth having for any event that does count, but #41 then moves on to
  the core-2 against core-4 comparison and the busy-share record.
- Numbers from this work come from the instrumented build and are for the fast/slow comparison only, not for
  quoting speed. Name the build, protocol and core with every number, and commit the logs before quoting them
  (house rules, #39; enforced since f87ee43 by the scripts, `reps_report.py` and `test_house_rules.py`).
- The raw codes and the on-board check depend on the board's kernel (6.6.63 on `jbpi` as of 2026-09-27) and
  firmware; write both down with the table.

## Done (2026-09-27)

- Raw event codes found, checked, and tabled (`docs/metrics.md`, "Raw events"; #41 §8.1; `0026-raw-probe.c`/`.txt`).
- `perf_counters.h`: an optional raw group beside the usual one (`perf_gemm_open_raw`, `PERF_X60_L2_PRESET`).
- Driver: `--raw-events name=code,...|x60-l2`, a `Pack_`/`Compute_` pair per event in `reps_*.txt`, `nan` rules,
  refusals (uninstrumented build, no `--save-reps`, bad lists), overview line.
- `compile_macro_bench.sh`: `--raw-events` (refused without `--instr` or with `--no-save-reps`), provenance lines
  for the raw events and each core's busy share **during each timed run** (changed after review from "over the
  launch", which counted the board build).
- `reps_report.py`: fast/slow table per counter, pooled over launches; conditions line shows raw events and busy share.
- On-board check in `test_perf_counters.py`; config `config_issue41_l2_counters.json` and its dataset.
- Review (two axes) findings fixed: shared fast/slow split in `reps_report.py`; busy share per timed run; firmware
  version noted as unreadable without root; per-row source/checked columns in the table.
