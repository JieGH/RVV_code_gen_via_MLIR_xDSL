# Execution: running the pipeline against the RISC-V board

Two connection modes reach the board, `jbpi`/`jbpi2` (BananaPi M2 Pro). Both accept the same `--riscv-user`/`--riscv-ip`/`--riscv-port` flags:

- **Direct**: SSH straight to the board. `<script> --riscv-user <user> --riscv-ip <ip> --riscv-port <port> ...`
- **Jump-host**: SSH through an intermediate host (`fej`) to reach the board, used when the board isn't directly reachable from this machine. Set `REMOTE_HOST`, `REMOTE_PORT`, and (if needed) `REMOTE_PASS` as env vars, and add `--jump`.

## House rules for a benchmark result (issue #39)

The #31-#34 audit found conclusions resting on setups the project already knew were weak: 32 launches on core 0 the day core 0 was measured 5-10% slower, differences of 0.3-1.6% read from single launches while the same binary moved up to 1.7% between launches, and numbers quoted from runs whose logs were never saved. From now on:

1. **Never pin to core 0.** The board sends every device interrupt there. Use core 4 (`BENCH_CPU_DEFAULT`, the default of `compile_macro_bench.sh` and `run_ref_baseline.sh`) or another non-zero core, and name the core next to every number. **Enforced:** every script refuses `--cpu 0` while parsing its arguments unless `--allow-core-0` is also given (only to reproduce an old core-0 run, e.g. the overnight sweeps of 19-20 Sep), and no script defaults to core 0 (`board_pin.sh:pin_check`; tested in `test_board_pin.py`).
2. **One launch is not a result.** When the difference you want to report is under about 3%, put both settings inside **one** launch (two rows of one dataset, e.g. one label per setting with a per-label `nc` override) and repeat the launch at least three times, alternating the order (A B B A A B) when the settings need separate launches (a packer, a build). Report the spread over launches with the number. **Built in:** `compile_macro_bench.sh` saves every rep by default (`--no-save-reps` to skip), and `reps_report.py` gives the across-launch average with the lowest and highest launch, and prints a WARNING for any case seen in fewer than 3 launches.
3. **Every quoted number comes from a saved log** under `tests/pure_xdsl_blis_rvv/logs/`, committed, and cited by its run directory. A number from a pasted console, a run on another machine, or a log that was not kept is labelled as such and is not used to decide anything. **Tested:** `test_house_rules.py` fails when an issue file, report or doc cites a `run_*` / `refbase_*` folder that is not committed.
4. **Say which build, protocol and core** a number came from (`INSTRUMENTED=0`, reference protocol, `--tmin`, `--cpu`). The instrumented build is for finding where time goes, never for a number quoted against the reference. **Built in:** every run folder's `provenance.txt` records them, and `reps_report.py` prints them for each run it reads (with a WARNING for a core-0 run).

## Entry points

- **`tests/pure_xdsl_blis_rvv/compile_macro_bench.sh`**: standalone build & benchmark for the pure xDSL BLIS Project Generator (see `CONTEXT.md`) macrokernel project. Takes `--config <name>.json` (default `config.json`). Activates `.venv` itself; must be run from the repo root. Documented in `README.md`.
- **`tests/pure_xdsl_blis_rvv/run_head_to_head.sh`**: our macrokernel bench and the reference design back to back on one pinned core, then the comparison. The command for a number quoted against the reference (see below).
- **`tests/pure_xdsl_blis_rvv/run_pack_swap.sh`**: vector-pack and scalar-pack GEMM back to back (vector, scalar, scalar, vector) on one pinned core, over B3 and S1 (or `--config config_pack_swap_b2b3b4.json` for B2, B3 and B4), then `compare_pack_swap.py` (issue #26). Takes the connection flags, `--cpu`/`--no-pin`, `--tmin`, `--config` (default `config_pack_swap.json`), `--shipped-nc`, `--dry-run`. Its defaults: `--no-nc-overrides --cpu 4 --tmin 10` (core 4 since issue #39; it was core 0 before). Board time: four runs.
- **`tests/pure_xdsl_blis_rvv/run_scalar_pack_sweep.sh`**: the overnight sweep (every layer of `config.json`, `jbpi` and `jbpi2` taking turns, one run every 30 min, 8 per board) repeated with the scalar packer only (issue #26), then compared layer by layer with the vector-pack overnight runs of 19-20 Sep via `compare_pack_swap.py`. Flags `--no-nc-overrides --cpu 4 --tmin 10` (the 19-20 Sep runs it compares with were core 0: add `--cpu 0 --allow-core-0` to match them), one run at a time under the cron wrapper's lock, a failed run is logged and the sweep goes on. Start it detached: `setsid nohup ./tests/pure_xdsl_blis_rvv/run_scalar_pack_sweep.sh > tests/pure_xdsl_blis_rvv/logs/sweep_scalar.out 2>&1 < /dev/null &`. Takes `--start-at HH:MM`, `--boards`, `--runs`, `--interval`, `--tmin`, `--cpu`, `--vs-from/--vs-to`, `--dry-run`. About 8 h; do not start other benchmarks on the boards meanwhile (manual runs do not take the lock).
- **`tests/pure_xdsl_blis_rvv/run_ref_sweep.sh`**: the same schedule as the two sweeps above (`jbpi` and `jbpi2` taking turns, one run every 30 min, 8 per board, core 4 (`--cpu 0 --allow-core-0` to match the earlier core-0 runs), `--tmin 10`) but for the **reference design alone** (issue #29): `run_ref_baseline.sh --opt LDX --variants 2 --bracket` per run, i.e. the reference's own packers, its LDX kernels (`lda`/`ldb` as runtime arguments) and the `macc` variant, on the ten layers of `ref_baseline/cases.tsv`. Nothing of ours runs. A run that exits non-zero or leaves a row out (`run_ref_baseline.sh` only warns when a build fails) is logged as failed and the sweep goes on. It ends with `compare_ref_sweep.py`: the reference per layer beside our vector-pack overnight runs and the newest scalar-pack sweep. Same lock as the other two. Takes `--start-at`, `--boards`, `--runs`, `--interval`, `--tmin`, `--cpu`/`--no-pin`, `--opt`, `--variants`, `--direct`, `--riscv-user`/`--riscv-port`, `--scalar-sweep`, `--vs-from/--vs-to`, `--no-compare`, `--dry-run`. Do not run `--dry-run` or `run_ref_baseline.sh` while a sweep is going: both rewrite `ref_baseline/generated/`, which a running reference run stages from. Start it detached: `setsid nohup ./tests/pure_xdsl_blis_rvv/run_ref_sweep.sh > tests/pure_xdsl_blis_rvv/logs/sweep_ref.out 2>&1 < /dev/null &`. About 8 h; do not start other benchmarks on the boards meanwhile.
- **`tests/pure_xdsl_blis_rvv/ref_baseline/run_ref_baseline.sh`**: the reference design alone, from `../ext/RVV_ukernels_benchmark`. `run_head_to_head.sh` and `run_ref_sweep.sh` call it; run it directly only to refresh a reference baseline on its own.

## Reference command (jump-host, pure xDSL BLIS macrokernel bench)

```bash
source .venv/bin/activate && REMOTE_PASS="" REMOTE_HOST="fej" REMOTE_PORT="3322" \
  ./tests/pure_xdsl_blis_rvv/compile_macro_bench.sh --jump \
  --riscv-user jlei --riscv-ip jbpi --riscv-port 22 \
  --no-nc-overrides --cpu 4 --tmin 10
```

This runs our side alone, pinned to core 4 (the default; `--cpu 4` only makes it visible in the command). The default command for a benchmark run is the head-to-head one below, which runs this same side and the reference together. `--no-nc-overrides` makes B3 run at the reference's `nc=384`; drop it for the shipped-configuration (`nc=128`) number.

Adjust `REMOTE_HOST`/`REMOTE_PORT`/`--riscv-user`/`--riscv-ip`/`--riscv-port` per the operator's actual jump-host and board; the values above are this project's current defaults, not fixed constants.

## Run protocol and build flags (`compile_macro_bench.sh`)

The reference command above needs no protocol flag: the uninstrumented build times the way the reference design does (issue #27) — no warm-up, reps until `--tmin` seconds, correctness checked after timing. Every run opens with an overview of the settings in effect, each marked `[default]` or with what set it.

| flag | effect |
|---|---|
| *(none)* | uninstrumented build, **reference protocol**, `tmin` 3.0 s. Logs `raw_*_noinstr_refproto.txt` |
| `--tmin <s>` | time budget for the reference protocol. Refused where the reference protocol is not running |
| `--no-ref-protocol` | the older **warm-up protocol** instead: 5 warm-up calls, a scalar-pack pass, then the fixed `iterations` from `config.json`. Logs `raw_*_noinstr.txt` |
| `--scalar-pack` | every timed call packs with the scalar packer instead of the RVV one (issue #26). Logs get a `_scalarpack` suffix (before `_no_nc_override`), so it never mixes with a vector-live run. See `docs/metrics.md`, "Vector pack against scalar pack, end to end" |
| `--zero-pack-lines` | before each pack, clear with `cbo.zero` every whole 64-byte line of the Ac/Bc block the packer is about to write, inside the pack timer (issue #43). AXPY kernels only (the driver refuses it for DOT). Off by default. Logs get a `_zeroline` suffix (after `_scalarpack`, before `_no_nc_override`); the overview's `Pack lines` line and `provenance.txt` say whether it was on. Comparison config: `config_issue43_zero_lines.json`. See `docs/metrics.md`, "Clearing the pack lines" |
| `--no-nc-overrides` | ignore the per-shape `nc` overrides in `config.json` (B3 ships at `nc=128`; the reference runs `nc=384`). The head-to-head configuration; a run without it is the shipped one. Logs get a `_no_nc_override` suffix, so the two never mix (issue #16) |
| `--cpu <n>` | pin the timed run to core `n` with `taskset -c n` (issue #21). Default: core 4, the same default `run_ref_baseline.sh` uses, so both sides time on one core. Not core 0: the board sends every device interrupt there |
| `--allow-core-0` | let `--cpu 0` through (refused otherwise, issue #39); only to reproduce an old core-0 run. Every script takes it, and the wrappers pass it on |
| `--no-pin` | run unpinned, to reproduce the pre-#21 numbers. Refused together with `--cpu` |
| `--instr` | instrumented debug build. Its default is the warm-up protocol, since the reference protocol cannot run there; asking it for `--ref-protocol` is refused. Logs `raw_*.txt`. Never quote against the reference |
| `--save-reps` / `--no-save-reps` | **on by default** (issue #39): write `logs/reps_*.txt` (same suffixes as the raw log; fetched gzipped as `reps_*.txt.gz`, summarised by `reps_report.py`): one row per timed rep, in the order the reps ran, with the row's `mc`/`nc`/`kc`, and in the `--instr` build each rep's pack and compute time and cycles. Written after each shape's timed loop, so it adds nothing to the timing (issues #35, #36). See `docs/metrics.md`, "Per-rep log" |
| `--raw-events <list>` | `--instr` only, and needs the per-rep log (refused with `--no-save-reps`): count the core's own events (`name=code,...`, or the preset `x60-l2`: the X60's L2 and L1D-prefetch events) per rep, as a `Pack_<name>` and a `Compute_<name>` column each in `reps_*.txt` (issue #42). Every launch also records the raw events and each core's busy share over the launch in `provenance.txt`. See `docs/metrics.md`, "Raw events" |
| `--dry-run` | print the run overview and stop, without generating, syncing or running anything |

`compare_ref_baseline.py` defaults to the instrumented log name; point it at a default run with `--raw-name all_raw_<project>_noinstr_refproto.txt` (add `_no_nc_override` for a run made with `--no-nc-overrides`; the fringe column then follows on its own). Details and the measured warm-up effect: `docs/metrics.md`.

Every run writes `<run dir>/provenance.txt`: the pin, the affinity `taskset` reports under it, the CPU governor, the compiler, the load average and `/proc/cpuinfo`, taken on the board just before the build and run. `run_ref_baseline.sh` puts the same block in its own `provenance.txt`. `compare_ref_baseline.py` reads them and prints a `WARNING` when the two sides were not pinned, were pinned to different cores, or were made more than an hour apart; a run from before issue #21 has no provenance and is reported as pinning unknown.

## Head-to-head against the reference design

```bash
source .venv/bin/activate && REMOTE_PASS="" REMOTE_HOST="fej" REMOTE_PORT="3322" \
  ./tests/pure_xdsl_blis_rvv/run_head_to_head.sh --jump \
  --riscv-user jlei --riscv-ip jbpi --riscv-port 22 \
  --cpu 4
```

This is the default command for a benchmark run. It pins both sides to core 4 (never core 0, which takes every device interrupt: see the house rules above). Runs our side (`compile_macro_bench.sh --no-nc-overrides`, the default build and protocol) and then the reference side (`run_ref_baseline.sh --bracket`), both pinned to the same core, and compares exactly that pair. This is the command for a number quoted against the reference; a reference baseline from another day is not comparable (issue #21). Take `--cpu`, `--no-pin`, `--tmin` and the connection flags; `--opt OPT|LDX` picks the reference kernel set. For anything else run the two scripts yourself, then `compare_ref_baseline.py --result-dir ... --runs ... --raw-name ...`.

## Correctness pass

Every row of the shipped `config.json` sets `skip_ref_run`, so a benchmark run logs `Status=SKIP` and checks nothing. The check lives in a separate config, `tests/pure_xdsl_blis_rvv/config_correctness.json`: the same three pinnings (16x15, 32x6, 32x7) with the scalar oracle on, over `datasets/correctness_small.dat` (three small shapes: all full tiles, fringe rows and columns with three K blocks, and one Fringe Tile that is the whole matrix). The scalar oracle (`gemm_ref`) is O(m·n·k), so no shape above B2 (384×384×64) may run under it; `test_correctness_pass.py` fails if a row of `config.json` or `config_correctness.json` turns the check on for a bigger one.

```bash
source .venv/bin/activate && REMOTE_PASS="" REMOTE_HOST="fej" REMOTE_PORT="3322"   ./tests/pure_xdsl_blis_rvv/compile_macro_bench.sh --jump   --riscv-user jlei --riscv-ip jbpi --riscv-port 22   --no-nc-overrides --cpu 4 \
  --config config_correctness.json --tmin 0.2
```

It is the benchmark command with `--config` and `--tmin 0.2` added (the timing is not the point; a short `tmin` keeps it to a few minutes). Read `Status` in `<run dir>/all_raw_*.txt`: `PASS` for all nine rows. `SKIP` means the check did not run; `FAIL` prints the first mismatches.

`PASS` means: the whole `C` our kernels leave (the packers, the microkernel and the fringe path together, after the timed reps) is within a relative error of `max(1e-5, k·1e-6)` of the `C` the scalar oracle (`gemm_ref`) computes. The error is `‖C_ours − C_ref‖₂ / ‖C_ref‖₂` over all `m×n` entries, taken in double. `A` and `B` are uniform random in [0, 1], `alpha = 1`, `beta = 0`. It is a tolerance, not a bitwise match, and it is loose enough that one wrong `k`-step in one entry would slip through; a wrong row, column or vector chunk of a panel would not (at `k = 300` the bound is 3e-4; one wrong `k`-step in one entry is about 5e-5, while a whole wrong 8-row chunk of a 32-row panel is at least about 4e-2, and 5e-1 if it is zeroed). The bitwise check of the packed layout is the pack test below. `beta ≠ 0` from the caller is not exercised: the driver never passes it; the beta=1 microkernel runs only as the later K blocks' accumulate. The run overwrites the tracked `pj_gen/` like any `--config` run: regenerate it from `config.json` afterwards (`cd tests/pure_xdsl_blis_rvv && python3 generate_pure_blis_project.py --config config.json`, then put back the trailing newline the `.mlir` files lose).

The vector packers get their own check, per shipped `RR` (6, 7, 15, 16, 32): `pytest tests/pure_xdsl_blis_rvv/test_pack_correctness.py`. It builds the real packers, compiles them on the board through `fej`, and skips when the board is not reachable.

## One-time dependency builds

Not part of the per-run pipeline above — see `docs/dependencies_setup.md` for building LLVM/MLIR locally and OpenBLAS/BLIS on the board.
