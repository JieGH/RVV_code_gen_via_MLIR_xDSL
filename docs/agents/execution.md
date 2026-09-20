# Execution: running the pipeline against the RISC-V board

Two connection modes reach the board, `jbpi`/`jbpi2` (BananaPi M2 Pro). Both accept the same `--riscv-user`/`--riscv-ip`/`--riscv-port` flags:

- **Direct**: SSH straight to the board. `<script> --riscv-user <user> --riscv-ip <ip> --riscv-port <port> ...`
- **Jump-host**: SSH through an intermediate host (`fej`) to reach the board, used when the board isn't directly reachable from this machine. Set `REMOTE_HOST`, `REMOTE_PORT`, and (if needed) `REMOTE_PASS` as env vars, and add `--jump`.

## Entry points

- **`tests/pure_xdsl_blis_rvv/compile_macro_bench.sh`**: standalone build & benchmark for the pure xDSL BLIS Project Generator (see `CONTEXT.md`) macrokernel project. Takes `--config <name>.json` (default `config.json`). Activates `.venv` itself; must be run from the repo root. Documented in `README.md`.
- **`tests/pure_xdsl_blis_rvv/run_head_to_head.sh`**: our macrokernel bench and the reference design back to back on one pinned core, then the comparison. The command for a number quoted against the reference (see below).
- **`tests/pure_xdsl_blis_rvv/run_pack_swap.sh`**: vector-pack and scalar-pack GEMM back to back (vector, scalar, scalar, vector) on one pinned core, over B3 and S1 (or `--config config_pack_swap_b2b3b4.json` for B2, B3 and B4), then `compare_pack_swap.py` (issue #26). Takes the connection flags, `--cpu`/`--no-pin`, `--tmin`, `--config` (default `config_pack_swap.json`), `--shipped-nc`, `--dry-run`. Its defaults are the default run command's, not `compile_macro_bench.sh`'s: `--no-nc-overrides --cpu 0 --tmin 10`. Board time: four runs.
- **`tests/pure_xdsl_blis_rvv/run_scalar_pack_sweep.sh`**: the overnight sweep (every layer of `config.json`, `jbpi` and `jbpi2` taking turns, one run every 30 min, 8 per board) repeated with the scalar packer only (issue #26), then compared layer by layer with the vector-pack overnight runs of 19-20 Sep via `compare_pack_swap.py`. Same flags as the default run command (`--no-nc-overrides --cpu 0 --tmin 10`), one run at a time under the cron wrapper's lock, a failed run is logged and the sweep goes on. Start it detached: `setsid nohup ./tests/pure_xdsl_blis_rvv/run_scalar_pack_sweep.sh > tests/pure_xdsl_blis_rvv/logs/sweep_scalar.out 2>&1 < /dev/null &`. Takes `--start-at HH:MM`, `--boards`, `--runs`, `--interval`, `--tmin`, `--cpu`, `--vs-from/--vs-to`, `--dry-run`. About 8 h; do not start other benchmarks on the boards meanwhile (manual runs do not take the lock).
- **`tests/pure_xdsl_blis_rvv/ref_baseline/run_ref_baseline.sh`**: the reference design alone, from `../ext/RVV_ukernels_benchmark`. `run_head_to_head.sh` calls it; run it directly only to refresh a reference baseline on its own.

## Reference command (jump-host, pure xDSL BLIS macrokernel bench)

```bash
source .venv/bin/activate && REMOTE_PASS="" REMOTE_HOST="fej" REMOTE_PORT="3322" \
  ./tests/pure_xdsl_blis_rvv/compile_macro_bench.sh --jump \
  --riscv-user jlei --riscv-ip jbpi --riscv-port 22 \
  --no-nc-overrides
```

This runs our side alone (pinned to core 4 by default). The default command for a benchmark run is the head-to-head one below, which runs this same side and the reference together. `--no-nc-overrides` makes B3 run at the reference's `nc=384`; drop it for the shipped-configuration (`nc=128`) number.

Adjust `REMOTE_HOST`/`REMOTE_PORT`/`--riscv-user`/`--riscv-ip`/`--riscv-port` per the operator's actual jump-host and board; the values above are this project's current defaults, not fixed constants.

## Run protocol and build flags (`compile_macro_bench.sh`)

The reference command above needs no protocol flag: the uninstrumented build times the way the reference design does (issue #27) — no warm-up, reps until `--tmin` seconds, correctness checked after timing. Every run opens with an overview of the settings in effect, each marked `[default]` or with what set it.

| flag | effect |
|---|---|
| *(none)* | uninstrumented build, **reference protocol**, `tmin` 3.0 s. Logs `raw_*_noinstr_refproto.txt` |
| `--tmin <s>` | time budget for the reference protocol. Refused where the reference protocol is not running |
| `--no-ref-protocol` | the older **warm-up protocol** instead: 5 warm-up calls, a scalar-pack pass, then the fixed `iterations` from `config.json`. Logs `raw_*_noinstr.txt` |
| `--scalar-pack` | every timed call packs with the scalar packer instead of the RVV one (issue #26). Logs get a `_scalarpack` suffix (before `_no_nc_override`), so it never mixes with a vector-live run. See `docs/metrics.md`, "Vector pack against scalar pack, end to end" |
| `--no-nc-overrides` | ignore the per-shape `nc` overrides in `config.json` (B3 ships at `nc=128`; the reference runs `nc=384`). The head-to-head configuration; a run without it is the shipped one. Logs get a `_no_nc_override` suffix, so the two never mix (issue #16) |
| `--cpu <n>` | pin the timed run to core `n` with `taskset -c n` (issue #21). Default: core 4, the same default `run_ref_baseline.sh` uses, so both sides time on one core. Not core 0: the board sends every device interrupt there |
| `--no-pin` | run unpinned, to reproduce the pre-#21 numbers. Refused together with `--cpu` |
| `--instr` | instrumented debug build. Its default is the warm-up protocol, since the reference protocol cannot run there; asking it for `--ref-protocol` is refused. Logs `raw_*.txt`. Never quote against the reference |
| `--dry-run` | print the run overview and stop, without generating, syncing or running anything |

`compare_ref_baseline.py` defaults to the instrumented log name; point it at a default run with `--raw-name all_raw_<project>_noinstr_refproto.txt` (add `_no_nc_override` for a run made with `--no-nc-overrides`; the fringe column then follows on its own). Details and the measured warm-up effect: `docs/metrics.md`.

Every run writes `<run dir>/provenance.txt`: the pin, the affinity `taskset` reports under it, the CPU governor, the compiler, the load average and `/proc/cpuinfo`, taken on the board just before the build and run. `run_ref_baseline.sh` puts the same block in its own `provenance.txt`. `compare_ref_baseline.py` reads them and prints a `WARNING` when the two sides were not pinned, were pinned to different cores, or were made more than an hour apart; a run from before issue #21 has no provenance and is reported as pinning unknown.

## Head-to-head against the reference design

```bash
source .venv/bin/activate && REMOTE_PASS="" REMOTE_HOST="fej" REMOTE_PORT="3322" \
  ./tests/pure_xdsl_blis_rvv/run_head_to_head.sh --jump \
  --riscv-user jlei --riscv-ip jbpi --riscv-port 22 \
  --cpu 0
```

This is the default command for a benchmark run. It pins both sides to core 0, which is also where the board sends every device interrupt: expect that core to be noisier than the default, core 4. Runs our side (`compile_macro_bench.sh --no-nc-overrides`, the default build and protocol) and then the reference side (`run_ref_baseline.sh --bracket`), both pinned to the same core, and compares exactly that pair. This is the command for a number quoted against the reference; a reference baseline from another day is not comparable (issue #21). Take `--cpu`, `--no-pin`, `--tmin` and the connection flags; `--opt OPT|LDX` picks the reference kernel set. For anything else run the two scripts yourself, then `compare_ref_baseline.py --result-dir ... --runs ... --raw-name ...`.

## Correctness pass

Every row of the shipped `config.json` sets `skip_ref_run`, so a benchmark run logs `Status=SKIP` and checks nothing. The check lives in a separate config, `tests/pure_xdsl_blis_rvv/config_correctness.json`: the same three pinnings (16x15, 32x6, 32x7) with the scalar oracle on, over `datasets/correctness_small.dat` (three small shapes: all full tiles, fringe rows and columns with three K blocks, and one Fringe Tile that is the whole matrix). The scalar oracle (`gemm_ref`) is O(m·n·k), so no shape above B2 (384×384×64) may run under it; `test_correctness_pass.py` fails if a row of `config.json` or `config_correctness.json` turns the check on for a bigger one.

```bash
source .venv/bin/activate && REMOTE_PASS="" REMOTE_HOST="fej" REMOTE_PORT="3322"   ./tests/pure_xdsl_blis_rvv/compile_macro_bench.sh --jump   --riscv-user jlei --riscv-ip jbpi --riscv-port 22   --no-nc-overrides --cpu 0 \
  --config config_correctness.json --tmin 0.2
```

It is the benchmark command with `--config` and `--tmin 0.2` added (the timing is not the point; a short `tmin` keeps it to a few minutes). Read `Status` in `<run dir>/all_raw_*.txt`: `PASS` for all nine rows. `SKIP` means the check did not run; `FAIL` prints the first mismatches.

`PASS` means: the whole `C` our kernels leave (the packers, the microkernel and the fringe path together, after the timed reps) is within a relative error of `max(1e-5, k·1e-6)` of the `C` the scalar oracle (`gemm_ref`) computes. The error is `‖C_ours − C_ref‖₂ / ‖C_ref‖₂` over all `m×n` entries, taken in double. `A` and `B` are uniform random in [0, 1], `alpha = 1`, `beta = 0`. It is a tolerance, not a bitwise match, and it is loose enough that one wrong `k`-step in one entry would slip through; a wrong row, column or vector chunk of a panel would not (at `k = 300` the bound is 3e-4; one wrong `k`-step in one entry is about 5e-5, while a whole wrong 8-row chunk of a 32-row panel is at least about 4e-2, and 5e-1 if it is zeroed). The bitwise check of the packed layout is the pack test below. `beta ≠ 0` from the caller is not exercised: the driver never passes it; the beta=1 microkernel runs only as the later K blocks' accumulate. The run overwrites the tracked `pj_gen/` like any `--config` run: regenerate it from `config.json` afterwards (`cd tests/pure_xdsl_blis_rvv && python3 generate_pure_blis_project.py --config config.json`, then put back the trailing newline the `.mlir` files lose).

The vector packers get their own check, per shipped `RR` (6, 7, 15, 16, 32): `pytest tests/pure_xdsl_blis_rvv/test_pack_correctness.py`. It builds the real packers, compiles them on the board through `fej`, and skips when the board is not reachable.

## One-time dependency builds

Not part of the per-run pipeline above — see `docs/dependencies_setup.md` for building LLVM/MLIR locally and OpenBLAS/BLIS on the board.
