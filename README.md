# RVV Code Generation via MLIR & xDSL

Generate BLIS-family GEMM kernels for RISC-V Vector with xDSL, cross-compile them on a RISC-V board, benchmark them, and pull the results back.

Everything the kernels do — the register-tiled microkernel, the RVV packing routines, and the macrokernel loop structure — is emitted by a Python/xDSL generator, not written by hand. Blocking dimensions (`MC`, `NC`, `KC`) come from an analytical cache model rather than manual tuning.

See [CONTEXT.md](CONTEXT.md) for the vocabulary used throughout: microkernel, macrokernel, packer, kernel family, reference design, reference protocol.

## Layout

| Path | What it is |
|---|---|
| [src/xdsltemplate/](src/xdsltemplate/) | The xDSL backend. `dialects/` defines the RVV, GEMM and EmitC operations; `transforms/` lowers them to C++. |
| [tests/pure_xdsl_blis_rvv/](tests/pure_xdsl_blis_rvv/) | The project generator and its benchmark harness. |
| `tests/pure_xdsl_blis_rvv/pj_gen/` | Generated output: one self-contained project per `(MR, NR)` variant. Not tracked by git; wiped and rebuilt on every run. |
| [tests/pure_xdsl_blis_rvv/ref_baseline/](tests/pure_xdsl_blis_rvv/ref_baseline/) | Runs the reference design for comparison. Needs the sibling `../ext/RVV_ukernels_benchmark` tree. |
| [docs/](docs/) | Metrics definitions, ADRs, and the agent-facing execution notes. |

## Requirements

**Local:** `python3`, `uv`, `ssh`, `rsync`, and `sshpass` for jump-host mode.

**Board** (BananaPi M2 Pro, VLEN 256): GCC 14 or newer, for RVV support, plus `make`. The generated Makefile links only `-lm` — no OpenBLAS or BLIS needed.

Create the local environment:

```bash
make install
```

## Running a benchmark

Run from the repo root. The script activates `.venv` itself, generates the kernels, syncs and builds them on the board, then fetches the logs into a timestamped directory under `tests/pure_xdsl_blis_rvv/logs/`.

```bash
source .venv/bin/activate && REMOTE_PASS="" REMOTE_HOST="fej" REMOTE_PORT="3322" \
  ./tests/pure_xdsl_blis_rvv/compile_macro_bench.sh --jump \
  --riscv-user jlei --riscv-ip jbpi --riscv-port 22 \
  --no-nc-overrides --cpu 4 --tmin 10
```

`REMOTE_HOST` and `REMOTE_PORT` reach the intermediate host; the `--riscv-*` flags reach the board behind it. For a board you can reach directly, drop `--jump` and the `REMOTE_*` variables. Add `--config <name>.json` for a project config other than `config.json`.

Add `--dry-run` to print the settings and stop without touching the board. Every run opens with that same overview, marking each setting `[default]` or naming the flag that set it.

### The flags that change what a number means

| Flag | Effect |
|---|---|
| *(default)* | Uninstrumented build, reference protocol: no warm-up, reps until `--tmin` seconds, correctness checked after timing. These are the numbers comparable to the reference design. |
| `--instr` | Keeps the per-tile timers and perf counters. They cost 11–17% of measured time, so **never quote these against the reference**. Use it to see where our own time goes. |
| `--no-ref-protocol` | Warm-up first, then a fixed rep count from `config.json`. Steadier on short shapes, but not the reference's protocol. |
| `--no-nc-overrides` | Ignores the per-shape `nc` table in `config.json`, so every shape uses the heuristic's `nc`. Needed for a fair head-to-head; the reference has no counterpart for those overrides. Drop it for the shipped-configuration number. |
| `--scalar-pack` | Every timed call packs with the scalar packer instead of the RVV one (issue #26). Own log names (`_scalarpack`). `run_pack_swap.sh` runs vector-live and scalar-live back to back and compares them; see [docs/metrics.md](docs/metrics.md). |
| `--zero-pack-lines` | Before each pack, clears every whole 64-byte line of the Ac/Bc block with `cbo.zero`, so the packer's stores no longer read those lines from the L2 first (issue #43). AXPY kernels only; off by default. Own log names (`_zeroline`). See [docs/metrics.md](docs/metrics.md), "Clearing the pack lines". |
| `--cpu N` / `--no-pin` | Pins the timed run to one core (default 4). Never use core 0: it takes every device interrupt and runs 5-10% slower. The house rules for a quotable number (non-zero core, several launches for small gaps, saved logs) are in [docs/agents/execution.md](docs/agents/execution.md). |
| `--save-reps` / `--no-save-reps` | Saves every timed rep's own time, in the order the reps ran, to `reps_*.txt` (on by default since issue #39). |
| `--raw-events <list>` | With `--instr` only: counts the core's own events per rep into the per-rep log, e.g. `--raw-events x60-l2` for the X60's L2 and prefetch events (issue #42). See [docs/metrics.md](docs/metrics.md), "Raw events". |

Each combination writes to its own log name, so the two can never be mixed up: `raw_*_noinstr_refproto_no_nc_override.txt` for the command above.

### Comparing against the reference design

A reference baseline from another day is not comparable. To quote a number against the reference, run both sides back to back on the same pinned core:

```bash
source .venv/bin/activate && REMOTE_PASS="" REMOTE_HOST="fej" REMOTE_PORT="3322" \
  ./tests/pure_xdsl_blis_rvv/run_head_to_head.sh --jump \
  --riscv-user jlei --riscv-ip jbpi --riscv-port 22 --cpu 4
```

Our KerGOPs figure excludes packing while the reference's numbers include it. [docs/metrics.md](docs/metrics.md) defines each column; [tests/pure_xdsl_blis_rvv/ref_baseline/README.md](tests/pure_xdsl_blis_rvv/ref_baseline/README.md) covers the fringe-tile difference between the two designs.

### Finding the best microkernel size (Tile Search)

A Tile Search times a list of Candidate Tiles on every shape of a dataset and names the fastest tile per shape. The config is one pinned row per tile (e.g. `configs/tile_search/config_yolo_search.json`, the 19 YOLO26 tiles of issue #59), so each tile times exactly as it does alone. Run it with the benchmark command plus `--config configs/tile_search/config_yolo_search.json`, three launches, then read the run folders:

```bash
python3 tests/pure_xdsl_blis_rvv/search_report.py logs/run_<A> logs/run_<B> logs/run_<C> --compare 16x8 --compare n=4:16x4
```

It ranks by GOPs of the whole call (packing included) by mean time, with GOPs by median rep beside it, and adds the best single tile and the best tile per `n` over the dataset. It needs the per-rep logs (on by default; not with `--no-save-reps`). Check the tiles first with `--config configs/correctness/config_yolo_correctness.json --tmin 0.2` (every row `PASS`).

## Generating without a board

The generator half runs on its own, which is the quick way to inspect the emitted MLIR and C++:

```bash
source .venv/bin/activate
cd tests/pure_xdsl_blis_rvv && python3 generate_pure_blis_project.py --config config.json
```

`pj_gen/` is gitignored and rebuilt from scratch each run, so never edit it by hand; change the generator or `templates/` instead.

## Tests

```bash
source .venv/bin/activate && python3 -m pytest tests/ -q
```

145 tests, of which 139 run on the host alone. Six, named `*_on_board`, ssh through `fej` to `jbpi` and run a harness there; they skip themselves when the board is unreachable. To run the host-only set: `python3 -m pytest tests/ -q -k "not on_board"`.

The tests cover the generated packing and dispatch code (`test_pack_correctness`, `test_pad_zeroing`), the config and cache model, and the benchmark harness itself: the run protocol, core pinning, perf counters, and the comparison and summary scripts (all but `compare_pack_swap.py`, which was checked by hand on real logs). Several read real logs under `logs/` for their expected values, so those log directories must stay.

## More

- [docs/agents/execution.md](docs/agents/execution.md) — entry points, connection modes, run protocol in full
- [docs/metrics.md](docs/metrics.md) — what each reported column means
- [docs/ROADMAP.md](docs/ROADMAP.md) — planned work
- `.scratch/issues/` — write-ups of past debugging and performance investigations
