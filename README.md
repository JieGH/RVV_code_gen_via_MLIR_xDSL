# RVV Code Generation via MLIR & xDSL

Generate BLIS-family GEMM kernels for RISC-V Vector with xDSL, cross-compile them on a RISC-V board, benchmark them, and pull the results back.

Everything the kernels do — the register-tiled microkernel, the RVV packing routines, and the macrokernel loop structure — is emitted by a Python/xDSL generator, not written by hand. Blocking dimensions (`MC`, `NC`, `KC`) come from an analytical cache model rather than manual tuning.

See [CONTEXT.md](CONTEXT.md) for the vocabulary used throughout: microkernel, macrokernel, packer, kernel family, reference design, reference protocol.

## Layout

| Path | What it is |
|---|---|
| [src/xdsltemplate/](src/xdsltemplate/) | The xDSL backend. `dialects/` defines the RVV, GEMM and EmitC operations; `transforms/` lowers them to C++. |
| [tests/pure_xdsl_blis_rvv/](tests/pure_xdsl_blis_rvv/) | The project generator and its benchmark harness. |
| `tests/pure_xdsl_blis_rvv/pj_gen/` | Generated output: one self-contained project per `(MR, NR)` variant. Regenerated, not edited. |
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
  --no-nc-overrides --cpu 0 --tmin 10
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
| `--cpu N` / `--no-pin` | Pins the timed run to one core. Core 0 also takes every device interrupt, so expect it to be noisier than core 4. |

Each combination writes to its own log name, so the two can never be mixed up: `raw_*_noinstr_refproto_no_nc_override.txt` for the command above.

### Comparing against the reference design

A reference baseline from another day is not comparable. To quote a number against the reference, run both sides back to back on the same pinned core:

```bash
source .venv/bin/activate && REMOTE_PASS="" REMOTE_HOST="fej" REMOTE_PORT="3322" \
  ./tests/pure_xdsl_blis_rvv/run_head_to_head.sh --jump \
  --riscv-user jlei --riscv-ip jbpi --riscv-port 22 --cpu 0
```

Our KerGOPs figure excludes packing while the reference's numbers include it. [docs/metrics.md](docs/metrics.md) defines each column; [tests/pure_xdsl_blis_rvv/ref_baseline/README.md](tests/pure_xdsl_blis_rvv/ref_baseline/README.md) covers the fringe-tile difference between the two designs.

## Generating without a board

The generator half runs on its own, which is the quick way to inspect the emitted MLIR and C++:

```bash
source .venv/bin/activate
cd tests/pure_xdsl_blis_rvv && python3 generate_pure_blis_project.py --config config.json
```

`pj_gen/` is tracked, so regenerate it before committing after any change to the generator or the templates.

## Tests

```bash
source .venv/bin/activate && python3 -m pytest tests/ -q
```

These cover the generator, the cache model, packing correctness, the pad zeroing, and the protocol and build flags. They need no board.

## More

- [docs/agents/execution.md](docs/agents/execution.md) — entry points, connection modes, run protocol in full
- [docs/metrics.md](docs/metrics.md) — what each reported column means
- [docs/ROADMAP.md](docs/ROADMAP.md) — planned work
- `.scratch/issues/` — write-ups of past debugging and performance investigations
