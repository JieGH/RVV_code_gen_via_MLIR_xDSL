# RVV Code Generation via MLIR & xDSL

Generate BLIS-family GEMM kernels for RISC-V Vector with xDSL, cross-compile them on a RISC-V board, benchmark them, and pull the results back.

Everything the kernels do — the register-tiled microkernel, the RVV packing routines, and the macrokernel loop structure — is emitted by a Python/xDSL generator, not written by hand. Blocking dimensions (`MC`, `NC`, `KC`) come from an analytical cache model rather than manual tuning.

**Start with the [user guide](docs/user_guide.md)**: every high-level operation (correctness pass, benchmark, head-to-head against the Reference Design, Tile Search, packing comparison, data types, threads and YOLO, reading results, tests) with its command and how to read the result. Vocabulary: [CONTEXT.md](CONTEXT.md).

## What it does

| Feature | Entry point |
|---------|-------------|
| Generate one GEMM project per microkernel tile (fp32; f16f16f32 mixed precision; int8) | `generate_pure_blis_project.py` |
| Build and run it on the board, with a correctness check against a scalar oracle | `compile_macro_bench.sh` |
| Compare against the Reference Design on the same core | `runners/run_head_to_head.sh` |
| Find the best MRxNR tile per shape (Tile Search) | `runners/run_yolo_sweep.sh`, `runners/run_paper_sweep.sh`, `readers/search_report.py` |
| Compare packers: RVV, scalar, B-pack variants, cleared pack lines | `--scalar-pack`, `--pack-b`, `--zero-pack-lines`, `runners/run_pack_swap.sh` |
| N-thread GEMM; Héctor's YOLO layers | `--threads N`, `runners/run_parallel_gemm.sh`, `runners/run_hector_gemm.sh` |
| Hand-over microkernels for use outside the harness | `runners/gen_handover_ukernels.py` |

All of these live in `tests/pure_xdsl_blis_rvv/`. Its top level holds only the main workflow (the generator and `compile_macro_bench.sh`); `runners/` holds the multi-run scripts and `readers/` the tools that read run logs.

## Layout

| Path | What it is |
|---|---|
| [src/xdsltemplate/](src/xdsltemplate/) | The xDSL backend. `dialects/` defines the RVV, GEMM and EmitC operations; `transforms/` lowers them to C++. |
| [tests/pure_xdsl_blis_rvv/](tests/pure_xdsl_blis_rvv/) | The Project Generator, the bench, the runners and the readers. Its [README](tests/pure_xdsl_blis_rvv/README.md) maps each feature to its configs and tests. |
| `tests/pure_xdsl_blis_rvv/runners/` | Multi-run scripts: Tile Search, threads, pack swap, Reference Design sweeps, hand-over kernels. Each wraps `compile_macro_bench.sh`. |
| `tests/pure_xdsl_blis_rvv/readers/` | Read run folders and logs: per-run summary, per-rep stats, Tile Search report, pack comparison. |
| `tests/pure_xdsl_blis_rvv/configs/` | Configs by purpose: `correctness/`, `tile_search/`, `benchmark/`. `config.json` beside them is the default. |
| `tests/pure_xdsl_blis_rvv/experiments/issueNN/` | One folder per past investigation: its run script, configs, report and tests. |
| `tests/pure_xdsl_blis_rvv/tests/` | Host tests. |
| `tests/pure_xdsl_blis_rvv/logs/` | Run folders, committed: every quoted number cites one. |
| `tests/pure_xdsl_blis_rvv/pj_gen/` | Generated output, one project per tile. Gitignored; wiped and rebuilt on every run. |
| [tests/pure_xdsl_blis_rvv/ref_baseline/](tests/pure_xdsl_blis_rvv/ref_baseline/) | Runs the Reference Design for comparison. Needs the sibling `../ext/RVV_ukernels_benchmark` tree. |
| [docs/](docs/) | User guide, metric definitions, ADRs, the agent-facing execution notes. |

## Requirements

**Local:** `python3`, `uv`, `ssh`, `rsync`, and `sshpass` for jump-host mode.

**Board** (BananaPi M2 Pro, VLEN 256): `g++` with RVV support (13 is enough) and `make`; f16f16f32 projects need `g++-14`. The generated Makefile links only `-lm`: no OpenBLAS or BLIS needed.

```bash
make install        # creates .venv
```

## Quick start

From the repo root. The script activates `.venv` itself, generates the kernels, syncs and builds them on the board, then fetches the logs into `tests/pure_xdsl_blis_rvv/logs/run_<stamp>/`.

```bash
source .venv/bin/activate && REMOTE_PASS="" REMOTE_HOST="fej" REMOTE_PORT="3322" \
  ./tests/pure_xdsl_blis_rvv/compile_macro_bench.sh --jump \
  --riscv-user jlei --riscv-ip jbpi --riscv-port 22 \
  --no-nc-overrides --cpu 4 --tmin 10
```

`REMOTE_HOST` and `REMOTE_PORT` reach the intermediate host; the `--riscv-*` flags reach the board behind it. For a board you reach directly, drop `--jump` and the `REMOTE_*` variables. Add `--config <path>` (relative to `tests/pure_xdsl_blis_rvv/`) for another config, e.g. `--config configs/correctness/config_correctness.json --tmin 0.2` for the correctness pass. Add `--dry-run` to print the settings and stop without touching the board.

The default is the uninstrumented build and the reference protocol: the numbers comparable to the Reference Design. GOPs are by mean time of the whole call, packing included. `--instr` adds per-region timers and perf counters but costs 11-17%: never quote it against the reference. The other flags, and what each does to a number: [user guide §5](docs/user_guide.md#5-run-a-benchmark). The house rules for a number you quote (never core 0, several launches for small gaps, committed logs): [user guide §12](docs/user_guide.md#12-rules-for-a-number-you-quote).

Generate without a board, to read the emitted MLIR and C++:

```bash
cd tests/pure_xdsl_blis_rvv && python3 generate_pure_blis_project.py --config config.json
```

## Tests

```bash
cd tests/pure_xdsl_blis_rvv && ../../.venv/bin/python -m pytest
```

681 tests, about 17 minutes. The full regime (host tests, snapshot compare, feature baseline on the board): [Test regime](tests/pure_xdsl_blis_rvv/README.md#test-regime). Seven (`*_on_board`) compile on the board through `fej` and skip when it is not reachable; `-k "not on_board"` runs the host-only set. Several read real logs under `logs/` for their expected values, so those log folders must stay. Never run the tests during a board series: they rewrite `pj_gen/` and `ref_baseline/generated/`.

## More

- [docs/user_guide.md](docs/user_guide.md) — every operation, step by step
- [docs/agents/execution.md](docs/agents/execution.md) — entry points in full, connection modes, run protocol, house rules
- [docs/metrics.md](docs/metrics.md) — what each reported column means
- [docs/ROADMAP.md](docs/ROADMAP.md) — planned work
- `.scratch/issues/` — write-ups of past investigations
