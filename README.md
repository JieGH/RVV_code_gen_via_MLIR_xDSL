# RVV Code Generation via MLIR & xDSL

Generate GEMM kernels for RISC-V Vector (RVV) from Python with xDSL, then build, check and time them on a RISC-V board.

You choose the **data type**, the **packing method**, the **microkernel tile** (MRxNR: one fixed tile, or a sweep of every tile) and the **problem shapes**. The generator writes a complete C++ project: packers, microkernels, fringe path and a benchmark driver. One script builds it on the board, checks it against a scalar reference and fetches the results.

Step by step: **[user guide](docs/user_guide.md)**. Terms: [CONTEXT.md](CONTEXT.md).

## Setup

**Host:** `python3`, `uv`, `ssh`, `rsync`; `sshpass` to reach the board through a jump host.
**Board** (BananaPi, VLEN 256): `g++` with RVV (13 or later; `g++-14` for f16f16f32) and `make`. No BLAS library needed.

```bash
make install        # creates .venv
```

Set the board connection once per shell. Through a jump host:

```bash
export REMOTE_PASS="" REMOTE_HOST="<jump-host>" REMOTE_PORT="<jump-port>"
CONN="--jump --riscv-user <user> --riscv-ip <board> --riscv-port 22"
```

For a board you reach directly: `CONN="--direct --riscv-user <user> --riscv-ip <board> --riscv-port 22"`.

## Run a model

From the repo root. Each command generates, builds and times one model's GEMM layers on core 4.

```bash
B=./tests/pure_xdsl_blis_rvv/compile_macro_bench.sh

# BERT-large, layers B1-B5 (tiles 16x15, 32x6, 32x7; about 7 minutes)
$B $CONN --no-nc-overrides --cpu 4 --tmin 10 --config configs/benchmark/config_bert.json

# Square, S1-S5 = 1000^3 to 5000^3 (tile 32x7; about 3 minutes)
$B $CONN --no-nc-overrides --cpu 4 --tmin 10 --config configs/benchmark/config_square.json
```

## Feature examples

Each command checks one feature against the scalar reference on small shapes. Each takes a few minutes, and every row must say `PASS`.

```bash
CHECK="$B $CONN --no-nc-overrides --tmin 0.2"
C=configs/correctness

# Data type and method
$CHECK --cpu 4 --config $C/config_correctness.json                              # fp32, AXPY method
$CHECK --cpu 4 --config $C/config_feature_dot.json                              # fp32, DOT method
$CHECK --cpu 4 --config $C/config_feature_int8.json                             # int8
$CHECK --cpu 4 --config $C/config_issue112_mixed_correctness.json --scalar-pack # f16 inputs, fp32 output

# Packing method
$CHECK --cpu 4 --config $C/config_correctness.json --scalar-pack     # scalar instead of RVV packing
$CHECK --cpu 4 --config $C/config_correctness.json --pack-b col-g4   # another B packer
$CHECK --cpu 4 --config $C/config_correctness.json --zero-pack-lines # clear the pack buffer first
$CHECK --cpu 4 --config $C/config_correctness.json --pad-ldb 16      # pad B's leading dimension

# Microkernel tile: forced, or swept
$CHECK --cpu 4 --config $C/config_correctness.json    # forced: one project per tile (16x15, 32x6, 32x7)
$CHECK --cpu 4 --config $C/config_feature_search.json # swept: every tile up to 32x7, best one reported (about 25 minutes)

# Threads
$CHECK --threads 8 --config $C/config_correctness.json
```

To change the problem shapes, list them in a dataset file (`label M N K`, one per line) under `tests/pure_xdsl_blis_rvv/datasets/` and name it in a config (see [Choosing what to generate](#choosing-what-to-generate)). `runners/run_feature_baseline.sh` runs all the checks above in one go.

## Essential commands

All from the repo root.

```bash
B=./tests/pure_xdsl_blis_rvv/compile_macro_bench.sh

# 1. Generate only (no board): writes the projects to tests/pure_xdsl_blis_rvv/pj_gen/
(cd tests/pure_xdsl_blis_rvv && ../../.venv/bin/python generate_pure_blis_project.py --config config.json)

# 2. Check correctness on the board (a few minutes; every row must say PASS)
$B $CONN --no-nc-overrides --cpu 4 --config configs/correctness/config_correctness.json --tmin 0.2

# 3. Benchmark (default config: config.json)
$B $CONN --no-nc-overrides --cpu 4 --tmin 10

# 4. Check every feature on the board (about an hour)
./tests/pure_xdsl_blis_rvv/runners/run_feature_baseline.sh $CONN

# 5. Host tests (about 17 minutes; a few skip without ../ext/, see Reference Design below)
cd tests/pure_xdsl_blis_rvv && ../../.venv/bin/python -m pytest
```

Add `--dry-run` to a board command to see its plan without touching the board. Results land in `tests/pure_xdsl_blis_rvv/logs/run_<stamp>/`; `comparison.md` there is the summary.

## Choosing what to generate

A config is a JSON list of rows (`//` comments allowed). `--config` paths are relative to `tests/pure_xdsl_blis_rvv/`.

```json
["fp32", "AXPY_FP32_Pack", [16, 15], 256, "jbpi", ["correctness_small.dat"], false, 5]
```

| Field | Choices |
|-------|---------|
| data type, method | `fp32` with `AXPY_FP32_Pack` or `DOT_FP32_LANE_Pack`; `f16f16f32` with `AXPY_FP32_Pack` (run with `--scalar-pack`); `int8` with `AXPY_INT8_Pack` |
| tile | `[MR, NR]`, or a list of tiles |
| vlen, arch | `256`; the board name |
| datasets | files in `datasets/`, one shape per line: `label M N K`; a sub-path such as `experiments/issue85_sweep.dat` works ([datasets/README](tests/pure_xdsl_blis_rvv/datasets/README.md)) |
| skip_ref_run | `false` checks every result against the scalar reference; `true` skips the check (timing runs) |
| iterations | present: **pinned**, one project for the first tile. Left out: **search mode**, one project with every tile up to the largest |

Packing and threads are flags of `compile_macro_bench.sh`:

| Flag | Effect |
|------|--------|
| (none) | RVV packing |
| `--scalar-pack` | scalar packing |
| `--pack-b <name>` | another B packer (fp32 AXPY): `row` (default), `row-intr`, `col-g1`, `col-g2`, `col-g4`, `col-gall`, `col-g1-m4` |
| `--zero-pack-lines` | clear the pack buffer's cache lines before each pack (AXPY) |
| `--pad-ldb <n>` | pad B's leading dimension by n elements |
| `--threads <n>` | run each GEMM on n threads (fp32 AXPY, RVV packing) |
| `--cpu <n>` | the core to pin to: 4 (never 0) |
| `--tmin <s>` | seconds to time each shape |

## Other workflows

Scripts in `tests/pure_xdsl_blis_rvv/runners/` ([README](tests/pure_xdsl_blis_rvv/runners/README.md)):

| Task | Script |
|------|--------|
| Find the best tile per shape (Tile Search) | `run_yolo_sweep.sh`, `run_paper_sweep.sh` |
| Compare with the Reference Design on the same core (needs `../ext/`, below) | `run_head_to_head.sh` |
| Vector against scalar packing | `run_pack_swap.sh` |
| 1 and 8 threads on the YOLO layers | `run_parallel_gemm.sh`, `run_hector_gemm.sh` |
| Standalone microkernels to hand over | `gen_handover_ukernels.py` |

Tools that read the results: [`readers/`](tests/pure_xdsl_blis_rvv/readers/README.md).

### Reference Design

The Reference Design is not in this repo. `run_head_to_head.sh`, `run_ref_sweep.sh` and `ref_baseline/` read it from a fixed path beside the repo. Clone it there once, from the repo root, at the commit we use:

```bash
git clone https://github.com/adcastel/RVV_ukernels_benchmark.git ../ext/RVV_ukernels_benchmark
git -C ../ext/RVV_ukernels_benchmark checkout eb84ded
```

Without it these scripts stop with `Reference Design not found`, and the host tests that run them are skipped.

## Layout

| Path | What it is |
|---|---|
| [tests/pure_xdsl_blis_rvv/](tests/pure_xdsl_blis_rvv/) | The generator, the bench, runners, readers, configs and tests. Start here. |
| [src/xdsltemplate/](src/xdsltemplate/) | The xDSL backend: RVV, GEMM and EmitC dialects and their lowering to C++. |
| [docs/](docs/) | [User guide](docs/user_guide.md), [metrics](docs/metrics.md), [board execution and house rules](docs/agents/execution.md), [roadmap](docs/ROADMAP.md), ADRs. |
| `.scratch/issues/` | Write-ups of past investigations. [Its index](.scratch/issues/README.md) maps each GitHub issue to its write-up, experiment, datasets and reports. |

Before quoting a number, read the house rules: [user guide §12](docs/user_guide.md#12-rules-for-a-number-you-quote).
