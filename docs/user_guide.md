# User guide

What you can do with this project, one operation per section: when to use it, the command, what it writes, and how to read the result. Words like Microkernel, Candidate Tile, Tile Search and Reference Design are defined in [CONTEXT.md](../CONTEXT.md). The full flag tables and the connection details are in [agents/execution.md](agents/execution.md); what each log column means is in [metrics.md](metrics.md).

All commands run from the repo root unless a section says otherwise.

## Contents

1. [Set up](#1-set-up)
2. [Generate a project without a board](#2-generate-a-project-without-a-board)
3. [Write a config](#3-write-a-config)
4. [Check correctness on the board](#4-check-correctness-on-the-board)
5. [Run a benchmark](#5-run-a-benchmark)
6. [Compare against the Reference Design](#6-compare-against-the-reference-design)
7. [Find the best microkernel size (Tile Search)](#7-find-the-best-microkernel-size-tile-search)
8. [Compare packing methods](#8-compare-packing-methods)
9. [Other data types](#9-other-data-types)
10. [Threads and the YOLO layers](#10-threads-and-the-yolo-layers)
11. [Read the results](#11-read-the-results)
12. [Rules for a number you quote](#12-rules-for-a-number-you-quote)
13. [Run the host tests](#13-run-the-host-tests)
14. [Add a new experiment](#14-add-a-new-experiment)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. Set up

**Here:** `python3`, `uv`, `ssh`, `rsync`; `sshpass` for the jump-host route.
**Board** (BananaPi M2 Pro, SpacemiT K1/X60, VLEN 256): `g++` with RVV (13 is enough), `make`; `g++-14` only for f16f16f32 projects. The generated Makefile links `-lm` only.

```bash
make install            # creates .venv with xdsl and the test tools
```

Every board command below uses the same connection part. Through the jump host `fej` (the default here):

```bash
source .venv/bin/activate
export REMOTE_PASS="" REMOTE_HOST="fej" REMOTE_PORT="3322"
CONN="--jump --riscv-user jlei --riscv-ip jbpi --riscv-port 22"
```

For a board you reach directly, drop `--jump` and the `REMOTE_*` variables. There are two boards, `jbpi` and `jbpi2`; they share one NFS home (see [§15](#15-troubleshooting)).

## 2. Generate a project without a board

Use this to look at the MLIR and C++ the generator writes.

```bash
cd tests/pure_xdsl_blis_rvv
python3 generate_pure_blis_project.py --config config.json
```

**Writes** `pj_gen/<datatype>_<method>_vlen<V>[_MR<m>NR<n>]/`, one folder per config row, and `pj_gen/compile_order.txt`. Each folder is a full project: `gemm_pack_xdsl.{mlir,cpp}` (packers), `gemm_b0_xdsl.*` / `gemm_b1_xdsl.*` (microkernels for beta 0 and 1), `gemm_dispatch_xdsl.cpp` (macrokernel loops and fringe path), `driver_benchmark.cpp`, `Makefile`.

`pj_gen/` is gitignored and wiped on every run. Change the generator or `templates/`, never `pj_gen/`. `--threads N` generates the N-thread GEMM (fp32 AXPY only).

## 3. Write a config

A config is a JSON list of rows (comments with `//` are allowed):

```
[datatype, method, [MR, NR], vlen, arch, [datasets], skip_ref_run, iterations, {nc overrides}]
["fp32", "AXPY_FP32_Pack", [32, 7], 256, "jbpi", ["paper_bert_square.dat"], true, 1000]
```

| Field | Values |
|-------|--------|
| datatype | `fp32`, `f16f16f32` (FP16 A and B, FP32 C), `int8`, `i8i8i32` |
| method | `AXPY_FP32_Pack` (the main kernel), `DOT_FP32_LANE_Pack`, `AXPY_INT8_Pack` |
| `[MR, NR]` | the microkernel tile |
| datasets | file names in `tests/pure_xdsl_blis_rvv/datasets/`; each line is `label m n k` |
| skip_ref_run | `true`: no correctness check (timing configs); `false`: check against the scalar oracle. The oracle is O(m·n·k): only shapes up to B2 (384x384x64) |
| iterations | rep count for the warm-up protocol. **Its presence also pins the row:** one project for the first MRxNR only. Leave it out (7 values) for search mode: one project holding every tile up to the largest |
| nc overrides | optional `{label: nc}`; ignored with `--no-nc-overrides` |

Where configs live in `tests/pure_xdsl_blis_rvv/`:

| Folder | For |
|--------|-----|
| `config.json` | the default of `compile_macro_bench.sh`: the shipped timing configuration |
| `configs/correctness/` | oracle-on passes on small shapes |
| `configs/tile_search/` | one pinned row per Candidate Tile |
| `configs/benchmark/` | fixed-tile timing runs (packing, threads, YOLO) |
| `experiments/issueNN/` | configs of one past investigation |

Pass any of them as `--config <path>`, relative to `tests/pure_xdsl_blis_rvv/`, e.g. `--config configs/correctness/config_correctness.json`.

## 4. Check correctness on the board

Run it after any change to the generator, a template or a packer, and before every timing series.

```bash
./tests/pure_xdsl_blis_rvv/compile_macro_bench.sh $CONN --no-nc-overrides --cpu 4 \
  --config configs/correctness/config_correctness.json --tmin 0.2
```

A few minutes. **Read** the `Status` column of `logs/run_<stamp>/all_raw_*.txt`: `PASS` for all 9 rows (3 tiles x 3 shapes). `FAIL` prints the first mismatches; `SKIP` means the row had `skip_ref_run` on.

To check every option at once (data types, packers, `--pack-b` variants, pinned and search tiles, threads), run the feature baseline, about an hour:

```bash
./tests/pure_xdsl_blis_rvv/runners/run_feature_baseline.sh $CONN      # --list names the 15 cases, --only NAME,... runs some
```

Each case is one run like the one above; the table of verdicts is in `logs/feature_baseline_<stamp>.tsv`.

PASS means our whole `C` is within a relative error of `max(1e-5, k·1e-6)` of the oracle's. Other passes in `configs/correctness/`: `config_paper_correctness.json` (B2, B3, a 200³ square), `config_yolo_correctness.json` (the 19 YOLO tiles), `config_yolo_correctness_8t.json` (with `--threads 8`), `config_issue112_mixed_correctness.json` (f16f16f32, needs `--scalar-pack`).

The bitwise check of the packed layout, per shipped `RR`, is a host test that compiles on the board: `pytest tests/pure_xdsl_blis_rvv/tests/test_pack_correctness.py` (skips when the board is not reachable).

## 5. Run a benchmark

```bash
./tests/pure_xdsl_blis_rvv/compile_macro_bench.sh $CONN --no-nc-overrides --cpu 4 --tmin 10
```

Generates every row of the config, syncs to the board, builds, runs each project on every dataset, fetches the logs. Add `--dry-run` to print the settings and stop; every real run opens with the same overview, each setting marked `[default]` or with the flag that set it.

**Writes** `tests/pure_xdsl_blis_rvv/logs/run_<YYYY-mm-dd_HHMMSS>/`:

| File | What |
|------|------|
| `all_raw_<dt>_<method>_<suffix>.txt` | one row per case: shape, mc/nc/kc, `RVV_Time_ms` (mean time per whole call), GOPs, Status |
| `reps_*.txt.gz` (per project) | every timed rep, in the order the reps ran |
| `comparison.md` | the table `summarize_run.py` writes |
| `provenance.txt` | pin, governor, compiler, load, other users' busy processes, per-core busy share |

The log suffix names the build and protocol, so different settings never mix: `_noinstr_refproto` (the default), plus `_scalarpack`, `_zeroline`, `_no_nc_override` when those flags are on.

Flags that change what a number means:

| Flag | Effect |
|------|--------|
| *(none)* | uninstrumented build, **reference protocol**: no warm-up, reps until `--tmin` s (default 3), correctness after timing. The number to quote |
| `--tmin S` | time budget per case |
| `--no-nc-overrides` | ignore the config's per-shape `nc`; needed against the Reference Design (B3 ships at `nc=128`, the reference runs 384) |
| `--cpu N` | pin to core N (default 4). Core 0 is refused (`--allow-core-0` only to repeat an old run) |
| `--instr` | instrumented build: pack and compute time per rep, perf counters. Costs 11-17%: **never quote it** against the reference |
| `--no-ref-protocol` | warm-up protocol: 5 warm-up calls, then `iterations` reps |
| `--scalar-pack`, `--pack-b <v>`, `--zero-pack-lines` | packing methods, [§8](#8-compare-packing-methods) |
| `--threads N` | N-thread GEMM, [§10](#10-threads-and-the-yolo-layers) |
| `--raw-events <list>` | with `--instr`: core events per rep (`x60-l2` preset) |
| `--no-save-reps` | skip the per-rep logs (Tile Search and `reps_report.py` need them) |

## 6. Compare against the Reference Design

Use this for any number quoted against the reference (the `RVV_ukernels_benchmark` tree in `../ext/`). A reference run from another day is not comparable: run both sides back to back on the same core.

```bash
./tests/pure_xdsl_blis_rvv/runners/run_head_to_head.sh $CONN --cpu 4
```

Runs `compile_macro_bench.sh --no-nc-overrides`, then `ref_baseline/run_ref_baseline.sh --bracket`, then compares the pair. `--opt OPT|LDX` picks the reference kernel set. The reference side alone: `ref_baseline/run_ref_baseline.sh` (see [its README](../tests/pure_xdsl_blis_rvv/ref_baseline/README.md), which also covers how the two designs differ on fringe tiles). Overnight, reference only, both boards: `run_ref_sweep.sh`.

## 7. Find the best microkernel size (Tile Search)

A Tile Search times every Candidate Tile on every shape of a dataset and names the fastest tile per shape. The config has one pinned row per tile, so each tile times as it would alone.

1. **Correctness first:** `--config configs/correctness/config_yolo_correctness.json --tmin 0.2` (§4); every row PASS.
2. **The series:** one board, three launches with the rows rotated, an idle-board wait before each, the report after each:

   ```bash
   setsid nohup ./tests/pure_xdsl_blis_rvv/runners/run_yolo_sweep.sh --tmin 2 \
     > tests/pure_xdsl_blis_rvv/logs/yolo_sweep.out 2>&1 < /dev/null &
   ```

   Default config `configs/tile_search/config_yolo_search.json` (19 tiles x 50 YOLO26 shapes, about 90 min per launch). `--config` takes another search config; `--threads 8`, `--scalar-pack`, `--launches`, `--boards`, `--dry-run`. The paper's shapes (BERT B1-B5, Square S1-S5) at 1 and 8 threads in A-B-B-A order: `run_paper_sweep.sh --issue <N>` (`--issue` is required: the report copies go to `.scratch/reports/issue-<N>/`).
3. **Read it** (the sweep runs this itself into `logs/yolo_sweep_<stamp>/` and copies it to `.scratch/reports/issue-<N>/`):

   ```bash
   python3 tests/pure_xdsl_blis_rvv/readers/search_report.py logs/run_<A> logs/run_<B> logs/run_<C> \
     --compare 16x8 --compare n=4:16x4 --heatmap heat.png
   ```

   Per shape: the best tile by GOPs by mean time (average over launches, lowest and highest launch), the best single tile over the dataset, the best tile per `n`, and a grid of every tile on every shape.

## 8. Compare packing methods

| Packer | How |
|--------|-----|
| RVV vector pack | the default |
| scalar pack | `--scalar-pack` (own log suffix `_scalarpack`) |
| B-pack variants | `--pack-b row\|row-intr\|col-g1\|col-g2\|col-g4\|col-gall\|col-g1-m4` (fp32 AXPY; not with `--scalar-pack`) |
| cleared pack lines | `--zero-pack-lines`: `cbo.zero` on each whole 64-byte line before packing (AXPY only, suffix `_zeroline`) |

Vector against scalar, end to end, alternating V S S V on one core:

```bash
./tests/pure_xdsl_blis_rvv/runners/run_pack_swap.sh $CONN          # B3 and S1; --config configs/benchmark/config_pack_swap_b2b3b4.json for B2-B4
python3 tests/pure_xdsl_blis_rvv/readers/compare_pack_swap.py --vector logs/run_<V1> logs/run_<V2> --scalar logs/run_<S1> logs/run_<S2>
```

GOPs are `2·m·n·k / RVV_Time_ms`, packing included. What each packer does per tile: `python3 tests/pure_xdsl_blis_rvv/readers/pack_dims_table.py`.

## 9. Other data types

| Datatype | State |
|----------|-------|
| `fp32` | the main path: every feature, every runner |
| `f16f16f32` | FP16 A and B in memory, FP32 C (#112). Scalar packing only: the driver refuses a run without `--scalar-pack`. Needs `g++-14` (Zvfh) on the board. No trustworthy GOPs yet |
| `int8`, `i8i8i32` | the generator builds the project; no runner or correctness config uses it |

Correctness of f16f16f32: §4 with `--config configs/correctness/config_issue112_mixed_correctness.json --scalar-pack`.

**Hand-over kernels:** standalone fp32 and f16f16f32 microkernels with Héctor's GEMM argument order, plus a naive reference and a test, for use outside this harness:

```bash
python3 tests/pure_xdsl_blis_rvv/runners/gen_handover_ukernels.py --out /tmp/handover
```

## 10. Threads and the YOLO layers

`compile_macro_bench.sh --threads N` runs each GEMM call on N OpenMP threads on cores 0..N-1 (fp32 AXPY only; `--threads 1` is the one-core code). It refuses `--cpu`, `--no-pin`, `--pack-b` other than `row`, and `--zero-pack-lines`.

| Runner (`runners/`) | What |
|--------|------|
| `run_parallel_gemm.sh` | our GEMM at 1 and 8 threads on the YOLO26 shapes, order 1T 8T 8T 1T 1T 8T, report after each launch (`parallel_gemm_report.py`) |
| `run_hector_gemm.sh --src <clone>` | Héctor's GEMMLib_yolo (`b292c92`), unedited, at 1 and 8 threads, report after each launch (`hector_gemm_report.py`) |
| `run_yolo_sweep.sh --threads 8` | the Tile Search at 8 threads |

Start them detached as in §7. 8-thread runs use cores 0-7, core 0 included: the one exception to the core rule, and it is stated next to every such number. YOLO datasets: `yolo26_k1.dat` (50 layers), `yolo26_k1_53.dat`, `yolo26_k1_n1.dat` (the three `n = 1` layers). Layers are reported in Héctor's order Y00..Y52.

## 11. Read the results

| Tool (`readers/` unless noted) | Reads | Gives |
|------|-------|-------|
| `summarize_run.py <run dir>` | `all_raw_*`, `all_perf_*` | `comparison.md` (written by every run) |
| `reps_report.py <run dir> [...]` | `reps_*.txt[.gz]` | per case: GOPs by mean time, CV, slow-rep share, then the average over launches with lowest and highest. Prints a WARNING under 3 launches and a 🔴 RED NOTE for a run made while another user was busy |
| `search_report.py` | a Tile Search's run folders | §7 |
| `compare_pack_swap.py` | vector and scalar runs | §8 |
| `ref_baseline/compare_ref_baseline.py` | our run and a reference run | per-case comparison, with a WARNING when pins differ or the runs are over an hour apart |
| `slow_state_survey.py` | every committed per-rep log | the B3 two-speed survey (#47) |

Quote **GOPs by mean time**: total operations over total time of the whole call, packing included, as the reference computes it. Say next to it which build, protocol, core and board it came from.

## 12. Rules for a number you quote

From [agents/execution.md](agents/execution.md), "House rules" (issue #39):

1. Never core 0 (default 4). The exception is 8-thread runs, which say so.
2. One launch is not a result. A gap under ~3% needs both settings inside one launch, repeated at least 3 times, in A-B-B-A order when they need separate launches.
3. Every quoted number comes from a committed log under `tests/pure_xdsl_blis_rvv/logs/`, cited by its run folder. `test_house_rules.py` fails on a cited folder that is not committed.
4. Name the build, protocol and core (`INSTRUMENTED=0`, reference protocol, `--tmin`, `--cpu`).
5. Check who else is on the board. A timing run that decides something waits for an idle board; the runners do this themselves. A correctness run may run on a busy board. Never kill or renice someone else's job.

## 13. Run the host tests

```bash
cd tests/pure_xdsl_blis_rvv && ../../.venv/bin/python -m pytest
```

681 tests, about 17 minutes (the idle-gate tests really sleep). The full regime (host tests, snapshot compare, feature baseline on the board): [Test regime](../tests/pure_xdsl_blis_rvv/README.md#test-regime). Use the `.venv` Python; the conda one lacks `xdsl`. Seven tests (`*_on_board`, in `test_pack_correctness.py` and `test_perf_counters.py`) compile on the board through `fej` and skip when it is not reachable. Host only: `-k "not on_board"`. The feature smoke tests alone (seconds): `pytest tests/test_features.py`.

Where they are: `tests/` for the tools, `experiments/issueNN/` for each past investigation. The folder [README](../tests/pure_xdsl_blis_rvv/README.md) maps each feature to its tests.

**Never run the tests during a board series.** They rewrite `pj_gen/` and `ref_baseline/generated/`, and some start runners with `--dry-run`. `ref_baseline/generated/` is tracked: `git diff` it after a test run.

## 14. Add a new experiment

1. Open an issue, then a branch named after it, cut from `pure-xdsl`.
2. Put everything for it in `tests/pure_xdsl_blis_rvv/experiments/issue<N>/`: its configs, its run script, its report script, its tests.
3. In a run script, take the project folder as `HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"`, `source "${HERE}/board_pin.sh"` for every board access (ssh, rsync, other users, the idle gate, series folder names), and pass configs as `--config experiments/issue<N>/<config>.json`. In Python, `HERE = Path(__file__).resolve().parents[2]`.
4. Prefer an existing runner with `--config` over a new script: `run_yolo_sweep.sh` already does launches, rotation, the idle wait and the report.
5. Correctness first (§4), then timing under the house rules (§12). Commit the run folders you quote.

## 15. Troubleshooting

| Symptom | Cause, fix |
|---------|------------|
| `ModuleNotFoundError: No module named 'xdsl'` | wrong Python; use `.venv/bin/python` or `source .venv/bin/activate` |
| a series stalls at "waiting" | another user's job is busy on the board; the runner waits (`--wait-min`). Check with `ps` on the board; do not kill it |
| two series at once overwrite each other | the boards share one NFS home: give each series its own `--workspace` and its own checkout here (`git worktree`, `.venv` linked in) |
| `--cpu 0` refused | house rule 1; `--allow-core-0` only to repeat an old core-0 run |
| `--tmin` refused | the instrumented build cannot run the reference protocol; drop `--tmin` or `--instr` |
| `Status=SKIP` everywhere | the config has `skip_ref_run` on; use a config from `configs/correctness/` |
| f16f16f32 run refused | needs `--scalar-pack` |
| an old issue or log names `run_issue55.sh`, `config_yolo_search.json`, ... | files moved on 2026-11-21 (#127): see `tests/pure_xdsl_blis_rvv/experiments/README.md` and `configs/` |
