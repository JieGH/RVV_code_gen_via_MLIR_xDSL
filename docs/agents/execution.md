# Execution: running the pipeline against the RISC-V board

Two connection modes reach the board, `jbpi`/`jbpi2` (BananaPi M2 Pro). Both accept the same `--riscv-user`/`--riscv-ip`/`--riscv-port` flags:

- **Direct**: SSH straight to the board. `<script> --riscv-user <user> --riscv-ip <ip> --riscv-port <port> ...`
- **Jump-host**: SSH through an intermediate host (`fej`) to reach the board, used when the board isn't directly reachable from this machine. Set `REMOTE_HOST`, `REMOTE_PORT`, and (if needed) `REMOTE_PASS` as env vars, and add `--jump`.

## Entry points

- **`./compile.sh`** (repo root): top-level pipeline — build `mlir-translate` locally if missing, build OpenBLAS/BLIS on the board if missing, generate kernels, compile, benchmark, pull results to `tests/output/`. Documented in `README.md`.
- **`tests/pure_xdsl_blis_rvv/compile_macro_bench.sh`**: standalone build & benchmark for the pure xDSL BLIS Project Generator (see `CONTEXT.md`) macrokernel project. Takes `--config <name>.json` (default `config.json`). Activates `.venv` itself; must be run from the repo root.
- **`tests/packing/compile_packing.sh`**, **`tests/gemm_blis_xdsl/compile_merged.sh`**: same jump-host convention, scoped to the packing and merged-GEMM subprojects respectively.

## Reference command (jump-host, pure xDSL BLIS macrokernel bench)

```bash
source .venv/bin/activate && REMOTE_PASS="" REMOTE_HOST="fej" REMOTE_PORT="3322" \
  ./tests/pure_xdsl_blis_rvv/compile_macro_bench.sh --jump \
  --riscv-user jlei --riscv-ip jbpi --riscv-port 22
```

Adjust `REMOTE_HOST`/`REMOTE_PORT`/`--riscv-user`/`--riscv-ip`/`--riscv-port` per the operator's actual jump-host and board; the values above are this project's current defaults, not fixed constants.

## Run protocol and build flags (`compile_macro_bench.sh`)

The reference command above needs no protocol flag: the uninstrumented build times the way the reference design does (issue #27) — no warm-up, reps until `--tmin` seconds, correctness checked after timing. Every run opens with an overview of the settings in effect, each marked `[default]` or with what set it.

| flag | effect |
|---|---|
| *(none)* | uninstrumented build, **reference protocol**, `tmin` 3.0 s. Logs `raw_*_noinstr_refproto.txt` |
| `--tmin <s>` | time budget for the reference protocol. Refused where the reference protocol is not running |
| `--no-ref-protocol` | the older **warm-up protocol** instead: 5 warm-up calls, a scalar-pack pass, then the fixed `iterations` from `config.json`. Logs `raw_*_noinstr.txt` |
| `--no-nc-overrides` | ignore the per-shape `nc` overrides in `config.json` (B3 ships at `nc=128`; the reference runs `nc=384`). The head-to-head configuration; a run without it is the shipped one. Logs get a `_no_nc_override` suffix, so the two never mix (issue #16) |
| `--instr` | instrumented debug build. Its default is the warm-up protocol, since the reference protocol cannot run there; asking it for `--ref-protocol` is refused. Logs `raw_*.txt`. Never quote against the reference |
| `--dry-run` | print the run overview and stop, without generating, syncing or running anything |

`compare_ref_baseline.py` defaults to the instrumented log name; point it at a default run with `--raw-name all_raw_<project>_noinstr_refproto.txt` (add `_no_nc_override` for a run made with `--no-nc-overrides`; the fringe column then follows on its own). Details and the measured warm-up effect: `docs/metrics.md`.

## One-time dependency builds

Not part of the per-run pipeline above — see `docs/dependencies_setup.md` for building LLVM/MLIR locally and OpenBLAS/BLIS on the board.
