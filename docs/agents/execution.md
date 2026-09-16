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

## One-time dependency builds

Not part of the per-run pipeline above — see `docs/dependencies_setup.md` for building LLVM/MLIR locally and OpenBLAS/BLIS on the board.
