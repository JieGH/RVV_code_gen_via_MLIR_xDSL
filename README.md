# RVV xDSL Microkernel Runner

Generate RVV GEMM microkernels locally, sync them to a RISC-V board, run benchmarks, and fetch results back to your host.

## Quick Start

1. Install local tools
- `git`, `cmake`, `ninja`, a C/C++ compiler, `python3`, `uv`, `ssh`, `rsync`

2. Install remote tools (on the RISC-V board)
- `gcc-14`, `gfortran-14`, `make`, `git`, `ssh`

3. Create the local Python environment
```bash
make install
```

4. Edit `compile.sh` (top of file) with your board info
- `RISCV_USER`
- `RISCV_REMOTE_IP`
- `RISCV_SSH_PORT` (optional)
- Optional tuning: `FAMILIES`, `VLEN_BITS`, `KC_PROFILE`, `DATASETS`

5. Run the pipeline
```bash
REMOTE_PASS="" REMOTE_HOST="fej" REMOTE_PORT="3322" ./compile.sh --jump --riscv-user jlei --riscv-ip jbpi2 --riscv-port 22 --families "32,32" --vlen 256 --precision fp32
```

Example with explicit connection and tuning options:
```bash
./compile.sh --riscv-user jlei --riscv-ip jbpi2 --riscv-port 22 --families "2,2" --vlen 256
```

That’s it. The script will:
- Build `mlir-translate` locally if missing.
- Build OpenBLAS and BLIS on the board if missing.
- Generate kernels, compile, benchmark, and pull results back.

## Outputs

Results are saved to `tests/output/` on your local machine:
- `*.txt` logs
- `*.csv` sweep tables
- `*.png` / `*.pdf` plots

## Common Variations

Use a different MLIR build directory:
```bash
bash compile.sh --mlir-build-dir ../llvm-project
```

Override families or VLEN:
```bash
bash compile.sh --families "4,4" --vlen 256
```

## One-Off Setup Only

Local MLIR/LLVM build:
```bash
bash setup_env.sh --local-mlir-only --mlir-build-dir ../llvm-project
```
Remote OpenBLAS/BLIS build:
```bash
bash setup_env.sh --remote-only --vlen 256 --riscv-workspace xdsl_rvv_microkernel
```

## Inspecting Assembly

You can dump and inspect the actual RISC-V assembly generated for any microkernel.

### Local Assembly Dump
Use the dedicated script to cross-compile kernels to assembly locally:
```bash
cd tests/api_gen_cpp
./dump_asm.sh 4x4_b0_fp32       # Dump a single kernel
./dump_asm.sh --all              # Dump all fp32 kernels
```
Files are saved to `tests/api_gen_cpp/asm/`.

### Remote Assembly Dump
Alternatively, compile natively on the board and pull the `.s` file back:
```bash
./compile.sh --dump-asm 4x4_b0_fp32
```

## Side Note / Dev Note

To compile in **jump-host mode** (x86 → fej → jbpi2), the following command is used:
```bash
REMOTE_PASS="" REMOTE_HOST="fej" REMOTE_PORT="3322" ./compile.sh --jump --riscv-user jlei --riscv-ip jbpi2 --riscv-port 22 --families "32,32" --vlen 256 --precision fp32

REMOTE_PASS="" REMOTE_HOST="fej" REMOTE_PORT="3322" ./compile.sh --jump --riscv-user jlei --riscv-ip jbpi2 --riscv-port 22 --families "8,8" --vlen 256 --precision fp32
```

```
REMOTE_PASS="" REMOTE_HOST="fej" REMOTE_PORT="3322" ./tests/packing/compile_packing.sh --jump --riscv-user jlei --riscv-ip jbpi2 --riscv-port 22
```


05 may test rvv packing
```
REMOTE_PASS="" REMOTE_HOST="fej" REMOTE_PORT="3322" bash tests/gemm_blis_xdsl/compile_merged.sh --jump --riscv-user jlei --riscv-ip jbpi --riscv-port 22

```



```
REMOTE_PASS="" REMOTE_HOST="fej" REMOTE_PORT="3322" ./compile.sh \
    --jump --riscv-user jlei --riscv-ip jbpi2 --riscv-port 22 \
    --families "8,8" --vlen 256 --precision fp32 \
    --dump-asm 8x8_b1_fp32
```