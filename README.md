# RVV xDSL Microkernel Runner

Generate RVV GEMM microkernels locally, sync them to a RISC-V board, run benchmarks, and fetch results back to your host.

## Preprint Paper: Enabling RISC-V Vector Code Generation in MLIR through Custom xDSL Lowerings

https://doi.org/10.48550/arXiv.2603.17800

<img width="1046" height="581" alt="Screenshot 2026-04-27 at 19 24 58" src="https://github.com/user-attachments/assets/04a4283a-0c45-4368-ad64-f58852cc4632" />



This paper presents a compilation approach that combines MLIR with xDSL to bridge the missing lowering stages required for RVV code generation. Using custom intermediate representations and transformation passes implemented in xDSL, we systematically translate high-level operations into specialized, hardware-aware C code invoking RVV intrinsics. The resulting kernels are emitted as portable C functions that can be directly integrated into existing applications, enabling incremental adoption without modifying surrounding software stacks. We demonstrate the approach on the General Matrix Multiplication (GEMM) kernel and evaluate the generated micro-kernels on two real RISC-V platforms, the K230 and the BananaPi F3, comparing against OpenBLAS for both square-matrix benchmarks and transformer-based workloads derived from the BERT-Large model. When integrated into a matrix multiplication kernel, the proposed approach consistently outperforms OpenBLAS, reaching up to 12.2 GFLOPS compared to the baseline's 5.1 GFLOPS and providing performance improvements between 10--35\% across the evaluated workloads. These results demonstrate that combining MLIR with xDSL provides a practical pathway to portable, optimized code generation for RISC-V platforms.

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
bash compile.sh
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
