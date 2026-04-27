# Mixed Precision (i8i8i32) Project Handoff

This document provides a detailed overview of the mixed-precision implementation for RISC-V Vector (RVV) microkernels within this repository. It is intended for the next AI agent or developer to continue the integration and optimization of these kernels.

## 1. Project Context
The goal is to generate highly optimized `i8 * i8 -> i32` GEMM microkernels using MLIR/xDSL, targeting hardware with **VLEN=256** (e.g., SpaceMiT K1).

### Current Status
- **Success**: We have functional MLIR-to-C generation for 8x8 mixed-precision tiles.
- **Limitation**: The logic is currently contained in standalone tests (`tests/test_mixed_precision_macc.py`) and is not yet integrated into the main `compile.sh` benchmarking pipeline.
- **Challenge**: The "Widen-Once MAC" strategy is extremely fast but hits register pressure limits at larger tile dimensions (e.g., 32x8).

---

## 2. Key Files & Resources

| File Path | Description |
| :--- | :--- |
| `docs/mac_method_comparison.md` | **Critical Reading.** Compares MAC strategies and documents the register pressure bottleneck. |
| `tests/test_mixed_precision_macc.py` | The main generator script for an 8x8 `i8i8i32` kernel. |
| `tests/test_mixed_precision_arith.py` | Lower-level test for individual RVV mixed-precision intrinsics. |
| `src/xdsltemplate/transforms/rvv_to_emitc.py` | Contains the lowering patterns from RVV MLIR to EmitC (C intrinsics). |
| `tmp_int_codeGen/` | Local directory containing generated `.mlir` and `.c` files. |

---

## 3. Technical Implementation Details

### Data Layout & Types
For **VLEN=256**, we use `mf4` (fractional LMUL) types to ensure vector registers hold exactly 8 elements:
- `vint8mf4_t`: Input A row (8 elements).
- `vint16mf2_t`: Intermediate product (after first widening).
- `vint32m1_t`: Accumulator (8 elements).

### The "Deeply Optimized" Inner Loop
The current implementation uses a two-instruction accumulation sequence per column:
1. `vwmacc.vx`: Multiplies `i8` (vector) by `i8` (scalar) and adds to an `i16` intermediate.
2. `vwadd.wv`: Widens the `i16` intermediate to `i32` and adds to the main accumulator.

**Note**: To maximize performance, we use the "Widen-Once" strategy where A is sign-extended to `i16` *once* outside the inner loop.

---

## 4. How to Run & Verify

### Generate Kernels Locally
```bash
# Set PYTHONPATH and run the generator
PYTHONPATH=src .venv/bin/python tests/test_mixed_precision_macc.py
```
This will:
1. Build the MLIR module.
2. Lower it to EmitC.
3. Call `mlir-translate` to produce `tmp_int_codeGen/mixed_precision_macc.c`.
4. Perform an intrinsic check (looking for `__riscv_vwmacc`, etc.).

### Inspect Assembly
#### Local Inspection
Use the dedicated script to cross-compile kernels to assembly locally:
```bash
# In tests/api_gen_cpp
./dump_asm.sh 8x8_b0_i8i8i32
```
Files are saved to `tests/api_gen_cpp/asm/`.

#### Remote Benchmark & Assembly Dump
To generate, compile, and benchmark mixed-precision kernels on the physical RISC-V board (via jump-host), use the following command:

```bash
REMOTE_PASS="" REMOTE_HOST="fej" REMOTE_PORT="3322" ./compile.sh \
    --jump --riscv-user jlei --riscv-ip jbpi2 --riscv-port 22 \
    --families "8,8" --vlen 256 --precision i8i8i32 \
    --dump-asm 8x8_b1_i8i8i32
```

**Command Breakdown:**
- `REMOTE_PASS="" REMOTE_HOST="fej"...`: Environment variables to coordinate the SSH tunnel through the jump-host (`fej`).
- `--jump`: Signals the script to use the multi-hop SSH logic.
- `--riscv-user/ip/port`: Point to the final destination board (`jbpi2`).
- `--precision i8i8i32`: Explicitly sets the generator and benchmark driver to use INT8/INT32 mixed precision.
- `--dump-asm 8x8_b1_i8i8i32`: Instructs the pipeline to compile the specified kernel natively on the RISC-V hardware and `rsync` the resulting `.s` assembly file back to the local `tests/api_gen_cpp/asm/` directory for manual audit.

---

## 5. Roadmap for Next Agent (Execution Plan)

### Phase 1: API Refactoring
Move the generator logic from `tests/test_mixed_precision_macc.py` into a proper class within `src/xdsltemplate/`. It should be parameterized by `MR`, `NR`, and `precision`.

### Phase 2: Structural Spilling Management
Implement "Chunked Widening" for the A-tile.
- **Problem**: Widening a 32x8 tile requires extending 32 elements to `i16`, consuming twice the registers. 
- **Solution**: Modify the MLIR generation to widen only a subset of the A-tile at a time, or interleave the `vsext` instructions to stay within the 32-register budget.

### Phase 3: Pipeline Integration
Update `compile.sh` to support `--precision int8`.
- Ensure it picks up the correct C templates.
- Link against appropriate benchmark drivers (INT8 versions).
- Validate performance gains on the SpacemiT K1 hardware.

---

> [!IMPORTANT]
> **Register Pressure Warning**: If you see `vle8_v` instructions paired with immediate `vse8_v` (spills) in the assembly, the tile size is too large for the "Widen-Once" strategy. You must switch to a chunked load approach.
