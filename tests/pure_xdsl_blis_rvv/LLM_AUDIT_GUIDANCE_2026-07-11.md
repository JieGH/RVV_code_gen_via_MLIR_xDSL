# LLM Audit Guidance: RVV GEMM xDSL Generator
**Date**: 2026-07-11
**Target Audience**: LLM Code Auditor / Architecture Reviewer
**Project**: Pure xDSL BLIS-style GEMM Code Generator for RISC-V Vector Extensions (RVV)

## 1. Project Context & What We Have Built
We are building a highly optimized Matrix Multiplication (GEMM) pipeline for RISC-V targets using MLIR and xDSL. Instead of writing inline assembly or C intrinsics, we generate the entire BLIS-style macro-kernel, memory packing logic, and micro-kernel entirely through xDSL Python builders.

**Recent Achievements:**
- Deprecated the legacy static C-string template injection system.
- Designed a unified Python generator (`generate_pure_blis_project.py`) that emits a fully self-contained C++ project containing MLIR-lowered C code.
- Added dynamic micro-kernel dispatch to perfectly handle arbitrary edge-case matrix dimensions that do not perfectly align to the `MR` x `NR` tile size.
- Debugged and fixed a massive divergence bug during macro-kernel `K`-dimension splitting by generating specialized `_b0` (initialize) and `_b1` (accumulate) micro-kernels.
- Built an automated correctness and benchmarking suite in C++ (`driver_benchmark.cpp`).

---

## 2. Algorithms to Audit (Prioritized)

Please review the mathematical and algorithmic correctness of the following components. They are prioritized by their risk of causing silent numerical corruption or segfaults.

### [Priority 1: CRITICAL] Macro-Kernel K-Dimension Splitting & Beta Accumulation
**File:** `tests/gemm_blis_xdsl/macro_builder.py` & `tests/pure_xdsl_blis_rvv/generate_pure_blis_project.py`
- **Mechanism**: For large matrices (e.g. $K=512$), the macro-kernel splits the $K$ dimension into smaller $KC$ cache blocks (e.g., 256). The `pc` loop iterates over these blocks. For the first block (`pc == 0`), `beta = 0.0f` to overwrite the dirty $C$ registers. For subsequent blocks (`pc > 0`), `beta = 1.0f` to accumulate the dot products into $C$.
- **Audit Task**: Verify that the C++ dispatch wrapper correctly routes `beta == 0.0f` to the `_b0` MLIR kernel and `beta == 1.0f` to the `_b1` MLIR kernel. Ensure that the `BetaMode.B0` and `BetaMode.B1` implementations in `test_codeGeneration.py` correctly initialize registers to zeros vs loading from `C` memory.

### [Priority 2: HIGH] Branchless Vectorized Memory Packing
**File:** `tests/gemm_blis_xdsl/pack_builder.py`
- **Mechanism**: The system packs the original Row-Major / Column-Major matrices $A$ and $B$ into optimized `Ac` and `Bc` sub-panels for contiguous memory access in the micro-kernel. It uses `vle32.v` and `vse32.v` with explicit `vl` (vector length) constraints to prevent out-of-bounds reads.
- **Audit Task**: Check the pointer math and the vector length calculation. Specifically, verify that `rem = arith.SubiOp(nc, j)` and `vl = min(rem, RR)` perfectly prevents reading past the boundaries of the source matrix when $M$ or $N$ is not a multiple of the packing block size.

### [Priority 3: HIGH] Macro-Kernel Pointer Math and Offset Strides
**File:** `tests/gemm_blis_xdsl/macro_builder.py`
- **Mechanism**: The macro kernel contains 5 nested loops (`jc`, `pc`, `ic`, `jr`, `ir`). Inside the innermost loops, it calculates the offsets into the packed `Ac` and `Bc` buffers before passing them to the micro-kernel dispatch.
- **Audit Task**: Verify the calculation of `off_Ac = ir * kc` and `off_Bc = jr * kc`. Ensure that advancing `ir` by `MR` correctly jumps over a full panel of size `MR * kc` elements, aligning perfectly with how `pack_builder.py` stores the data.

### [Priority 4: MEDIUM] Micro-Kernel Vector Register Allocation & FMAC
**File:** `tests/gemm_blis_xdsl/test_codeGeneration.py`
- **Mechanism**: Computes the outer product of `A` and `B` tiles using RVV `vwmacc.vx` (or `vfmacc.vf`). It holds the $C$ accumulators in multiple dual-vector registers.
- **Audit Task**: Ensure that the loop over `kc` correctly loads `A` (vector load) and `B` (scalar load/broadcast), and that the vector length (`vl`) is consistent during the multiply-accumulate operations to prevent tracking garbage data in the tail ends of the vectors.

### [Priority 5: LOW] C++ Validation Harness & Thresholds
**File:** `tests/pure_xdsl_blis_rvv/templates/driver_benchmark.cpp`
- **Mechanism**: Runs the identical $M \times N \times K$ size through a naive 3-loop scalar C++ implementation (`gemm_ref`) to establish a ground truth. Compares the Frobenius norm error between the RVV generated buffer and the reference.
- **Audit Task**: Ensure the error calculation `sqrt(error) / sqrt(nrm) < 1e-4` is mathematically sound for single-precision `float` operations across $512^3$ iterations where floating-point rounding errors naturally accumulate.

---

## 3. How to Conduct the Audit
1. Start by tracing the generation pipeline in `generate_pure_blis_project.py` to understand how the MLIR modules are composed.
2. Read the `macro_builder.py` loop nests alongside `pack_builder.py` to trace the lifecycle of a single float from the user's $A$ matrix into the `Ac` cache buffer.
3. Review the execution trace of the micro-kernel dispatch in `driver_benchmark.cpp`.
4. Document any potential memory leaks, out-of-bounds pointer accesses, or scenarios where the $C$ matrix could drop updates.
