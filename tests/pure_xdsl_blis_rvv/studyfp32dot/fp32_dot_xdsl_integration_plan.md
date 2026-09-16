# Integrating Emulated FP32 DOT Microkernel into xDSL

This plan details the steps to fully integrate the emulated FP32 DOT-product matrix engine layout (which achieved 4.74 GFLOPS) into the xDSL/MLIR code generation pipeline. 

## The Target C Code (What we want to generate)
To understand what we need to generate in xDSL (MLIR), here is the target C code from our study (`rvv_dot_gemm_fp32.c`) that we proved works effectively on the SpacemiT K1. We must generate the MLIR equivalent of these exact loops.

### 1. The Target Packing Code
We must generate Column-First packing for Matrix B (just like INT8 `vmadot`), and Row-Major packing for Matrix A. This differs completely from the standard AXPY packing.

**Matrix A Packing (`pack_A_fp32`)**:
```c
// Target Pack A (Row-Major Block)
void pack_A_fp32(int mc, int kc, const float *M, int ldM, float *Ar, int MR_) {
    int p = 0;
    for (int i = 0; i < mc; i += MR_) {
        int mr = (mc - i < MR_) ? (mc - i) : MR_;
        for (int j = 0; j < kc; j += KR) {
            int kr = (kc - j < KR) ? (kc - j) : KR;
            // Store MR rows, each with KR depth values (row-first within tile)
            for (int ii = 0; ii < mr; ii++) {
                for (int jj = 0; jj < kr; jj++)
                    Ar[p++] = Mcol(i + ii, j + jj);   // A(row, k_depth)
                for (int jj = kr; jj < KR; jj++)
                    Ar[p++] = 0.0f;                   // zero-pad K edge
            }
            for (int ii = mr; ii < MR_; ii++)
                for (int jj = 0; jj < KR; jj++)
                    Ar[p++] = 0.0f;                   // zero-pad row edge
        }
    }
}
```

**Matrix B Packing (`pack_B_fp32`)**:
```c
// Target Pack B (Column-First Block for DOT layout)
void pack_B_fp32(int kc, int nc, const float *M, int ldM, float *Br, int NR_) {
    int p = 0;
    for (int j = 0; j < nc; j += NR_) {
        int nr = (nc - j < NR_) ? (nc - j) : NR_;
        for (int i = 0; i < kc; i += KR) {
            int kr = (kc - i < KR) ? (kc - i) : KR;
            // COLUMN-FIRST: for each output column ii, store KR depth values.
            for (int ii = 0; ii < nr; ii++) {
                for (int jj = 0; jj < kr; jj++)
                    Br[p++] = Mcol(i + jj, j + ii);   // B(k_depth, col)
                for (int jj = kr; jj < KR; jj++)
                    Br[p++] = 0.0f;                   // zero-pad K edge
            }
            for (int ii = nr; ii < NR_; ii++)
                for (int jj = 0; jj < KR; jj++)
                    Br[p++] = 0.0f;                   // zero-pad col edge
        }
    }
}
```

### 2. The Target Microkernel Code
We must generate an MLIR loop that replicates the strided gather and broadcast FMAs. 
```c
// Target Microkernel (Inside K Loop)
void microkernel_rvv_dot_fp32(int mr, int nr, int kc,
                              const float *Ar, const float *Br,
                              float *Cr, float beta, int ldC)
{
    size_t vl = __riscv_vsetvl_e32m2(NR);
    vfloat32m2_t vC0, vC1, vC2, vC3;
    
    // ... Initialization of vC0..vC3 omitted for brevity ...
    
    const float *Ar_ptr = Ar;
    const float *Br_ptr = Br;

    // K outer loop: stride KR
    for (int k = 0; k < kc; k += KR) {
        // Lane loop: l = 0..KR-1
        for (int l = 0; l < KR; l++) {
            // Load B lane: strided by KR floats (16 bytes).
            vfloat32m2_t vB_lane = __riscv_vlse32_v_f32m2(&Br_ptr[l], KR * sizeof(float), vl);
            
            // Broadcast scalar A and accumulate
            vC0 = __riscv_vfmacc_vf_f32m2(vC0, Ar_ptr[0 * KR + l], vB_lane, vl);
            vC1 = __riscv_vfmacc_vf_f32m2(vC1, Ar_ptr[1 * KR + l], vB_lane, vl);
            vC2 = __riscv_vfmacc_vf_f32m2(vC2, Ar_ptr[2 * KR + l], vB_lane, vl);
            vC3 = __riscv_vfmacc_vf_f32m2(vC3, Ar_ptr[3 * KR + l], vB_lane, vl);
        }
        Ar_ptr += MR * KR;
        Br_ptr += KR * NR;
    }
    
    // ... Store vC0..vC3 omitted for brevity ...
}
```

## Proposed xDSL Implementation (Python)
To generate the above C code via our MLIR pipeline, we need to add the following Python code to our xDSL builders:

---

### [MODIFY] `tests/gemm_blis_xdsl/pack_builder.py`
We need to add MLIR builder logic for the `DOT_FP32_LANE_Pack` packing method. We will create Python functions that construct the `scf.for` and `memref.store` operations for the loops shown in the C code above.

1. **Add `build_pack_RB_dot_fp32_lane_if` & `build_pack_RB_dot_fp32_lane_else`**: 
   - These Python functions will emit MLIR loops for Matrix A that iterate `mc` by `MR`, and `kc` by `KR`, storing elements into the `Mc` buffer matching `pack_A_fp32` logic.
2. **Add `build_pack_CB_dot_fp32_lane_if` & `build_pack_CB_dot_fp32_lane_else`**: 
   - These Python functions will emit MLIR loops for Matrix B that iterate `nc` by `NR`, and `kc` by `KR`. It will compute the inner index specifically for Column-First DOT layout matching `pack_B_fp32` logic.
3. **Update `build_pack_module`**: 
   - Add a branch: `if method == "DOT_FP32_LANE_Pack":` to route execution to these newly created Python builder functions instead of the standard AXPY builders.

---

### Root Cause Analysis for FP32 DOT Microkernel Mismatches

After extensive debugging of the MLIR generation and the C++ dispatcher code, I have identified the exact root cause of the `Diff: 52.69` mismatches (where `RVV` values were seemingly random and entirely incorrect, like `26.09` vs `78.78`).

The issue was a **Function Signature Mismatch** between the C++ dispatcher and the MLIR-generated microkernel, which caused undefined behavior, skipped loops, and stack corruption at runtime.

### 1. The Signature Mismatch
The test framework's macro dispatcher expects all microkernels to follow a standard 10-argument function signature. It defines a function pointer type like this:
```cpp
typedef void (*xdsl_kernel_b0_t)(void* ctx, size_t kc, float* alpha, float* A, size_t lda, float* B, size_t ldb, float* beta, float* C, size_t ldc);
```
And it invokes the kernel by passing exactly 10 arguments:
```cpp
func(NULL, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
```

However, when I originally wrote the `build_gemm_dot_fp32` builder in `test_codeGeneration.py`, I simplified the MLIR function definition to only accept 9 arguments, intentionally omitting the dummy `ctx` (or `num_k_tiles`) argument at the beginning:
```python
input_types = [index_t, memref_f32, memref_f32, index_t, memref_f32, index_t, memref_f32, memref_f32, index_t]
# which maps to (kc, alpha, A, lda, B, ldb, beta, C, ldc)
```

### 2. Runtime Consequences (RISC-V Calling Convention)
Because the dispatcher pushes 10 arguments but the callee expects 9, the RISC-V registers get mismatched:
*   The dispatcher puts `NULL` (0) in register `a0`.
*   The callee reads `a0` as its 1st argument (`kc`). So `kc` evaluates to `0`!
*   As a result, the inner `for` loop `for (int k = 0; k < kc; ...)` runs **zero times**. The mathematical logic was never even executing!

### 3. Why the Output Was `26.09` Instead of `0.0`
You might wonder why `RVV:` reported `26.09` if the kernel did zero work.
*   The 8th argument in the callee signature is `C` (`float* v8`).
*   Due to the argument shift, the dispatcher's 8th argument (`&beta`, a pointer to a local stack variable) gets mapped to `C`.
*   When the microkernel attempts to initialize and write `0.0f` to `C`, it actually overwrites `&beta` on the stack, causing a silent buffer overflow.
*   Meanwhile, the actual heap-allocated `C` matrix was left completely untouched by the microkernel.
*   The benchmark harness `driver_benchmark.cpp` allocates `C` using `alloc_aligned` and initializes it with **random data** between `0.0` and `0.99`. The `26.09` and other numbers were simply the initial random noise leftover in memory!

### 5. Missing Zero-Padding for Partial K-Blocks (Edge Case Mismatches)
After fixing the signature and running the tests again, I noticed that `Edge_3x5x7` (where $K=7$) still failed, even though perfectly divisible sizes like `Edge_1x1x1` passed perfectly.

**The issue:** The microkernel's `for(k=0; k<kc; k+=4)` loop is heavily unrolled to always execute 4 iterations (MACs) per K-block. For a partial block like $K=7$, the loop bounds pad it to $K=8$. However, the memory packing functions (`pack_RB` and `pack_CB`) only pack the *valid* 7 elements, leaving the 8th element untouched.
Because the `Ac` and `Bc` buffers are allocated via `malloc` (in `driver_benchmark.cpp`), the untouched 8th element contains random garbage memory. The microkernel then multiplies this garbage and accumulates it into `C`, causing mismatches.

**The fix:** I added a `memset(Ac, 0, ...)` and `memset(Bc, 0, ...)` step inside `gemm_dispatch_xdsl.cpp` directly before the dispatch loop starts. This ensures that any padded elements beyond the valid elements are strictly `0.0f`. Since $0 \times 0 = 0$, the extra microkernel MAC operations correctly contribute nothing to the final result.

## User Review Required
Please review the root cause explanation above. Since I already applied the fixes in `test_codeGeneration.py` and `generate_pure_blis_project.py`, you can test it by proceeding with the execution of the compilation script.

---

### [MODIFY] `tests/test_codeGeneration.py`
We need to write the Python logic that generates the `vfmacc.vf` microkernel loop.
1. **Update the signature of `build_gemm`**: 
   - Add the `method: str` parameter.
2. **Add a new generator function: `build_gemm_dot_fp32(...)`**:
   - Instead of emitting `vle32.v` (contiguous load) for B, this Python function will emit `rvv.vlse` (strided load) from the `RVV` dialect.
   - It will construct a loop where, for each `k` iteration, it emits `rvv.vlse` for the B vector, and then emits `MR` consecutive `memref.load` and `rvv.vfmacc_vf` operations for Matrix A, perfectly matching `microkernel_rvv_dot_fp32`.
3. **Route the call**: 
   - Inside `build_gemm`, add `if method == "DOT_FP32_LANE_Pack": return build_gemm_dot_fp32(...)`.

---

### [MODIFY] `tests/pure_xdsl_blis_rvv/generate_pure_blis_project.py`
This script orchestrates the generation process.
1. Update all calls to `build_gemm` to pass `method=config.method`.
2. Update `generate_header` to emit C declarations for `pack_CB_v_if_DOT_FP32_LANE_Pack_fp32_NR...` when `config.method == "DOT_FP32_LANE_Pack"`.
