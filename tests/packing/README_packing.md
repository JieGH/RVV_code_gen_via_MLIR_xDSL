# xDSL/RVV Vectorized Packing — Comprehensive Guide

This document is the unified reference guide for the xDSL/RVV packing infrastructure. It covers the overall development strategy, practical compilation instructions, theoretical packing methods, specific details on the SpacemiT K1 matrix engine, and low-level code mappings.

---

## 1. Overview & Strategy

The vectorized packing infrastructure leverages **xDSL and MLIR EmitC** to translate high-level matrix reshaping algorithms directly into RISC-V Vector (RVV) intrinsics. The architecture focuses on parameterized generation to support both Row-Major and Column-Major layouts dynamically via `strided` and `unit-stride` vector memory operations.

### Current Implementation Status
- ✅ **Method 1 (FP32 AXPY):** Fully implemented, verified against BLIS scalar ground truth, and benchmarked.
- ✅ **Method 3 (INT8 DOT Lane):** Packing implemented and verified against the strict zero-padded ground truth. 
- ❌ **Method 2 (INT8 DOT Widen) & Method 4 (SpacemiT IME 2D Tile):** Pending implementation in the xDSL builder.

### The Builder Architecture (`pack_builder.py`)
The generator emits MLIR IR using `scf.ForOp` loops and custom RVV dialect ops (`vlse32`, `vle32`, `vse32`, etc.). It produces a single unified C file containing multiple branch kernels (e.g., `_if` for strided loads and `_else` for unit-stride). This cleanly handles transpositions without branching inside the innermost loop.

---

## 2. Compilation & Usage Guide

The single source of truth for generating, compiling, and testing the design is the `compile_packing.sh` script.

### Default Compilation
To generate C code with default block sizes (`NR=8`, `MR=8`) and run tests on the RISC-V board:
```bash
REMOTE_PASS="<your_password>" REMOTE_HOST="fej" REMOTE_PORT="3322" \
./tests/packing/compile_packing.sh \
    --jump \
    --riscv-user jlei \
    --riscv-ip jbpi2 \
    --riscv-port 22 
```

### Custom Block Sizes and Types
You can configure block dimensions (`NR`, `MR`), hardware `VLEN`, the target data type (`dtype`), and the packing method using CLI flags:
```bash
REMOTE_PASS="<your_password>" REMOTE_HOST="fej" REMOTE_PORT="3322" \
./tests/packing/compile_packing.sh \
    --NR 16 --MR 16 --vlen 256 \
    --dtype i8 \
    --method DOT_INT8_LANE_Pack \
    --jump \
    --riscv-user jlei --riscv-ip jbpi2 --riscv-port 22 


REMOTE_PASS="" REMOTE_HOST="fej" REMOTE_PORT="3322" ./tests/packing/compile_packing.sh     --NR 8 --MR 8     --dtype fp32     --method AXPY_FP32_Pack     --jump --riscv-user jlei --riscv-ip jbpi2 --riscv-port 22


```

### Direct SSH Execution
If you are on the same network as the board, bypass the jump host:
```bash
./tests/packing/compile_packing.sh \
    --direct \
    --riscv-user jlei --riscv-ip jbpi2 --riscv-port 22
```

---

## 3. Packing Methods Reference

The design categorizes packing layouts based on the target hardware instruction.

### Method 1 — AXPY Flat Micro-Panel Packing (FP32)
*   **Target:** `vfmacc_vf_f32m1` (RVV)
*   **Layout:** 1D Horizontal Strip ($1 \times 8$ elements). Elements from the same depth slice are contiguous. 
*   **Status:** Implemented (`AXPY_FP32_Pack`).

### Method 2 — DOT Widening Packing (INT8 → INT32)
*   **Target:** `vwmul.vv` + `vwmacc.vv` chain (RVV)
*   **Layout:** 1D Vertical Strip ($4 \times 1$ elements). 4 consecutive depth values are stored per column before moving to the next column.

### Method 3 — DOT Lane Packing (INT8 → INT32)
*   **Target:** Indexed Dot Product (`vdot_lane`)
*   **Layout:** 1D Vertical Strip ($4 \times 1$ elements). Conceptually identical to Method 2 in byte layout, but the index math differs to feed the dot lane instructions properly.
*   **Status:** Implemented (`DOT_INT8_LANE_Pack`). Employs strictly padded `NR x KR` memory buffers.

### Method 4 — 2D Tile Block Packing (SpacemiT K1 IME / Intel AMX)
*   **Target:** Hardware Matrix Engine (`vmadot` on K1)
*   **Layout:** 2D Square Tile ($8 \times 8$ elements). Reads a full $8 \times 8$ 2D block row-by-row until all 64 elements of the tile are contiguous in memory.

---

## 4. SpacemiT K1 IME Deep Dive (Method 4)

The SpacemiT K1 features an Integrated Matrix Engine (IME) optimized for quantized deep learning. It operates with a vector length (VLEN) of 256 bits (32 bytes).

### The `vmadot` Instruction
The core operation is `vmadot v0, v10, v11`. It performs matrix multiplication on 2D sub-tiles mapped onto vector registers.
*   **Matrix A (`v10`)**: Treated as a **$4 \times 8$** matrix of INT8 (32 bytes).
*   **Matrix B (`v11`)**: Treated as an **$8 \times 4$** matrix of INT8 (32 bytes).
*   **Accumulator C (`v0`, `v1`)**: The $4 \times 8 \times 8 \times 4$ MAC results in a **$4 \times 4$** INT32 matrix (64 bytes). Because this requires 64 bytes, the output spans **two consecutive vector registers** starting at an even register.

### Hardware-Mandated Packing
The IME hardware **does not pack data**. It demands that data is pre-arranged sequentially in memory in 32-byte blocks. The reference scalar C code accomplishes this via triple-nested loops, walking through standard row/col-major memory to deposit perfectly interleaved $4 \times 8$ and $8 \times 4$ sub-tiles into contiguous memory chunks. 

Once packed in software, the micro-kernel executes a single, highly efficient `vle8.v` instruction to load the 32-byte chunks and fire `vmadot` in $O(1)$ time.

---

## 5. xDSL/MLIR Implementation Details

The core of the framework lies in translating scalar C operations into single RVV intrinsics.

### Memory Access Visualization (Col-Major `pack_CB`)
The standard scalar code requires 4 separate memory requests and 4 separate scalar writes to pack 4 elements. The vectorized version eliminates the innermost loop by leveraging hardware strides.

```text
Input Matrix M (Column-Major memory)
---------------------------------------------------------------------------------
... [i, j+0] ... [i, j+1] ... [i, j+2] ... [i, j+3] ...
       │            │            │            │
       +--- ldM ----+--- ldM ----+--- ldM ----+        <-- Byte stride (ldM * 4 bytes)
       \______________________________________/
           Read in ONE step via Strided Load
                    __riscv_vlse32
                          │
                          ▼
             Temporary Vector Register
             ┌───┬───┬───┬───┬───┬───┬───┬───┐
  vfloat32 = │ 0 │ 1 │ 2 │ 3 │ - │ - │ - │ - │    (Active vl = 4 / NR = 8)
             └───┴───┴───┴───┴───┴───┴───┴───┘
                          │
                          ▼
             Written in ONE step via Unit-Stride Store
                    __riscv_vse32
                          │
                          ▼
Output Buffer Mc (Contiguous packed memory)
---------------------------------------------------------------------------------
... [i, j+0] [i, j+1] [i, j+2] [i, j+3] ...           <-- Contiguous block
```

### Low-Level Code Mapping
Here is how the theoretical visualization looks when lowered from MLIR EmitC to actual RVV Intrinsics C (`pack_CB_v_if_fp32_RR8`):

```cpp
void pack_CB_v_if_fp32_RR8(float* M, float* Mc, size_t mc, size_t nc, size_t ldM) {
  size_t RR = 8;
  size_t byte_stride = ldM * sizeof(float);
  
  for (size_t j = 0; j < nc; j += RR) {                  // Outer loop (j step RR)
    size_t nr = std::min(nc - j, RR);
    size_t vl = __riscv_vsetvl_e32m1(nr);                // Configure active vector length
    
    for (size_t i = 0; i < mc; i += 1) {                 // Inner loop (i step 1)
      float* src_base = &M[(j * ldM) + i];
      
      // Vector strided load (replaces scalar jj loop)
      vfloat32m1_t vec = __riscv_vlse32_v_f32m1(src_base, byte_stride, vl);
      
      float* dst_base = &Mc[(j * mc) + (i * RR)];
      
      // Vector unit-stride store
      __riscv_vse32_v_f32m1(dst_base, vec, vl);
    }
  }
}
```
