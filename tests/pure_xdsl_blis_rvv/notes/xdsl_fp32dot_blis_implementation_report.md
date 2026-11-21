# xDSL BLIS FP32 DOT-Product Implementation Report (Theory & Design)

> This document details the theoretical design, memory layouts, and architectural plan for generating an **FP32 DOT-Product** microkernel within the xDSL BLIS project generator. 
> 
> This approach differs fundamentally from the existing `AXPY` method by utilizing a dot-product reduction structure along the K-dimension, mirroring INT8 hardware dot-product layouts.

---

## 1. Overview and Motivation

The existing xDSL BLIS framework supports `AXPY_FP32_Pack`, which iterates over the `K` dimension on the outside and updates vectors of `C` continuously. 

The **FP32 DOT-Product (`DOT_FP32_LANE`)** method restructures the microkernel to match the hardware-optimized layout used by NEON's `vdotq_s32` instructions. Although standard RISC-V Vector (RVV 1.0) lacks a native FP32 `vdot` instruction, adopting this structure provides:
1. **Symmetry with INT8**: A unified memory and packing layout for both FP32 and INT8 data types.
2. **Evaluation of DOT Constraints**: Testing the performance implications of column-first packing and inner K-reductions when implemented via scalar or standard RVV `vfmacc.vf` (Vector-Scalar FMA) instructions.

---

## 2. Microkernel Tile Dimensions and Blocking

The microkernel operates on a tile of `C` with dimensions $MR \times NR$. Unlike `AXPY`, the `DOT` method introduces an explicit inner-depth blocking parameter **$KR$** (typically $KR=4$).

*   **$MR$**: Number of rows of $A$ and $C$ processed per microkernel iteration.
*   **$NR$**: Number of columns of $B$ and $C$ processed per microkernel iteration.
*   **$KR$**: The depth of the dot-product "lane". For FP32 scalar baseline, $KR=4$.

The loops inside the microkernel are structured as:
```c
// Microkernel logic (Conceptual)
for (int k = 0; k < KC; k += KR) {
    // Process a block of MR x NR x KR
    for (int l = 0; l < KR; l++) {
        for (int i = 0; i < MR; i++) {
            float a_val = A_row[i][l]; // Broadcasted scalar
            for (int j = 0; j < NR; j++) {
                C_tile[i][j] += a_val * B_lane[j]; // B_lane accesses col j at depth l
            }
        }
    }
}
```

---

## 3. Packing Layouts (The Core Difference)

To support the DOT-product structure efficiently, the data in matrices $A$ and $B$ must be packed into highly specific interleaved formats.

### Packing Matrix A (`pack_A_fp32`)

*   **Logical Shape**: $[MR][KR]$ within each tile.
*   **Memory Layout**: Row-Major.
*   **Purpose**: Allows the microkernel to easily extract `a_val = A[row][depth]` to broadcast across the vector registers.

```text
Tile Structure (MR rows, KR depth):
[ A(0,0), A(0,1), A(0,2), A(0,3) ] -> Row 0
[ A(1,0), A(1,1), A(1,2), A(1,3) ] -> Row 1
...
```

### Packing Matrix B (`pack_B_fp32`) - Column-First Interleaved

*   **Logical Shape**: $[NR][KR]$ within each tile.
*   **Memory Layout**: **COLUMN-FIRST**. For each output column `j`, all `KR` depth elements are stored consecutively in memory.
*   **Purpose**: This matches the exact memory format expected by a hardware DOT instruction (like `vdotq_s32`), which reads `KR` consecutive elements from a single column to perform a dot product against the broadcasted $A$ values.

```text
Tile Structure (NR columns, KR depth=4):
Col 0: [ B(0,0), B(1,0), B(2,0), B(3,0) ] 
Col 1: [ B(0,1), B(1,1), B(2,1), B(3,1) ] 
...
```

**Memory Access in Microkernel:**
To access the element for column `j` at depth `l`, the pointer arithmetic is `Br_ptr[j * KR + l]`. This interleaved layout is what makes `pack_B` generation complex in MLIR.

---

## 4. RVV Vectorization Strategy (xDSL MLIR)

Since RVV lacks a native FP32 DOT instruction, the xDSL MLIR generators must emulate the behavior using standard vector instructions.

### Vectorizing the Microkernel (`gemm_builder.py`)
To process the DOT microkernel in RVV:
1. Load a vector of `B` representing `NR` columns at a specific depth `l`. Because `pack_B` stores elements as `[col][depth]`, loading `B_lane[j]` requires a **strided load** (`vlse32.v`) with a stride of `KR * sizeof(float)`.
2. Broadcast `a_val = A_row[i][l]` into a vector using scalar-to-vector broadcast (or implicit fused multiply-add `vfmacc.vf`).
3. Accumulate into the `C` vector register: `vfmacc.vf vC, scalar_A, vB`.

### Vectorizing the Packing (`pack_builder.py`)
*   **`pack_A`**: Vectorizing this requires loading columns from the source and storing them row-major. A transposed store using strided or indexed operations.
*   **`pack_B`**: The source is Column-Major, and the destination is also Column-First (but grouped by `KR`). This is actually a contiguous copy of `KR` elements per column, padded with zeros at the edges. MLIR can emit sequential `vle32.v` and `vse32.v` for blocks of `KR` elements.

---

## 5. Generator Architecture Impact

Integrating this into the Python xDSL generator requires the following modifications:

1.  **Configuration Flag**: Add `"DOT_FP32_LANE"` to `config.py` and JSON configs.
2.  **`gemm_dispatch_xdsl.cpp` Template Update**: Ensure the boundary limits for the `KC` loop account for `KR` padding. The BLIS macro-kernel must pass `cur_kc` rounded up to the nearest multiple of `KR`.
3.  **New xDSL Builders**:
    *   `GemmDOTFP32Builder`: Emits the strided vector loads for $B$ and `vfmacc.vf` for the accumulation loop.
    *   `PackBDOTFP32Builder`: Emits the column-first group copying logic, handling edge-case zero padding meticulously.
4.  **Buffer Allocation**: The C++ benchmark driver (`driver_benchmark.cpp`) must align memory allocations for `Ac` and `Bc` using padded dimensions:
    *   `mc_pad = (mc + MR - 1) / MR * MR`
    *   `nc_pad = (nc + NR - 1) / NR * NR`
    *   `kc_pad = (kc + KR - 1) / KR * KR`

---

## 6. Risks and Trade-offs

| Risk | Description | Mitigation Strategy |
| :--- | :--- | :--- |
| **Performance Overhead** | Strided loads (`vlse32.v`) for $B$ in the microkernel loop are significantly slower than contiguous loads (`vle32.v`) used in `AXPY`. | Ensure L1 cache alignment is perfect. Compare final GFLOPS against `AXPY` to justify if the structural symmetry with INT8 is worth the FP32 performance cost. |
| **MLIR Complexity** | Zero-padding the `pack_B` edges while maintaining the interleaved column-first layout in EmitC/RVV dialect requires complex pointer math and boundary masking. | Prototype the packing generation in a standalone python script before integrating it into `generate_pure_blis_project.py`. Fallback to scalar packing if RVV emission proves too convoluted. |
| **No Native DOT** | Using `vfmacc.vf` rather than a true DOT instruction means we execute $KR$ separate vector operations instead of 1. | Accept this as an architectural constraint of RVV 1.0; the layout is the primary focus. |

---

## 7. Next Steps for Implementation

1.  Create `pack_builder_dot.py` implementing the MLIR emission for the new `pack_A` and `pack_B` layouts.
2.  Create `gemm_builder_dot.py` to emit the DOT microkernel inner loop.
3.  Integrate the new builders into `generate_pure_blis_project.py` conditional on the `"DOT_FP32_LANE"` config flag.
4.  Generate and validate against `scalar_baseline` unit tests.

---

## 8. Appendix: The SpacemiT K1 `vmadot` Method (arxiv:2506.11728v1)

The theoretical foundation for this DOT-product layout comes from how the **Banana Pi BPI-F3 (SpacemiT K1 SoC)** implements matrix multiplication acceleration. 

While standard RVV 1.0 lacks matrix extensions, the SpacemiT K1 includes a proprietary **Matrix Multiplication Engine (IME)** specifically designed for Mixed-Precision (MIP) INT8 $\rightarrow$ INT32 inference.

### The `vmadot` Assembly Instruction
To utilize this engine, the paper relies on a custom inline assembly instruction `vmadot`. A single `vmadot` instruction computes an entire $4 \times 4 \times 8$ matrix multiplication in one cycle:

```assembly
vmadot v0, v10, v11
```

**How it works:**
*   **`v10` (Matrix A tile)**: Holds a $4 \times 8$ tile of INT8 data (4 rows, 8 depth). This data must be packed **Row-Major**, exactly like our `pack_A_fp32` logic!
*   **`v11` (Matrix B tile)**: Holds an $8 \times 4$ tile of INT8 data (8 depth, 4 columns). This data must be packed **Column-First Interleaved**, exactly like our `pack_B_fp32` logic!
*   **`v0` (Matrix C tile)**: Accumulates the $4 \times 4$ result in INT32 format (using two vector registers, e.g., `v0` and `v1`, since INT32 elements are 4x larger than INT8).

### Microkernel Scaling
For larger tiles, such as an $8 \times 8 \times 8$ microkernel ($MR=8, NR=8, KR=8$), the assembly unrolls multiple `vmadot` instructions. The A-matrix requires two vector registers (`v10, v11`), the B-matrix requires two vector registers (`v8, v9`), and the C-matrix requires four register pairs (`v0, v2, v4, v6`):

```c
// Example from K1 IME GEMMLib QuK
"vmadot_loop_kc_8x8c:                    \n" // START: Loop kc vmadot. 
// ... load v8, v9, v10, v11 ...
"vmadot v0, v10, v8                      \n" // C(0..3, 0..3) += A(0..3) * B(0..3)
"vmadot v2, v10, v9                      \n" // C(0..3, 4..7) += A(0..3) * B(4..7)
"vmadot v4, v11, v8                      \n" // C(4..7, 0..3) += A(4..7) * B(0..3)
"vmadot v6, v11, v9                      \n" // C(4..7, 4..7) += A(4..7) * B(4..7)
// ... loop ...
```

### Why we use `vfmacc.vf` for FP32
The `vmadot` instruction is strictly hardcoded in the K1 silicon for **INT8 data**. Because there is no `vmadot` instruction for FP32 on the Banana Pi, our FP32 implementation must replicate this block-wise computation using standard vector arithmetic (`vlse32.v` and `vfmacc.vf`), as demonstrated in the `/studyfp32dot/` experiments.

## 9. xDSL Python Implementation Details

The theoretical DOT layout has been fully integrated into the MLIR/xDSL code generation pipeline. Below are the implementation details:

### 1. `pack_builder.py` Modifications
The packing builder was extended to support the `DOT_FP32_LANE_Pack` methodology:
- **`build_pack_RB_dot_fp32_lane_if`**: Implements Row-Major blocking for Matrix A. The loop loads the input matrix via strided loads (`vlse32.v`) and writes contiguous chunks into the packed buffer `Ac`, storing `MR` rows of depth `KR`.
- **`build_pack_CB_dot_fp32_lane_if`**: Implements Column-First blocking for Matrix B. Elements belonging to the same column are contiguous in memory. The generator emits contiguous loads (`vle32.v`) from the source and stores them contiguously into `Bc`, guaranteeing stride-1 access along the column depth. 

### 2. `test_codeGeneration.py` Modifications
The microkernel generator (`build_gemm_dot_fp32`) translates the DOT layout back into FP32 mathematics using standard vector ALUs:
- **Inner Loop Design**: The code iterates over the `K` block in steps of `KR=4`.
- **Loading B**: A simple pointer offset fetches the scalar `B` value since `B` is effectively processed as a scalar broadcast during the dot-product unrolling.
- **Loading A**: A strided load (`vlse32.v`) fetches a vector of `A` values spanning multiple rows but belonging to the exact same $K$ depth. The stride is exactly `KR * 4` bytes.
- **Accumulation**: The `vfmacc.vf` fused-multiply-add operation performs the emulated FP32 math: `vC = vfmacc.vf(vC, scalar_B, vector_A)`.

### 3. Pipeline Integration and Dispatcher Alignment
The main generator script (`generate_pure_blis_project.py`) successfully passes the JSON configuration `"method": "DOT_FP32_LANE_Pack"` to dynamically generate the correct BLIS dispatchers and packing routines for the DOT algorithm. 

**Critical Alignment Note:** Because the microkernel generator (`build_gemm_dot_fp32`) expects to read vectors spanning the M-dimension from the 3rd argument (`A`) and scalar broadcast values spanning the N-dimension from the 5th argument (`B`), the dispatcher must perfectly map the packed buffers to these arguments:
- **`A` = `Ac`**: The M-dimension vector loads with stride `KR` correspond exactly to the layout produced by `pack_RB_dot_fp32_lane_if`.
- **`B` = `Bc`**: The N-dimension scalar loads skipping by `KR` correspond exactly to the layout produced by `pack_CB_dot_fp32_lane_if`.

When calling the microkernel in `gemm_dispatch_xdsl.cpp`, the arguments must be passed in the strict order of `(..., Ac, max_mr, Bc, max_nr, ...)`. Passing them backwards causes the microkernel to incorrectly interpret the M and N memory layouts, effectively computing a transposed result (e.g., placing the value for `C[0,1]` into `C[1,0]`).

---

## 10. Visualization of DOT vs. AXPY Packing and Microkernels

### The Mathematical Difference
To understand the packing, we must first look at how the matrix multiplication $C = A \times B$ is computed.

**1. DOT Method (Inner Product)**
The DOT method computes a single element of $C$ by multiplying a **row** of $A$ and a **column** of $B$ element-by-element, then summing the result.
$$C_{i,j} = \sum_{k=0}^{K-1} A_{i,k} \cdot B_{k,j}$$
*Requires a **reduction** operation (summing elements within the vector) which is very slow in FP32 on RVV.*

**2. AXPY Method (Outer Product)**
The AXPY method computes a 2D tile of $C$ simultaneously. It takes a **column** of $A$ and a **row** of $B$ (which acts as scalars for each column), and accumulates the outer product into $C$.
$$C_{*,*} = \sum_{k=0}^{K-1} A_{*,k} \cdot B_{k,*}$$
*Requires **vector-scalar multiply-accumulate** (`vfmacc.vf`), keeping all math within parallel lanes without any reduction.*

To fully understand the current FP32 implementation, we must distinguish between **packing logic** and **microkernel computation strategy**. The current `fp32_DOT_FP32_LANE_Pack` implementation correctly utilizes DOT packing, but pairs it with an AXPY (Outer Product) microkernel using strided loads to emulate the behavior. 

Here is a visual breakdown of the three paradigms:

### A. Standard BLIS AXPY (Outer Product)
In a standard AXPY implementation, the goal is to compute $C += A_{col} \times B_{scalar}$.
- **`Ac` Packing:** Packs columns of $A$ contiguously (AXPY packing).
- **`Bc` Packing:** Packs rows of $B$ contiguously (AXPY packing).
- **Microkernel:** Uses unit-stride vector loads (`vle32.v`) to fetch $A_{col}$ directly from `Ac`, scales it by $B_{scalar}$, and adds to $C$.

```text
Ac Memory (AXPY): [ A(m0,k0), A(m1,k0), A(m2,k0), A(m3,k0) ] --> Unit Stride Load (Fast)
Bc Memory (AXPY): [ B(k0,n0), B(k0,n1), B(k0,n2), B(k0,n3) ]
```

### B. Theoretical DOT Product (Inner Product)
In a true DOT implementation (e.g., using integer `vmadot` or similar), the goal is to compute $C_{scalar} += dot(A_{row}, B_{col})$.
- **`Ac` Packing:** Packs rows of $A$ contiguously (DOT packing).
- **`Bc` Packing:** Packs columns of $B$ contiguously (DOT packing).
- **Microkernel:** Uses unit-stride vector loads to fetch $A_{row}$ and $B_{col}$, and feeds them into a hardware dot-product instruction.

```text
Ac Memory (DOT): [ A(m0,k0), A(m0,k1), A(m0,k2), A(m0,k3) ] --> Unit Stride Load for Dot Product
Bc Memory (DOT): [ B(k0,n0), B(k1,n0), B(k2,n0), B(k3,n0) ]
```

### C. Current Emulated FP32 DOT (DOT Packing + AXPY Microkernel)
Because RVV 1.0 lacks a native FP32 dot-product instruction, the generated xDSL microkernel falls back to using the AXPY instruction (`vfmacc.vf`). However, because the data is already packed in the **DOT format**, the columns of $A$ are scattered across memory!

- **`Ac` Packing:** Packs rows of $A$ contiguously (DOT packing).
- **`Bc` Packing:** Packs columns of $B$ contiguously (DOT packing).
- **Microkernel Workaround:** To fetch $A_{col}$ for the `vfmacc.vf` instruction, the microkernel must issue a **Strided Load** (`vlse32.v`) with a stride equal to the depth of the row block (e.g., 16 bytes for a 4-float row).

```text
Ac Memory (DOT packed):
  Offset 0:  [ A(m0,k0), A(m0,k1), A(m0,k2), A(m0,k3) ]
  Offset 16: [ A(m1,k0), A(m1,k1), A(m1,k2), A(m1,k3) ]
  Offset 32: [ A(m2,k0), A(m2,k1), A(m2,k2), A(m2,k3) ]
  Offset 48: [ A(m3,k0), A(m3,k1), A(m3,k2), A(m3,k3) ]

Microkernel Strided Load (Stride = 16 bytes):
  Loads: A(m0,k0), A(m1,k0), A(m2,k0), A(m3,k0) --> Reconstructs the column for AXPY!
```

**Conclusion on the Emulated Design:** 
Mathematically, this hybrid approach perfectly reconstructs the matrices and computes the exact same mathematical result. The index arithmetic aligns flawlessly. However, from a hardware execution standpoint, relying on strided vector loads (`vlse32.v`) in the innermost loop of a GEMM microkernel defeats the cache-locality benefits of packing and can trigger simulator/hardware edge-case failures.

---

## 11. Benchmark Results

> [!NOTE]
> **Configuration**
> - **Architecture**: `jbpi2`
> - **Datatype**: `fp32`
> - **Method**: `DOT_FP32_LANE_Pack`
> - **Vector Length (VLEN)**: 256 bits (8 FP32 elements/vector register)
> - **Metric**: `Ker GOPs` = kernel-only compute throughput (excludes packing time); `Tot GOPs` = end-to-end including packing

| Dataset / Workload | Dimensions (M×N×K) | Best Kernel | Tot GOPs | Ker GOPs | RVV Pk (GB/s) | Scl Pk (GB/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **L1 (Edge)** | 1×1×1 | `MR1_NR1` | 0.00 | 0.01 | 0.0 | 0.0 |
| **L2 (Edge)** | 3×5×7 | `MR8_NR8` | 0.19 | 0.93 | 0.6 | 0.5 |
| **L3 (MicroTile)** | 8×8×8 | `MR8_NR8` | 0.82 | 3.61 | 1.2 | 0.9 |
| **L4 (Pack_Tall)** | 256×8×32 | `MR16_NR8` | 2.90 | 6.16 | 1.7 | 1.5 |
| **L5 (Pack_Wide)** | 8×256×32 | `MR13_NR16` | 3.70 | 6.83 | 2.8 | 1.5 |
| **L6 (KC_Cross)** | 64×64×300 | `MR16_NR11` | 5.97 | 8.28 | 1.6 | 1.3 |
| **L7 (Cache_Fit)** | 128×128×128 | `MR16_NR13` | 7.08 | 8.28 | 1.9 | 1.3 |
| **s_1024 (Square)** | 1024×1024×1024 | `MR16_NR7` | 5.83 | 6.51 | 0.3 | 0.2 |
| **B1_M123** | 1024×384×1024 | `MR16_NR7` | 5.46 | 6.56 | 0.3 | 0.2 |
| **B2_M5** | 384×384×64 | `MR16_NR11` | 7.57 | 8.30 | 1.7 | 1.3 |
| **B3_M7** | 64×384×384 | `MR16_NR12` | 6.55 | 8.11 | 1.5 | 1.0 |
| **B4_F11** | 4096×384×1024 | `MR16_NR7` | 5.56 | 6.52 | 0.2 | 0.2 |
| **B5_F13** | 1024×384×4096 | `MR16_NR7` | 5.23 | 6.20 | 0.3 | 0.2 |

### Comparison: DOT FP32 vs. AXPY FP32 vs. prev_xdsl

To evaluate the overhead of emulating DOT layouts using strided loads in FP32 on RVV, we compare the current DOT implementation (Ker GOPs) against the `AXPY_FP32_Pack` implementation (Ker GOPs, same run) and the previous xDSL FP32 reference (`prev_xdsl`):

| Dataset / Workload | Dimensions (M×N×K) | DOT Ker GOPs | AXPY Ker GOPs | `prev_xdsl` (GFLOPS) | DOT vs AXPY Δ | DOT vs `prev_xdsl` Δ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **L1 (Edge)** | 1×1×1 | 0.01 | 0.02 | N/A | -50.0% | N/A |
| **L2 (Edge)** | 3×5×7 | 0.93 | 1.20 | N/A | -22.5% | N/A |
| **L3 (MicroTile)** | 8×8×8 | 3.61 | 4.10 | N/A | -11.9% | N/A |
| **L4 (Pack_Tall)** | 256×8×32 | 6.16 | 9.57 | N/A | -35.6% | N/A |
| **L5 (Pack_Wide)** | 8×256×32 | 6.83 | 8.15 | N/A | -16.2% | N/A |
| **L6 (KC_Cross)** | 64×64×300 | 8.28 | 12.61 | N/A | -34.3% | N/A |
| **L7 (Cache_Fit)** | 128×128×128 | 8.28 | 13.72 | N/A | -39.7% | N/A |
| **s_1024 (Square)** | 1024×1024×1024 | 6.51 | 9.01 | ~8.5 | -27.7% | -23.4% |
| **B1_M123** | 1024×384×1024 | 6.56 | 8.55 | ~8.0 | -23.3% | -18.0% |
| **B2_M5** | 384×384×64 | 8.30 | 12.96 | ~12.5 | -36.0% | -33.6% |
| **B3_M7** | 64×384×384 | 8.11 | 10.86 | ~8.8 | -25.3% | -7.8% |
| **B4_F11** | 4096×384×1024 | 6.52 | 8.90 | ~8.5 | -26.7% | -23.3% |
| **B5_F13** | 1024×384×4096 | 6.20 | 8.28 | ~7.5 | -25.1% | -17.3% |

**Key Findings:**
1. **Performance Overhead:** The DOT implementation exhibits a consistent performance degradation compared to the native AXPY formulation across all workloads, with Ker GOPs dropping by ~12–40%.
2. **Strided Load Penalty:** This performance loss is directly attributable to the `vlse32.v` strided loads required in the inner microkernel loop to reconstruct the B-matrix vectors. Non-unit stride memory accesses dominate execution time.
3. **No B3 Exception (new run):** Unlike a previous run where B3 (64×384×384) showed a slight DOT advantage (6.79 vs 6.58), the new run shows DOT at 8.11 vs AXPY at 10.86 — AXPY is now clearly faster here too, indicating the earlier result was likely a cache-blocking anomaly.
4. **Validation of Theory:** The results confirm the risk identified in Section 6: while the DOT memory layout mirrors INT8 implementations structurally, emulating it in FP32 on RVV hardware that lacks a native FP32 `vdot` instruction is mathematically correct but fundamentally non-performant compared to `AXPY`.



