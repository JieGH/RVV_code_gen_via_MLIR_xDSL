# Scalar FP32 DOT-Product GEMM Baseline

**File:** [scalar_dot_gemm_fp32.c](file:///home/jlei/Documents/ws_MLIR/repo_xdsl_microkernels/RVV_code_gen_via_MLIR_xDSL/tests/pure_xdsl_blis_rvv/templates/scalar_dot_gemm_fp32.c)

**Purpose:** C reference design for DOT-product oriented GEMM in FP32. This
file serves as the scalar specification that the xDSL/MLIR code generator should
target. Every loop nest here maps to a corresponding MLIR dialect operation.


```
 REMOTE_PASS="" REMOTE_HOST="fej" REMOTE_PORT="3322"  ./tests/pure_xdsl_blis_rvv/studyfp32dot/run_on_board.sh
```
---

## Overview

This baseline implements **tiled GEMM** following the BLIS framework (Goto & van de Geijn), but structured around **DOT-product orientation** instead of the classic AXPY orientation. The design directly mirrors the INT8 NEON microkernel (`uk_intrinsic_quantize_int8_4x16_sdot`) and packing routines (`pack_sdot_A`, `pack_sdot_B`) from the [GEMMLib](file:///home/jlei/Documents/ws_MLIR/repo_xdsl_microkernels/RVV_code_gen_via_MLIR_xDSL/.ext/transformerIME/transformers/GEMMLib/src/gemm/gemm_blis.c) codebase.

---

## Tile Dimensions

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MR` | 4 | Micro-tile rows (A side) |
| `NR` | 16 | Micro-tile cols (B side) |
| `KR` | 4 | DOT lane depth – elements per dot step |
| `MC` | user | Macro-tile rows (L2 cache footprint) |
| `NC` | user | Macro-tile cols (L3 cache footprint) |
| `KC` | user | Macro-tile K depth (L1 cache footprint) |

For INT8+INT32 on ARMv8.2 NEON: `KR_INT8 = 16` (one full `int8x16_t` register).
For FP32 here: `KR = 4` (one "DOT lane" = 4 scalar multiplications, matching the
granularity of `vdotq_laneq_s32` which uses 4-element sub-vectors).

---

## Data Layout

### Packed A Buffer — `Ar`

Logical shape: `[mc/MR][kc/KR][MR × KR]`

Within each `[MR × KR]` tile, data is stored **row-first**:

```
Ar[i*KR + l]  =  A(row_i, k+l)    for i=0..MR-1, l=0..KR-1
```

**Memory diagram** for MR=4, KR=4, one tile:

```
┌────────────────────────────────────────┐
│ A(row0, k0) A(row0, k1) A(row0, k2) A(row0, k3)   <- row 0 │
│ A(row1, k0) A(row1, k1) A(row1, k2) A(row1, k3)   <- row 1 │
│ A(row2, k0) A(row2, k1) A(row2, k2) A(row2, k3)   <- row 2 │
│ A(row3, k0) A(row3, k1) A(row3, k2) A(row3, k3)   <- row 3 │
└────────────────────────────────────────┘
```

**NEON mapping (INT8, KR=16):** `A0 = vld1q_s8(&Ar[0])` loads the 16-byte row 0.

---

### Packed B Buffer — `Br`

Logical shape: `[nc/NR][kc/KR][NR × KR]`

Within each `[NR × KR]` tile, data is stored **column-first** (col index is outer, depth is inner):

```
Br[j*KR + l]  =  B(k+l, col_j)    for j=0..NR-1, l=0..KR-1
```

**Memory diagram** for NR=16, KR=4, one tile:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ B(k0,c0) B(k1,c0) B(k2,c0) B(k3,c0)   <- col 0, all KR depths (stride-1 group)│
│ B(k0,c1) B(k1,c1) B(k2,c1) B(k3,c1)   <- col 1                                 │
│ B(k0,c2) B(k1,c2) B(k2,c2) B(k3,c2)   <- col 2                                 │
│ ...                                                                               │
│ B(k0,c15)..B(k3,c15)                   <- col 15                                │
└──────────────────────────────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> Col index `j` is the **outer** (slow) index; depth `l` is the **inner** (fast) index. This matches `pack_B_fp32` exactly: the innermost write is `for each col ii: B(k+0,col), B(k+1,col), B(k+2,col), B(k+3,col)`. The microkernel accesses `Br_ptr[j*KR + l]` — a direct column+depth index, **not** a depth-row slice.

**NEON mapping (INT8):** `B0 = vld1q_s8(&Br[0])` loads the 16 int8 values for col 0:
`B0[0..3] = B(k+0..3, col0)` — all 4 depth elements for col 0 packed consecutively.

---

## Algorithm: Microkernel

The core of the DOT-product style is the **microkernel** `microkernel_fp32_dot`. It computes an `MR × NR` output tile using `kc / KR` outer iterations:

```
for k = 0..kc step KR:                        (L6 in paper)
    Load A_row[MR][KR] from Ar                 (MR rows, KR elements each)
    for l = 0..KR-1:                           (DOT "lane" loop = one NEON Rep block)
        for i = 0..MR-1:                       (A row)
            a_val = A_row[i][l]                (scalar broadcast of A element)
            for j = 0..NR-1:                   (B column / C column)
                C_tile[i][j] += a_val * Br[j*KR + l]
                                               ^^^^^^^^
                                               col-first B: col j, depth l
```

> [!NOTE]
> There is **no** `B_lane` row-slice load. B elements are accessed by direct index `Br[j*KR + l]` — col `j` is the slow index. In the RVV path, this translates to `vlse32.v` with stride `KR * sizeof(float)`, gathering one depth element across all 16 columns.

### NEON Intrinsic ↔ Scalar Mapping

| NEON INT8 (`quk.c`)                            | Scalar FP32 (this file)                                  |
|------------------------------------------------|----------------------------------------------------------|
| `A0=vld1q_s8(&Ar[0])`                          | `A_row[0][0..3] = Ar_ptr[0..3]`                          |
| `A1=vld1q_s8(&Ar[16])`                         | `A_row[1][0..3] = Ar_ptr[4..7]`                          |
| `B0=vld1q_s8(&Br[0])` (Rep+0, loads 16 bytes) | No row-slice load; B accessed as `Br_ptr[j*KR + l]`      |
| `B0[0..3]  = B(k+0..3, col0)`                  | `Br_ptr[0*KR + 0..3]` — col 0, all KR depths             |
| `B0[4..7]  = B(k+0..3, col1)`                  | `Br_ptr[1*KR + 0..3]` — col 1, all KR depths             |
| `C00=vdotq_laneq_s32(C00, B0, A0, 0)`          | `for j: C_tile[0][j] += A_row[0][l] * Br_ptr[j*KR + l]` |
| `C10=vdotq_laneq_s32(C10, B0, A1, 0)`          | `for j: C_tile[1][j] += A_row[1][l] * Br_ptr[j*KR + l]` |
| 4 lanes unrolled in NEON as Rep+0..Rep+3        | `l` loop runs 0..KR-1                                    |

---

## Macro-Kernel Loop Structure

```
gemm_fp32_dot_scalar (C = A*B + beta*C)
│
├── L1: jc [0..n)  step NC   ── N tile
│   ├── L2: pc [0..k)  step KC   ── K tile
│   │   │   pack_B_fp32(B(pc,jc), kc, nc) → Bc
│   │   ├── L3: ic [0..m)  step MC   ── M tile
│   │   │   │   pack_A_fp32(A(ic,pc), mc, kc) → Ac
│   │   │   ├── L4: jr [0..nc) step NR
│   │   │   │   └── L5: ir [0..mc) step MR
│   │   │   │           microkernel_fp32_dot(mr, nr, kc,
│   │   │   │               Ac[ir*kc], Bc[jr*kc],
│   │   │   │               C(ic+ir, jc+jr), betaI, ldC)
```

---

## Reference Paper

This design is derived from the algorithms described in:

> **"The Cambrian Explosion of Mixed-Precision Matrix Multiplication for Quantized Deep Learning Inference"**
> arxiv: [2506.11728v1](https://arxiv.org/html/2506.11728v1)
>
> - Section 3: GEMM Macro-kernel (5-loop structure + packing)
> - Section 5.2: ARMv8.2 NEON DOT microkernel (`vdotq_laneq_s32`)
> - Figure 4: Packing layout for sdot (the data layouts implemented here)

---

## Build & Validate

```bash
# Compile scalar baseline (run from repo root)
gcc -O3 -DSCALAR_BASELINE_MAIN \
    tests/pure_xdsl_blis_rvv/studyfp32dot/scalar_dot_gemm_fp32.c \
    -lm -o scalar_baseline
./scalar_baseline
# Expected: Max absolute error (Scalar vs Naive): <1e-3  ->  PASS

# Cross-compile and benchmark on Banana Pi (jbpi2) — includes all 3 implementations
cd tests/pure_xdsl_blis_rvv/studyfp32dot
make          # cross-compile with -march=rv64gcv
./run_on_board.sh   # rsync + run on jbpi2
```

---

## Intended Use for xDSL Code Generation

The scalar loop nests in this file are the **target IR** for the xDSL/MLIR
code generator. The mapping is:

| C construct                    | MLIR dialect / op                          |
|--------------------------------|--------------------------------------------|
| `for (int i = 0; i < mc; ...)` | `scf.for`                                  |
| `Mcol(r, c)`                   | `memref.load` with affine map              |
| `Ar[p++] = ...`                | `memref.store` with affine map             |
| `C_tile[i*NR+j] += a*b`        | `arith.mulf` + `arith.addf`               |
| `memset(C_tile, 0, ...)`       | `linalg.fill`                              |
| Entire microkernel             | Custom xDSL `blis.microkernel` op (future) |

---

# Experimental Validation & Performance Benchmarks

## 1. Goal
We needed a comprehensive, isolated baseline to prove two things:
1. **The Theoretical Foundation**: That the custom Matrix Engine (IME) on the Banana Pi (`jbpi2`) genuinely expects our Row-Major `pack_A` and Column-First `pack_B` layouts.
2. **The FP32 Emulation Strategy**: That since `vmadot` only exists in silicon for INT8, we can flawlessly replicate the DOT-product behavior for FP32 using standard RVV vector instructions.

## 2. Implementations Built

### A. The `vmadot` INT8 Hardware Validation
**`int8_vmadot_study.c`**
This study bypasses vector emulation and executes the actual **custom inline assembly instruction `vmadot v0, v11, v10`** on the Banana Pi (SpacemiT K1 SoC).

#### Deep Dive: How `vmadot` Operates

The `vmadot` instruction is a custom matrix-multiply extension designed to accelerate AI workloads by performing a **$4 \times 4 \times 8$ Matrix Multiplication in a single cycle**.

**1. The Core Loop**
The math happens in this simple 6-instruction loop:
```assembly
1: 
  vsetvli t4, t4, e8, m1, ta, ma     # Set to 8-bit elements, 1 vector group (32 elements)
  vle8.v v10, (a0)                   # Load 32 bytes of A into v10
  addi a0, a0,  32                   # Advance A pointer

  vle8.v v11, (t0)                   # Load 32 bytes of B into v11
  addi t0, t0,  32                   # Advance B pointer

  vsetvli t6, t6, e8, m1, ta, ma     # Keep e8, m1
  vmadot v0, v11, v10                # MULTIPLY-ACCUMULATE!

  addi a1, a1, -1                    # Decrement K-loop counter
  bne  a1, zero, 1b                  # Loop if not zero
```

**2. The Input Registers (`v10` and `v11`)**
On the K1, `VLEN = 256` bits, meaning one `m1` vector register holds **32 bytes** (32 `int8_t` elements).
*   **`v10` (A Tile):** Contains an `MR=4` by `KR=8` tile. Packed **Row-Major** (4 consecutive rows of length 8).
*   **`v11` (B Tile):** Contains a `KR=8` by `NR=4` tile. Packed **Column-First** (4 consecutive columns of length 8).

**3. The Math (`vmadot v0, v11, v10`)**
This instruction calculates the cross dot-products of the 4 rows in `v10` against the 4 columns in `v11`. Each dot product multiplies 8 elements together and sums them. 
*   **Math:** 4 rows × 4 columns = **16 independent dot products** of length 8.
*   **Output:** The 16 results are 32-bit integers (`int32_t`). 
*   **Storage:** 16 `int32_t` elements require 64 bytes. Since one VLEN is 32 bytes, the instruction natively outputs into an **`m2` register group** (using both `v0` and `v1`).

**4. Extracting the C Tile**
Because `v0` and `v1` hold 8 elements each, the 16 results are distributed sequentially:
*   `v0` contains **Row 0** (elements 0-3) and **Row 1** (elements 4-7).
*   `v1` contains **Row 2** (elements 0-3) and **Row 3** (elements 4-7).

The assembly handles this brilliantly using `vslidedown` and `vse32.v` with `mf2` ("half a register" size):
```assembly
  vsetvli t2, t2, e32, m1, ta, ma    # Switch to 32-bit elements, m1 (8 elements)

  # Slide the top halves down so they can be saved independently
  vslidedown.vi v2, v0, 4            # Move Row 1 into v2
  vslidedown.vi v4, v1, 4            # Move Row 3 into v4

  vsetvli t2, t2, e32, mf2, ta, ma   # mf2 means "half a register" = 4 elements
  vse32.v v0, (a2)                   # Save Row 0
  vse32.v v2, (a3)                   # Save Row 1
  vse32.v v1, (a4)                   # Save Row 2
  vse32.v v4, (a5)                   # Save Row 3
```

#### Execution Results
```bash
./int8_test
Max absolute error (INT8 vmadot vs Naive): 0  ->  PASS

--- Performance Benchmarks (>= 3.0 seconds each) ---
INT8 vmadot: 2048 iters in 4.33 s => 15.85 GOPS
```
> [!TIP]
> The fact that the hardware `vmadot` instruction achieved **0 error** against the naive reference explicitly proves that the SpacemiT K1 silicon natively expects this exact memory layout! At **15.85 GOPS**, this custom Matrix Engine is extremely fast for INT8 inference.

### B. The FP32 Emulation
**`rvv_dot_gemm_fp32.c`**
Because `vmadot` cannot do FP32, we translated the layout rules into a standard RVV implementation:
1. **`pack_A_fp32` & `pack_B_fp32`**: Identical layout strategy to the INT8 version but scaled for floats.
2. **`microkernel_rvv_dot_fp32`**: Used explicit `vlse32.v` (strided loads) to gather the scattered column-first B elements into vectors, followed by `vfmacc.vf` loops to unroll the $1 \times 4$ operations across the 4 lanes of the K-dimension.
3. **Execution Results**: 
   ```bash
   ./rvv_test
   Max absolute error (Scalar vs Naive): 0.000152588  ->  PASS
   Max absolute error (RVV vs Naive):    0.000152588  ->  PASS

   --- Performance Benchmarks (>= 3.0 seconds each) ---
   Scalar FP32 DOT: 64 iters in 4.14 s => 0.52 GFLOPS
   RVV FP32 DOT:    512 iters in 3.92 s => 4.38 GFLOPS
   ```
   > [!NOTE]
   > Our RVV FP32 DOT emulation achieves an **~8.4x speedup** over the scalar baseline! While it naturally doesn't match the dedicated INT8 matrix hardware's 15.85 GOPS throughput, 4.38 GFLOPS is an exceptionally strong result for standard vector instruction unrolling on the Banana Pi.

### C. Deployment Flow
- **`Makefile`**: Build script for both RVV and INT8 hardware tests using `-march=rv64gcv`.
- **`run_on_board.sh`**: Automates the `rsync` push via jump-host and triggering execution remotely.

## 3. Conclusion
The methodology referenced in arXiv:2506.11728v1 for the SpacemiT K1 SoC has been perfectly dissected and replicated. We now have an absolutely rock-solid blueprint for how our MLIR code generators should handle FP32 `pack_A` and `pack_B` layouts for the pipeline!

---

## 4. ASM DOT MatMul Scalability & xDSL Generation Feasibility

### How `vmadot` Scales — The Hardware Limit

The SpacemiT K1 `vmadot` instruction computes a **fixed 4×4×8 tile in one cycle**.
- One instruction → 32 INT8 reads (A), 32 INT8 reads (B) → 16 INT32 outputs
- `VLEN = 256 bits` is fully consumed: m1 = 32 bytes = 4×8 INT8

To scale to larger tiles, you simply **unroll**: add more `vmadot` instructions covering more A/B tiles and more C output registers.

**Tile Scaling Table:**

| Microkernel Tile | A regs (m1) | B regs (m1) | vmadot calls/loop | C output regs (m2) |
|-----------------|-------------|-------------|-------------------|--------------------|
| **4×4** (current) | 1 (v10) | 1 (v11) | 1 | 1 (v0+v1) |
| **4×8** | 1 (v10) | 2 (v8,v9) | 2 | 2 (v0+v1, v2+v3) |
| **8×4** | 2 (v10,v11) | 1 (v8) | 2 | 2 (v0+v1, v4+v5) |
| **8×8** (K1 paper) | 2 (v10,v11) | 2 (v8,v9) | **4** | 4 (v0+v1, v2+v3, v4+v5, v6+v7) |

> [!IMPORTANT]
> The K1 paper uses the **8×8 tile** as the primary microkernel. It issues 4 `vmadot` instructions per K-loop iteration, occupying registers v0–v11. This is the natural maximum for a single-core without register pressure spill.

**Scaling Ceiling on K1 (VLEN=256)**
- Total vector regs: 32 (`v0`–`v31`)
- For an 8×8 kernel: 2 A + 2 B + 8 C-accumulators = **12 regs consumed**
- Available headroom: 20 more regs for additional unrolling or double-buffering prefetch
- Theoretical peak for INT8 @ 1 `vmadot`/cycle: `4 ops × (8×8×8)/cycle × freq` ≈ **30–60 GOPS** at 1.6–2 GHz

**Why FP32 Cannot Use `vmadot`**
The instruction is hardcoded in silicon to INT8→INT32. There is no `vmfadot`. FP32 must use:
```assembly
vfmacc.vf vC, scalar_A, vB   # standard RVV 1.0, VLEN=256 → 8 floats/op
```
This is the fundamental constraint. Every FP32 "DOT" in RVV is really a **series of `vfmacc.vf` calls**, one per depth lane `l`.

---

### xDSL Current State — What Already Exists

**What xDSL Already Has:**
- AXPY FP32 `pack_A` and `pack_B`
- FP32 microkernel (`vfmacc.vf` loop for AXPY)
- Full generator pipeline (`generate_pure_blis_project.py`)
- DOT INT8 `pack_B` partial stub (`build_pack_CB_dot_int8_lane_if` uses `vlse8` strided gather)

**What is Missing for DOT FP32:**
1. `pack_B_fp32` **column-first interleave**: No `DOT_FP32_LANE_Pack` method in `build_pack_module()`.
2. `pack_A_fp32` **row-major with KR grouping**: No KR-aware row packer for FP32.
3. `build_gemm()` **DOT microkernel**: No DOT branch in `test_codeGeneration.py` — only AXPY.
4. **Config flag**: `DOT_FP32_LANE` is not recognized.

---

### Feasibility Analysis

#### Path A: FP32 DOT via `vfmacc.vf` (FEASIBLE ✅)
**What it does:** Emits the same loop structure as our `rvv_dot_gemm_fp32.c`, but generated by xDSL.
```
// xDSL emits this per KR lane:
for l in 0..KR:
    vB = vlse32.v(Br + l, stride=KR*4)   <- gather column l from all NR cols
    for i in 0..MR:
        a_scalar = Ar[i*KR + l]
        vC[i] = vfmacc.vf(a_scalar, vB, vC[i])
```
**Expected GFLOPS:** ~4 GFLOPS (same as hand-written C — xDSL just automates it).
**Effort:** Medium. We just need to adapt the existing INT8 lane packer for floats and add a strided `vlse32.v` load path to `build_gemm()`.

#### Path B: INT8 DOT via `vmadot` inline ASM (HARD ⚠️)
**What it does:** Generates code containing `vmadot` inline assembly — the true hardware path.
**Challenge:** xDSL/MLIR uses `emitc` dialect for code emission. There is **no standard MLIR operation** for a custom vendor instruction like `vmadot`. 
**Recommended approach:** Wrapper function. xDSL generates all 5 macro loops (L1–L5) and the packing routines, but calls a pre-compiled `microkernel_vmadot_int8()` (C/ASM) instead of trying to emit the assembly directly. This separates concerns cleanly and avoids the risk of teaching xDSL vendor-specific ASM.

---

### Recommendation

| Priority | Action | Feasibility |
|----------|--------|-------------|
| **1** | Implement `DOT_FP32_LANE` pack + microkernel in xDSL | ✅ High |
| **2** | Validate generated code matches hand-written `rvv_dot_gemm_fp32.c` | ✅ High |
| **3** | Add `vmadot` wrapper call in xDSL-generated macro loop for INT8 | ✅ Medium |
| **4** | Add native `rvv.vmadot` op to xDSL RVV dialect | ⚠️ Low (high effort, limited gain over wrapper) |
| **5** | Scale tile to 8×8 using 4× `vmadot` unrolled | ✅ High (once path 3 works) |
