# Scalar FP32 DOT-Product GEMM Baseline

**File:** [scalar_dot_gemm_fp32.c](file:///home/jlei/Documents/ws_MLIR/repo_xdsl_microkernels/RVV_code_gen_via_MLIR_xDSL/tests/pure_xdsl_blis_rvv/templates/scalar_dot_gemm_fp32.c)

**Purpose:** C reference design for DOT-product oriented GEMM in FP32. This
file serves as the scalar specification that the xDSL/MLIR code generator should
target. Every loop nest here maps to a corresponding MLIR dialect operation.

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

Logical shape: `[nc/NR][kc/KR][KR × NR]`

Within each `[KR × NR]` tile, data is stored **depth-first**:

```
Br[l*NR + j]  =  B(k+l, col_j)    for l=0..KR-1, j=0..NR-1
```

**Memory diagram** for KR=4, NR=16, one tile:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ B(k0,c0)..B(k0,c15)   <- depth l=0, all 16 output cols (one vector load)      │
│ B(k1,c0)..B(k1,c15)   <- depth l=1                                             │
│ B(k2,c0)..B(k2,c15)   <- depth l=2                                             │
│ B(k3,c0)..B(k3,c15)   <- depth l=3                                             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**NEON mapping (INT8):** For lane `l`, `B0 = vld1q_s8(&Br[l * NR])` loads the 16 B values.

---

## Algorithm: Microkernel

The core of the DOT-product style is the **microkernel** `microkernel_fp32_dot`. It computes an `MR × NR` output tile using `kc / KR` outer iterations:

```
for k = 0..kc step KR:                        (L6 in paper)
    Load A_row[MR][KR] from Ar                 (MR rows, KR elements each)
    for l = 0..KR-1:                           (DOT "lane" loop)
        B_lane = &Br[l * NR]                   (NR floats for this depth)
        for i = 0..MR-1:                       (A row)
            a_val = A_row[i][l]                (scalar broadcast)
            for j = 0..NR-1:                   (B column / C column)
                C_tile[i][j] += a_val * B_lane[j]
```

### NEON Intrinsic ↔ Scalar Mapping

| NEON INT8 (`quk.c`)                           | Scalar FP32 (this file)                         |
|-----------------------------------------------|-------------------------------------------------|
| `A0=vld1q_s8(&Ar[0])`                         | `A_row[0][0..3] = Ar_ptr[0..3]`                 |
| `B0=vld1q_s8(&Br[0])` (Rep+0, lane=0)         | `B_lane = &Br_ptr[0*NR]`                        |
| `C00=vdotq_laneq_s32(C00, B0, A0, 0)`         | `for j: C_tile[0][j] += A_row[0][0]*B_lane[j]` |
| `C10=vdotq_laneq_s32(C10, B0, A1, 0)`         | `for j: C_tile[1][j] += A_row[1][0]*B_lane[j]` |
| 4 lanes unrolled in NEON as Rep+0..Rep+3       | `l` loop runs 0..KR-1                           |

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
# Compile standalone validation driver
gcc -O2 -DSCALAR_BASELINE_MAIN \
    tests/pure_xdsl_blis_rvv/templates/scalar_dot_gemm_fp32.c \
    -lm -o scalar_baseline

# Run correctness check (compares against naive triple-loop reference)
./scalar_baseline
# Expected: Max absolute error vs naive: ~0.000e+00  ->  PASS
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
