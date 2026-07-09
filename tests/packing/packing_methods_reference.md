# Packing Methods Reference
## Source: *"The Cambrian Explosion of Mixed-Precision Matrix Multiplication"* (arXiv:2506.11728v1)

This document is the **implementation roadmap** for replicating each packing method from the paper
into the xDSL/RVV design. Each method has a canonical name, the algorithm pseudocode, the output
memory layout rule, and notes on what needs to change in `pack_builder.py`.

---

> **Variable glossary** (unified across all methods)
>
> | Symbol | Meaning |
> |:---|:---|
> | `mc`, `nc` | Row/column block sizes passed to the packing function |
> | `mr`, `nr` | Micro-panel sizes = your `RR` (MR for A-side, NR for B-side) |
> | `kc` | Depth (reduction) block size |
> | `Ac`, `Bc` | Packed output buffers (your `Mc`) |
> | `A`, `B` | Source matrices (your `M`) |
> | `ldM` | Leading dimension of the source matrix |
> | `vl` | Active RVV vector length = `min(block_remaining, RR)` |

---

## Method 1 — AXPY Flat Micro-Panel Packing (FP32)

**Paper section:** §4.3 (Fig. 2, Table 2)
**Your status:** ✅ **ALREADY IMPLEMENTED** — `pack_CB_v_if/else` and `pack_RB_v_if/else`

### Name
`AXPY_FP32_Pack` — GotoBLAS2 / BLIS standard flat micro-panel packing for FP32 FMA kernels.

### When to use
- Source data type: **FP32**
- Compute instruction: `vfmacc_vf_f32m1` (RVV), `vfmaq_laneq_f32` (NEON), `svmla_lane_f32` (SVE)
- Micro-kernel type: **AXPY** (scalar × vector accumulate)

### Algorithm — B-side (Column-Block, `pack_CB`)

```
Input:  M[mc rows x nc cols], ldM, RR (= nr)
Output: Mc[nc_padded x mc] contiguous

for j = 0 to nc step RR:
    nr_actual = min(nc - j, RR)
    for i = 0 to mc step 1:          // walk rows (depth axis)
        for jj = 0 to nr_actual:     // gather nr elements across columns
            Mc[j*mc + i*RR + jj] = M[(j+jj)*ldM + i]   // col-major source
        // pad: Mc[j*mc + i*RR + nr_actual .. RR-1] = 0
```

**RVV vectorization (your current implementation):**
```
for j = 0 to nc step RR:
    vl = setvl(min(nc-j, RR))
    for i = 0 to mc step 1:
        vec = vlse32(&M[j*ldM + i], stride=ldM*4, vl)   // replaces jj loop
        vse32(&Mc[j*mc + i*RR], vec, vl)
```

### Output memory layout
```
Mc = [ col0_row0, col1_row0, ..., col(NR-1)_row0,    <- panel j=0, depth i=0
       col0_row1, col1_row1, ..., col(NR-1)_row1,    <- panel j=0, depth i=1
       ...
       col(NR)_row0, ...,                             <- panel j=1
       ... ]
```

### Key property
Elements from the **same depth slice** are contiguous. The micro-kernel reads one
scalar from `Ac` and multiplies it against an `nr`-wide vector from `Bc` (AXPY).

### xDSL builder functions (existing)
- `pack_CB_v_if_fp32_RR{N}` / `pack_CB_v_else_fp32_RR{N}`
- `pack_RB_v_if_fp32_RR{N}` / `pack_RB_v_else_fp32_RR{N}`

---

## Method 2 — DOT Widening Packing (ARMv8.0-A NEON, INT8 → INT16 → INT32)

**Paper section:** §5.1 (Fig. 3 top, using `vmull_s8` + `vmlal_s8` + `vpadalq_s16`)
**Your status:** ❌ Not yet implemented

### Name
`DOT_INT8_WIDEN_Pack` — 4-element depth-interleaved packing for multiply-widen-accumulate chains.

### When to use
- Source data type: **INT8**
- Compute chain: `vmull_s8` (INT8×INT8→INT16) + `vmlal_s8` + `vpadalq_s16` (→INT32)
- No native DOT instruction; uses 3-instruction chain to simulate 4-element dot product
- RVV equivalent: `vwmul.vv` + `vwmacc.vv` chain

### Why the layout changes
No single DOT instruction exists. To avoid redundant loads, 4 depth elements per column
must be packed **together** so the kernel can load 4 INT8 bytes at a single address.

### Algorithm — B-side (INT8, NR=8, depth groups of 4)

```
Input:  B[kc x nc] INT8, NR=8
Output: Bc

for j = 0 to nc step NR:
    nr_actual = min(nc - j, NR)
    for p = 0 to kc step 4:                // depth in groups of 4
        depth4 = min(kc - p, 4)
        for jj = 0 to nr_actual:           // columns within panel
            for pp = 0 to depth4:          // 4 depth elements per column
                Bc[j*kc + (p/4)*NR*4 + jj*4 + pp] = B[(p+pp)*ldB + (j+jj)]
```

### Output memory layout
```
Bc (per NR-panel j, depth group p):
  [ B[p,j],   B[p+1,j],   B[p+2,j],   B[p+3,j],    <- col j,   4 depth
    B[p,j+1], B[p+1,j+1], B[p+2,j+1], B[p+3,j+1],  <- col j+1
    ...
    B[p,j+NR-1], ..., B[p+3,j+NR-1] ]               <- col j+NR-1
```
**4 consecutive depth values are stored per column before moving to the next column.**

### xDSL builder function to create
`pack_CB_dot_int8_widen_NR{N}` — add `step=4` depth loop, inner `jj x pp` double loop.
New RVV ops needed: `vlse8_v_i8m1`, `vse8_v_i8m1`.

---

## Method 3 — DOT Lane Packing (ARMv8.2 NEON / SVE2, INT8 → INT32)

**Paper section:** §5.2 (Fig. 4, mr=4×nr=16) and §5.3 (SVE2 scalable variant)
**Your status:** ❌ Not yet implemented

### Name
`DOT_INT8_LANE_Pack` — 4-element depth-interleaved packing for the `vdotq_laneq_s32` / `svdot_lane_s32` instruction family.

### When to use
- Source data type: **INT8** (A and B), accumulate to **INT32** (C)
- Compute instruction: `vdotq_laneq_s32` (NEON), `svdot_lane_s32` (SVE2)
- RVV equivalent target: grouped `vwmaccsu.vx` or future `vdot`-style extension
- SpacemiT K1: this is the **RVV path** (vector registers, not IME)

### Why the layout changes
`vdotq_laneq_s32(Cr, Ar, Br, lane)` computes:
```
  Cr[i] += Ar[i*4+0]*Br[lane*4+0] + Ar[i*4+1]*Br[lane*4+1]
           + Ar[i*4+2]*Br[lane*4+2] + Ar[i*4+3]*Br[lane*4+3]
```
Both `Ar` and `Br` must have **4 consecutive depth elements grouped per element position**.
The `lane` index selects which group of 4 from `Br` to use for the dot product.

### Algorithm — B-side (INT8, NR, groups of 4 in depth, NR columns per group)

```
Input:  B[kc x nc] INT8, NR (must be multiple of 4 for lane addressing)
Output: Bc

for j = 0 to nc step NR:
    nr_actual = min(nc - j, NR)
    for p = 0 to kc step 4:            // depth in groups of 4
        depth4 = min(kc - p, 4)
        for jj = 0 to nr_actual:       // columns within panel
            for pp = 0 to depth4:      // 4 depth elements per column
                Bc[j*kc + p*NR + jj*4 + pp] = B[(p+pp)*ldB + (j+jj)]
```

**Flat address formula:**
```
Bc[ j * kc  +  p * NR  +  jj * 4  +  pp ]
   ^panel      ^depth      ^col        ^depth-element
```

### Algorithm — A-side (INT8, MR, groups of 4 in depth)

```
Input:  A[mc x kc] INT8, MR
Output: Ac

for i = 0 to mc step MR:
    mr_actual = min(mc - i, MR)
    for p = 0 to kc step 4:
        depth4 = min(kc - p, 4)
        for ii = 0 to mr_actual:       // rows within panel
            for pp = 0 to depth4:
                Ac[i*kc + p*MR + ii*4 + pp] = A[(i+ii)*ldA + (p+pp)]
```

### Output memory layout example (NR=8, kc=8)
```
Bc for panel j=0:
  depth group p=0: [B[0,0],B[1,0],B[2,0],B[3,0], B[0,1],B[1,1],B[2,1],B[3,1], ..., B[0,7]..B[3,7]]
  depth group p=4: [B[4,0],B[5,0],B[6,0],B[7,0], B[4,1]..B[7,1], ..., B[4,7]..B[7,7]]
```
Each group of 4 depth values for a column is stored **before** moving to the next column.

### Difference from Method 2
Methods 2 and 3 have the same loop structure. The difference is the **index formula**:
- Method 2 (`WIDEN`): `jj*4 + pp` groups depth-first inside each column group of a fixed-size panel chunk.
- Method 3 (`LANE`):  `jj*4 + pp` is identical but the flat offset uses `p * NR` not `(p/4) * NR * 4`.
In practice both produce the same byte layout for the standard case. The distinction lies
in the micro-kernel that consumes the data (3-instruction chain vs. single DOT instruction).

### xDSL builder functions to create
- `pack_CB_dot_int8_lane_NR{N}` — B-side packing
- `pack_RB_dot_int8_lane_MR{N}` — A-side packing
- Reuses `vlse8` for strided source; `vse8` for contiguous output.

---

## Method 4 — 2D Tile Block Packing (SpacemiT K1 IME / Intel AMX / ARM SME)

**Paper section:** §6.1 (SpacemiT K1), §6.2 (Intel AMX), §6.3 (ARM SME)
**Your status:** ❌ Not yet implemented

### Name
`TILE_INT8_Block_Pack` — 2D sub-tile packing for hardware matrix engines.
SpacemiT K1 variant: `TILE_INT8_K1_Pack` (8×8 INT8 blocks → INT32 via IME).

### When to use
- Source data type: **INT8**, accumulate to **INT32**
- Hardware: SpacemiT K1 IME, Intel AMX, ARM SME
- The hardware consumes a complete **2D tile** as a single operand, not a 1D vector

### Why the layout fundamentally changes
Matrix engines have **2D tile registers** (e.g., SpacemiT K1 IME uses 8×8 INT8 tiles).
One instruction feeds an entire `TILE_MR × TILE_NR` block. The data must be packed as
contiguous 2D blocks — **not** as 1D micro-panels.

### Algorithm — B-side (INT8, TILE_NR=8, TILE_MR=8 for SpacemiT K1)

```
Input:  B[kc x nc] INT8, TILE_NR=8, TILE_MR=8
Output: Bc — contiguous sequence of 8x8 INT8 tile blocks (64 bytes each)

tile_k = 0
for j = 0 to nc step TILE_NR:            // tile columns
    nr_tile = min(nc - j, TILE_NR)
    for p = 0 to kc step TILE_MR:        // tile rows (depth)
        mr_tile = min(kc - p, TILE_MR)

        // Fill one 8x8 tile block (row-major within tile)
        for pp = 0 to mr_tile:           // depth rows within tile
            for jj = 0 to nr_tile:       // columns within tile
                Bc[tile_k * TILE_MR * TILE_NR + pp * TILE_NR + jj]
                    = B[(p+pp)*ldB + (j+jj)]
            // zero-pad if nr_tile < TILE_NR
        // zero-pad rows if mr_tile < TILE_MR
        tile_k++
```

### Output memory layout (8×8 INT8 tiles, 64 bytes each)
```
Bc = [ tile(depth=0,col=0): B[0,0]..B[0,7], B[1,0]..B[1,7], ..., B[7,0]..B[7,7],  <- 64 bytes
       tile(depth=0,col=1): B[0,8]..B[0,15], ..., B[7,8]..B[7,15],                  <- 64 bytes
       tile(depth=1,col=0): B[8,0]..B[8,7],  ..., B[15,0]..B[15,7],                 <- 64 bytes
       ... ]
```
Each tile is a **row-major 8×8 block** stored contiguously.

### Intel AMX variant
Same structure but tile size is `16×64` bytes (INT8) → `TILE_MR=16`, `TILE_NR=64`.
The AMX `tdpbssd` instruction takes three tile registers directly.

### ARM SME variant
Same concept; tile dimensions are runtime-configurable via `svl` (streaming vector length).
The packing must query `svl` to set tile dimensions dynamically.

### xDSL builder function to create
`pack_CB_tile_int8_k1_TRNR{N}_TRMR{N}` — outermost two loops tile by `TILE_NR`/`TILE_MR`,
inner double loop `pp × jj` fills one 64-byte block. Uses `vle8` (unit-stride) for
row-major source; `vlse8` (strided) for column-major source.
Requires a new **IME dialect** (separate from RVV dialect) for the compute instruction.

---

## Summary Table

| # | Method Name | Intended Data Type | Hardware Instruction | How it groups the "Depth" (k-axis) | What it writes to memory *next* (The inner-most layout step) | Shape of the data chunk in memory |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **AXPY Flat** | **FP32** | Standard FMA<br>(e.g., `vfmacc`) | 1 row at a time | Reads **1 row** across all 8 columns.<br>*(col0, col1, col2 ... col7)* | **1D Horizontal Strip**<br>($1 \times 8$ elements) |
| **2** | **DOT Widen** | **INT8** | Widening Chain<br>(e.g., `vwmul` + `vwmacc`) | 4 rows at a time | Reads **4 vertical rows** from exactly 1 column, then moves to the next column.<br>*(col0_r0, col0_r1, col0_r2, col0_r3)* | **1D Vertical Strip**<br>($4 \times 1$ elements) |
| **3** | **DOT Lane** | **INT8** | Indexed Dot Product<br>(e.g., `vdot_lane`) | 4 rows at a time | Reads 1 row across all 8 columns, but does it 4 times before moving on to the next depth chunk.<br>*(Conceptually identical to Method 2 in byte layout, index math differs)* | **1D Vertical Strip**<br>($4 \times 1$ elements) |
| **4** | **Tile Block** | **INT8** | Matrix Engine<br>(e.g., SpacemiT IME) | $8$ rows at a time | Reads a full **$8 \times 8$ 2D block** row-by-row until all 64 elements of the tile are contiguous in memory. | **2D Square Tile**<br>($8 \times 8$ elements) |

---

## Implementation Order for xDSL Replication

```
Step 1 (done):
  Method 1 — AXPY_FP32_Pack
  Files: pack_builder.py (pack_CB_v_*, pack_RB_v_*)

Step 2 (next recommended — SpacemiT K1 RVV INT8 path):
  Method 3 — DOT_INT8_LANE_Pack
  Changes needed:
    - pack_builder.py: add depth loop step=4, inner jj x pp loops
    - rvv.py: add vlse8_v_i8m1Op, vse8_v_i8m1Op
    - New micro-kernel builder using INT8 x INT8 -> INT32 RVV chain

Step 3:
  Method 2 — DOT_INT8_WIDEN_Pack (if targeting ARMv8.0-A or RVV widening)
  Changes needed:
    - Same loop structure as Method 3 but different index formula
    - New micro-kernel: vwmul + vwmacc chain

Step 4 (advanced — SpacemiT K1 IME):
  Method 4 — TILE_INT8_Block_Pack
  Changes needed:
    - pack_builder.py: 2D tile outer loops, pp x jj inner block fill
    - New IME dialect (separate from RVV) for the matrix engine instruction
```

---

## Loop Structure Comparison (at a glance)

```
Method 1 (AXPY_FP32):
  for j(step=RR) -> for i(step=1) -> [vlse32 gathers RR fp32] -> vse32

Method 2 (DOT_INT8_WIDEN):
  for j(step=NR) -> for p(step=4) -> for jj(0..NR) -> for pp(0..4) -> store int8
                                       ^column first    ^depth second

Method 3 (DOT_INT8_LANE):
  for j(step=NR) -> for p(step=4) -> for jj(0..NR) -> for pp(0..4) -> store int8
                                       ^column first    ^depth second
  (identical loop order to Method 2; differs in flat index formula and consumer instruction)

Method 4 (TILE_Block):
  for j(step=TILE_NR) -> for p(step=TILE_MR) -> for pp(0..TILE_MR) -> for jj(0..TILE_NR) -> store int8
                                                  ^depth first          ^column second
  (Depth-first within tile; tiles traversed column-block-major)
```
