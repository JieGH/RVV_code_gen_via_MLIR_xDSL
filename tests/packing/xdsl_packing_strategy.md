# xDSL RVV Packing — Project Strategy Report
**Date:** 2026-07-07
**Target Hardware:** SpacemiT K1 (RISC-V, VLEN=256)

---

## 1. Where You Are Right Now

You have a working, validated **FP32 AXPY packing infrastructure** built on top of xDSL and MLIR EmitC. The key design points are:

- **pack_builder.py** generates MLIR IR using scf.ForOp loops and custom RVV dialect ops (vlse32, vle32, vse32).
- The generated code is parameterised: it handles **both Column-Major and Row-Major** source matrices by emitting two function variants (_if for strided, _else for unit-stride).
- The generated code is validated against a hand-written **C++ reference** (ref_pack_CB.cpp) using a digital twin memory comparison strategy.
- A **benchmark harness** (benchmark_packing.cpp) measures memory bandwidth to quantify performance.

This is a solid, production-quality foundation. The architecture is clean and the builder pattern is proven.

**The gap:** Everything is currently locked to FP32 and the AXPY micro-kernel pattern. The next phase of the paper's packing methods (INT8, DOT, IME Tile) requires extending both the data layout and the dialect.

---

## 2. Your Reference Code Landscape

You have two critical reference sources, both already on your machine:

### Source A: Your Current ref_pack_CB.cpp
Hand-written BLIS-style C++ for FP32. This is your **ground truth** for correctness testing. The pattern is: write a reference function, generate the xDSL version, compare the memory buffers.

### Source B: gemm_blis.c in .ext/transformerIME
File: .ext/transformerIME/transformers/GEMMLib/src/gemm/gemm_blis.c

This is the **gold standard C implementation** from the paper's authors. It contains every packing variant already implemented and dispatched via #define:

| Compile Flags | Packing Used | Maps to Our Method |
|:---|:---|:---|
| RISCV + K1 + Q_INT8_INT32 + IME_ENABLE | pack_vmadot_Ccol_A/B (line 661) | **Method 4** — IME Tile |
| A78AE + Q_INT8_INT32 | pack_sdot_A/B | **Method 3** — DOT Lane |
| Q_INT8_INT32 (ARM generic) | pack_2x8_int8_A/B (line 965) | **Method 2** — DOT Widen |
| NQ_FP16 | pack_6x16_fp16_A/B | FP16 AXPY variant |
| Default | pack_generic_A/B | **Method 1** — FP32 AXPY |

**This is your blueprint.** Every packing method you need is already available as a working C reference. You do not have to design the algorithms — you only have to translate them into xDSL IR builders.

---

## 3. Development Strategy — Three Phases

### Phase 1: INT8 Vector Packing (DOT Methods 2 and 3)
**Goal:** Extend the existing pack_builder.py to emit INT8 micro-panel packing suitable for RVV widening or DOT instructions.

**What stays the same:**
- The _if / _else branch architecture (same handling for Col-Major vs Row-Major).
- The scf.ForOp loop structure in the builders.
- The memory comparison testing strategy.

**What changes:**
1. **Add INT8 RVV ops to rvv.py:** vlse8_v_i8m1Op, vle8_v_i8m1Op, vse8_v_i8m1Op.
2. **Add a new _lmul_ops branch in pack_builder.py** that returns the INT8 ops for elem_bytes=1.
3. **Add an extra depth loop (step=KR=16 or 32)** around the inner loop, copying the structure from pack_2x8_int8_A/B (Method 2) or pack_sdot_A/B (Method 3).
4. **Implement Strict Zero-Padding for Memory Buffers:** DOT packing requires micro-blocks of fixed dimensions (e.g., `NR x KR` or `MR x KR`). The buffer allocation tests (e.g., `cb_packed_size`) must fully pad the `K` dimension (`nc` or `mc`) to a multiple of `KR`. The vector generator in `pack_builder.py` must offset its writes by skipping over this padding padding size (using `(nc + KR - 1)/KR * KR * MR`).
5. **Write a C++ reference for INT8 packing** in ref_pack_DOT_INT8_LANE_Pack.cpp to validate against. *Note on original C code:* The original `gemm_blis.c` implementation contained a bug where the final edge-block tail failed to pad missing columns for CB packing, while simultaneously over-padding beyond the unpadded allocated buffer size for RB packing (causing core dumps). The new reference completely rewrites the packing loops to guarantee exactly `NR x KR` or `MR x KR` elements are written to a perfectly padded buffer layout.

**Key reference:**
- `tests/packing/ref_pack_DOT_INT8_LANE_Pack.cpp` (Fixed ground truth)


---

### Phase 2: INT8 DOT Micro-kernel
**Goal:** Build the actual matrix-multiplication compute kernel that consumes the INT8 packed buffers produced in Phase 1.

**Why this is needed:** The INT8 packing is only useful if the micro-kernel can read and execute math on it. You must pair each packing method with a matching compute kernel.

**What needs building:**
1. **Add INT8 math ops to rvv.py:** vwmul_vv_i16m2Op (INT8->INT16 widening multiply), vwmacc_vv_i32m4Op (accumulate).
2. **Build a microkernel_builder.py** — a new file, mirroring the pattern of pack_builder.py but for the compute tile. It will generate the inner MR x NR accumulation loops.
3. **Write a C++ reference micro-kernel** and do the same memory-comparison validation as Phase 1.

---

### Phase 3: IME Tile Packing (Method 4 — SpacemiT K1)
**Goal:** Implement the 2D tile packing required for the SpacemiT K1 Matrix Engine (IME).

**Why this is a separate phase:** Architecturally different and more complex:
- You need a new **ime.py dialect** with 2D tile register types and IME-specific assembly operations.
- The packing loop structure changes completely from 1D micro-panels to 8x8 2D blocks.
- This is the most demanding phase but also the highest performance ceiling on K1.

**Key reference:**


---

## 4. The xDSL Builder Architecture — How Each Phase Fits In

Your pack_builder.py is built around a single, clean pattern:



Each new phase **extends this pattern, not replaces it**:

| Phase | New _lmul_ops case | New builder functions | New dialect file |
|:---|:---|:---|:---|
| Phase 1 (INT8 DOT) | elem_bytes=1 -> vle8, vlse8, vse8 | build_pack_CB_dot_int8_if/else | rvv.py (add INT8 ops) |
| Phase 2 (Micro-kernel) | N/A | build_microkernel_dot_int8 | rvv.py (add INT8 math ops) |
| Phase 3 (IME Tile) | N/A (tile ops, not vector) | build_pack_CB_tile_ime | New ime.py dialect |

---

## 5. Recommended First Steps (Immediate Next Actions)

Ordered from simplest to most impactful:

1. **Open rvv.py** — add vlse8_v_i8m1Op, vle8_v_i8m1Op, vse8_v_i8m1Op by copying and adapting the existing FP32 op definitions.
2. **Update _lmul_ops in pack_builder.py** — add an elem_bytes=1 branch that returns the new INT8 ops.
3. **Write build_pack_CB_dot_int8_if/else** — copy the FP32 builder, add the KR=16 depth step loop to match gemm_blis.c:965.
4. **Write the C++ reference** — port pack_2x8_int8_A/B from gemm_blis.c into ref_pack_CB.cpp.
5. **Run the memory comparison test** — confirm xDSL output == C++ reference.

---

## 6. Summary

| What you have | What you are building | End goal |
|:---|:---|:---|
| FP32 AXPY packing (Method 1) | INT8 DOT packing (Methods 2 and 3) | INT8 DOT packing (K1 RVV) |
| pack_builder.py (2-loop structure) | Extended pack_builder.py (3-loop with KR step) | Full INT8 packing suite |
| ref_pack_CB.cpp (FP32 reference) | New INT8 reference functions | All methods validated |
| rvv.py (FP32 ops only) | rvv.py extended with INT8 memory and math ops | Full INT8 RVV op coverage |
| No micro-kernel | microkernel_builder.py | End-to-end INT8 GEMM |
| No IME dialect | New ime.py dialect | SpacemiT K1 IME tile GEMM |

The cleanest path is: **extend what exists, do not redesign it.** Your architecture is already correct for the job — you are only adding new data types and deeper loop nesting, not changing the fundamental xDSL IR-builder approach.
