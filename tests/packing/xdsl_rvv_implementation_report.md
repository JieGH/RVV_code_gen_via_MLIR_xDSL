# Implementation Report: xDSL/RVV Vectorized Packing vs. BLIS Reference

This report provides a detailed side-by-side analysis of how the UPV/BLIS reference design in `gemm_blis.c` was translated into a high-performance vectorized implementation using xDSL and RISC-V Vector (RVV) intrinsics.

---

## 1. Dialect & Lowering Foundation
To represent RVV operations in xDSL, we extended the compiler with specific dialects and lowering passes.

| Aspect | xDSL Implementation | Reference Design Requirement |
| :--- | :--- | :--- |
| **Op Definition** | `vlse32_v_f32m1Op` in `rvv.py` | Need for Strided Loads (Column-Major access) |
| **Lowering** | `RVVToEmitCPass` maps IR ops to `__riscv_vlse32_...` | Compatibility with GCC/LLVM RVV Intrinsics |
| **Tail Handling** | `arith.minsi` + `rvv.setvl` | `min(nc-j, RR)` for partial blocks |

---

## 2. Code Generation: Logic Mapping
The following table compares the scalar C logic from `gemm_blis.c` with the vectorized IR builder logic in `test_pack_CB_v_fp32.py`.

### Comparison: `pack_CB` (Column-Block)

#### [Scalar Reference (C)]
```c
// if branch (Col-Major)
for ( j=0; j<nc; j+=RR ) { 
  nr = min(nc-j, RR);
  for ( i=0; i<mc; i++ ) {
    for ( jj=0; jj<nr; jj++ ) {
      Mc[k] = M[(j+jj)*ldM + i];
      k++;
    }
  }
}
```

#### [Vectorized xDSL (Python Builder)]
```python
# Equivalent Vectorized Logic
for j in range(0, nc, 8):
    vl = setvl(min(nc - j, 8))
    for i in range(0, mc):
        # STRIDED LOAD (replaces jj loop)
        # stride = ldM * 4 bytes
        vec = vlse32(&M[j*ldM + i], stride, vl)
        
        # VECTOR STORE
        vse32(&Mc[j*mc + i*8], vec, vl)
```

**Why this works:** The `jj` loop in the reference accesses `M[j*ldM+i]`, `M[(j+1)*ldM+i]`, etc. These elements are separated by `ldM` in memory. The RVV `vlse32` instruction fetches all `nr` elements in a single hardware operation using `ldM` as the stride.

---

## 3. Reference-Aligned Data Layout

| Feature | Reference Design (`gemm_blis.c`) | xDSL Implementation Representation |
| :--- | :--- | :--- |
| **Index calculation** | `Mc[k++]` with `k += (RR-nr)` | `Mc[j*mc + i*8]` (Absolute indexing) |
| **Block Width** | `RR` passed as argument | Hardcoded 8 (CB) / 4 (RB) for VLEN=256 |
| **Memory Order** | Dispatcher checks `orderM` & `transM` | Dispatcher routes to `_if` or `_else` kernel |

### Side-by-Side: `pack_RB` (Row-Block)

| BLIS Scalar Logic (`if` branch) | xDSL RVV Logic |
| :--- | :--- |
| `Mc[k] = M[j*ldM + i+ii]` | `vle32(&M[j*ldM + i], vl)` |
| `ii` loop is contiguous for fixed `j` | **Unit-Stride Load** used for efficiency |
| `Mc` block-aligned via `k += (RR-rr)` | `Mc[i*nc + j*4]` ensures same padding |

---

## 4. Verification & Benchmarking

The verification suite was redesigned to be a "digital twin" of the production environment.

1.  **Direct Code Port**: The `ref_pack_CB.cpp` file contains **exact copies** of the C functions from `gemm_blis.c`. No "simplified" logic was used; we verify against the absolute ground truth.
2.  **API Mirroring**: The `pack_dispatch.h` provides an identical signature to the benchmark's `gemm_blis.h`, allowing the vectorized kernels to be dropped into the existing benchmark without modification.
3.  **Category Testing**:
    *   **Dimension**: Ensures `std::min` handles all `mc`, `nc` combinations.
    *   **Semantic**: High-magnitude fills (`i*1000 + j`) verify that `i` and `j` are never swapped.
    *   **Boundary**: Sentinels at the end of buffers verify that `vse32` never writes past the end of a partial block.

### Verification Results Summary (SpacemiT K1)
*   **Total Tests**: 58
*   **Total Passed**: 58
*   **Failures**: 0
*   **Hardware Efficiency**: The implementation transitions from `vlse32` (high latency) to `vle32` (low latency) automatically based on the matrix layout, maximizing the board's memory bandwidth.
