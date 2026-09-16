# xDSL BLIS Implementation and Packing Report

> All claims in this document are cross-referenced against the actual source files in
> `tests/pure_xdsl_blis_rvv/` and the generated output in `pj_gen/fp32_AXPY_FP32_Pack_vlen256/`.

---

## Overview

The xDSL BLIS framework replaces a hand-maintained benchmarking workflow with a **fully automated project generator pipeline**. Given a configuration file, it generates an entire, self-contained C++ project containing all microkernel variants, packing functions, a benchmark driver, and a build system — then compiles and runs them on the RISC-V board (`jbpi2`) remotely.

The core insight is: instead of generating a single kernel and injecting it into a static project, the framework generates a **complete family** of all microkernel dimension combinations, pre-computes optimal cache blocking parameters per variant, and produces one binary that benchmarks all variants automatically.

---

## 1. Configuration

### `BLISProjectConfig` Dataclass
Defined in [`config.py`](file:///home/jlei/Documents/ws_MLIR/repo_xdsl_microkernels/RVV_code_gen_via_MLIR_xDSL/tests/pure_xdsl_blis_rvv/config.py):

```python
@dataclass
class BLISProjectConfig:
    datatype: str        # "fp32" | "int8"
    method: str          # "AXPY_FP32_Pack" | "DOT_INT8_LANE" | ...
    vlen: int            # 256 | 512 (bits)
    arch: str            # "jbpi2" | "bananapi" | "amd_zen" | "arm_carmel"
    mr_nr_family: list   # e.g. [[32, 16]] — defines the MAX bounds
    datasets: list[str]  # e.g. ["project_sample.dat"]
```

### Active `config.json`
```json
{
    "datatype": "fp32",
    "method": "AXPY_FP32_Pack",
    "vlen": 256,
    "arch": "jbpi2",
    "mr_nr_family": [[32, 16]],
    "datasets": ["project_sample.dat"]
}
```

### VLEN Bit-to-Element Conversion
`vlen` is specified in **bits**. The generator immediately converts it to an element count before passing it to any builder:

```python
vlen_elements = config.vlen // 32   # FP32: 256 // 32 = 8 elements per vector register
```

All xDSL pack/microkernel builders receive `vlen_elements` (8), not raw bits. This controls:
- How many RVV operations are emitted per loop body (`num_vecs = ceil(dim / vlen_elements)`)
- The AVL (Active Vector Length) computation for safe tail handling

---

## 2. Kernel Family Expansion

The `mr_nr_family` field does **not** define a literal list of kernels. It defines **maximum bounds**. The generator expands these into a **complete 2D grid** of all combinations:

```python
# From generate_pure_blis_project.py, lines 213-221
for mr, nr in config.mr_nr_family:      # e.g. [(32, 16)]
    for i in range(start_m, mr + 1):    # 2 .. 32
        for j in range(start_n, nr + 1): # 2 .. 16
            expanded_families.add((i, j))
```

For `config.mr_nr_family = [[32, 16]]`, this expands to **512 unique microkernel dimension pairs** (MR∈[2..32], NR∈[2..16]).

**Why a complete 2D grid?** The BLIS algorithm must handle matrix boundaries where remaining tile dimensions are smaller than the maximum MR or NR (e.g., if M=33 and MR=32, the last tile has cur_mr=1). Having every integer combination from (1,1) to (32,16) pre-compiled means the runtime dispatch can select the exact-fit kernel for any boundary tile with no branching overhead.

---

## 3. Generated Project Structure

The output directory is named `pj_gen/{datatype}_{method}_vlen{vlen}/`. For the current config, this is:
**`pj_gen/fp32_AXPY_FP32_Pack_vlen256/`**

| File | How Produced | Contents |
|:---|:---|:---|
| `gemm_b0_xdsl.cpp` | xDSL IR → `mlir-translate` | All 512 Beta=0 microkernel functions (~4.4 MB) |
| `gemm_b1_xdsl.cpp` | xDSL IR → `mlir-translate` | All 512 Beta=1 microkernel functions (~5.0 MB) |
| `gemm_pack_xdsl.cpp` | xDSL IR → `mlir-translate` | All packing functions for dim=1..32 (~200 KB) |
| `gemm_dispatch_xdsl.cpp` | Python string generation | BLIS 5-loop + runtime dispatch if/else table |
| `gemm_headers.h` | Python string generation | All forward declarations + helper macros |
| `generated_kernels.h` | Python generation | Driver-facing: `KernelVariant[]` table + typedefs |
| `driver_benchmark.cpp` | Copied from `templates/` | C++ benchmark harness |
| `Makefile` | Templated from `templates/` | Build rules, dataset filtering |

> [!IMPORTANT]
> **Files are NOT split per MR/NR pair.** All B0 microkernels are merged into a single `gemm_b0_xdsl.cpp`; all B1 into `gemm_b1_xdsl.cpp`; all packing into `gemm_pack_xdsl.cpp`. Each file can exceed 4 MB due to the large number of generated functions.

### xDSL MLIR Lowering Pass Pipeline
The three xDSL-generated files are produced by the following pass pipeline (from `generate_pure_blis_project.py`):

```python
pass_pipeline = [
    GemmToArithPass(),         # Lower GEMM dialect → arith/SCF ops
    MemRefToEmitCPass(),       # Lower memref types → C pointer semantics
    ArithToEmitCPass(),        # Lower arith ops → EmitC (C expressions)
    MemrefLoadToEmitcPass(),   # Lower memref.load → pointer deref
    MemrefStoreToEmitcPass(),  # Lower memref.store → pointer assignment
    SCFToEmitCPass(),          # Lower SCF for-loops → C for-loops
    RVVToEmitCPass(),          # Lower RVV dialect → riscv_vector.h intrinsics
]
```

After passes are applied, the resulting MLIR is dumped to a `.mlir` file and then lowered to C via `mlir-translate -mlir-to-cpp`. The generator wraps the output in `extern "C"` and prepends `#include "gemm_headers.h"`.

---

## 4. The Dispatch Layer (`gemm_dispatch_xdsl.cpp`)

This file is **entirely Python-string-generated**, not produced by xDSL IR lowering. It contains two functions:

### `gemm_microkernel_v_dispatch_generic`
A runtime dispatch table. Takes `size_t` arguments and selects the exact pre-compiled MR×NR kernel using a flat if/else chain:

```cpp
void gemm_microkernel_v_dispatch_generic(
        size_t cur_mr, size_t cur_nr, size_t kc,
        float alpha, float* Ac, float* Bc,
        float beta, float* C, size_t ldc,
        size_t max_mr, size_t max_nr) {
    if (beta == (float)0) {
        if (cur_mr == 1 && cur_nr == 1) { gemm_microkernel_v_fp32_1x1_b0(...); return; }
        if (cur_mr == 1 && cur_nr == 2) { gemm_microkernel_v_fp32_1x2_b0(...); return; }
        // ... continues for all 512 combinations ...
        if (cur_mr == 32 && cur_nr == 16) { gemm_microkernel_v_fp32_32x16_b0(...); return; }
    } else {
        // Same structure for _b1 variants
    }
}
```

Microkernel naming convention: `gemm_microkernel_v_{datatype}_{MR}x{NR}_b{0|1}`

### `gemm_blis_generic`
The BLIS 5-loop blocking structure. Uses `int32_t` parameters (matching the driver calling convention):

```cpp
void gemm_blis_generic(int32_t m, int32_t n, int32_t k,
        float alpha, float* A, int32_t lda, float* B, int32_t ldb,
        float beta, float* C, int32_t ldc,
        float* Ac, float* Bc, int32_t mc, int32_t nc, int32_t kc,
        int32_t mr, int32_t nr) {

    // Static function pointer tables, indexed by dimension
    static const pack_CB_fn pack_CB_funcs[] = { nullptr,
        pack_CB_v_if_AXPY_FP32_Pack_fp32_NR1,
        pack_CB_v_if_AXPY_FP32_Pack_fp32_NR2,
        // ...up to NR32
    };
    static const pack_RB_fn pack_RB_funcs[] = { nullptr,
        pack_RB_v_if_AXPY_FP32_Pack_fp32_MR1,
        // ...up to MR32
    };

    for (int32_t jc = 0; jc < n; jc += nc) {
        for (int32_t pc = 0; pc < k; pc += kc) {
            float beta_actual = GET_BETA(pc, beta); // = 0 only on first K panel
            pack_CB_funcs[nr](b_ptr, Bc, cur_kc, cur_nc, ldb);  // Pack B

            for (int32_t ic = 0; ic < m; ic += mc) {
                pack_RB_funcs[mr](a_ptr, Ac, cur_mc, cur_kc, lda);  // Pack A

                for (int32_t jr = 0; jr < cur_nc; jr += nr) {
                    for (int32_t ir = 0; ir < cur_mc; ir += mr) {
                        // cur_mr/cur_nr are boundary-clamped via std::min
                        gemm_microkernel_v_dispatch_generic(
                            cur_mr, cur_nr, cur_kc, alpha,
                            a_packed, b_packed, beta_actual, c_ptr, ldc, mr, nr);
                    }
                }
            }
        }
    }
}
```

`GET_BETA(pc, beta)` is a macro that returns `beta` only on the first K-panel pass (`pc==0`) and `1.0` on all subsequent passes, preventing double-application of the initial beta scaling.

---

## 5. Cache Blocking: `cache_config.py`

The `model_level()` function (ported from the reference `model_level.c`) calculates optimal cache panel sizes.

### Supported Architectures

| Architecture | L1 (NL×CL×WL) | L2 (NL×CL×WL) | L3 (NL×CL×WL) |
|:---|:---|:---|:---|
| `jbpi2` | 128 × 64B × 4-way | 1024 × 64B × 8-way | 4096 × 64B × 16-way |
| `bananapi` | 128 × 64B × 4-way | 1024 × 64B × 8-way | 4096 × 64B × 16-way |
| `amd_zen` | 64 × 64B × 8-way | 1024 × 64B × 8-way | 32768 × 64B × 16-way |
| `arm_carmel` | 256 × 64B × 4-way | 4096 × 64B × 8-way | 32768 × 64B × 16-way |

### `get_optim_mc_nc_kc()` Calculation Flow

```python
def get_optim_mc_nc_kc(arch, dataSize, mr, nr, ...) -> (mc, nc, kc):
    # Step 1: KC — fit A-panel (MR×KC) + B-panel (KC×NR) into L1
    kc = floor(model_level(isL3=False, NL1, CL1, WL1, dataSize, m=mr, n=nr))

    # Step 2: MC — fit A macro-panel (MC×KC) into L2
    mc = floor(model_level(isL3=False, NL2, CL2, WL2, dataSize, m=kc, n=nr))

    # Step 3: NC — fit B macro-panel (KC×NC) into L3
    nc = floor(model_level(isL3=True, NL3, CL3, WL3, dataSize, m=kc, n=mc))
```

**Cache way partitioning** prevents thrashing by dividing available ways proportionally between A and B matrices. For a given level with `WL` associativity ways, one way is reserved for eviction:

```python
CAr = math.floor((WL - 1.0) / (1.0 + n / m))   # Ways allocated to matrix A
CBr = ceil((n / m) * CAr)                          # Ways allocated to matrix B
```

In the benchmark driver (`driver_benchmark.cpp` line 332), MC/NC/KC are computed at runtime per-variant:
```cpp
int mc, nc, kc;
get_optim_mc_nc_kc(sizeof(float), m, n, k, variant.mr, variant.nr, &mc, &nc, &kc);
```

This means each MR/NR variant gets its own **hardware-optimal** cache blocking parameters at benchmark time.

---

## 6. Vectorized Packing via xDSL (RVV)

Packing functions are defined in MLIR IR via `pack_builder.py` and lowered to `gemm_pack_xdsl.cpp`.

### Packing Function Naming Convention
All packing functions follow this exact naming scheme (verified from `generated_kernels.h`):

```
pack_CB_v_if_{method}_{datatype}_NR{dim}     # Pack B: vectorized fast path
pack_CB_v_else_{method}_{datatype}_NR{dim}   # Pack B: scalar/tail path
pack_RB_v_if_{method}_{datatype}_MR{dim}     # Pack A: vectorized fast path
pack_RB_v_else_{method}_{datatype}_MR{dim}   # Pack A: scalar/tail path
```

Example names for the current config:
- `pack_CB_v_if_AXPY_FP32_Pack_fp32_NR16`
- `pack_RB_v_if_AXPY_FP32_Pack_fp32_MR32`

The `gemm_blis_generic` function always calls the `_v_if` (vectorized) variant via the static function pointer table.

### Packing Matrix A — `pack_RB_v_if` (Contiguous Loads)
Matrix A panels are $MR \times KC$, stored **Column-Major**. Elements within a column are contiguous in memory, enabling contiguous vector loads:

```python
# From gemm_blis_xdsl/pack_builder.py — Pack A (MR rows, KC columns)
c_chunk, vl_op = get_safe_chunk_avl(rr.result, v, vlen, idx, ib)

# Contiguous Vector Load from Matrix A (elements are contiguous in col-major A)
load_op = vle32_v_f32m1Op(M, src_off_chunk.result, vl_op.vl)

# Contiguous Vector Store into Packed Ac
store = vse32_v_f32m1Op(Mc, dst_off_chunk.result, load_op.results[0], vl_op.vl)
```

### Packing Matrix B — `pack_CB_v_if` (Strided Loads)
Matrix B panels are $KC \times NR$, stored **Row-Major**. Matrix B is in Column-Major format, so elements of a *row* are separated by stride `ldB`. xDSL uses strided vector loads:

```python
# From gemm_blis_xdsl/pack_builder.py — Pack B (KC rows, NR cols)
# Stride = leading dimension × bytes per element (ldM × 4 for FP32)
bstr = arith.MuliOp(ldM, cEB.result)

# Strided Vector Load (vlse32.v) — gathers one row of B from Column-Major layout
load_op = vlse32_v_f32m1Op(M, src_off_chunk.result, bstr.result, vl_op.vl)

# Contiguous Vector Store into packed Bc (Row-Major output)
store = vse32_v_f32m1Op(Mc, dst_off_chunk.result, load_op.result, vl_op.vl)
```

### Handling Dimensions Larger than VLEN

When a block dimension (MR or NR) exceeds `vlen_elements` (e.g., MR=32 > 8), a single vector instruction cannot cover the full row/column. The generator **unrolls at code generation time**:

```python
num_vecs = (RR + vlen - 1) // vlen   # ceil(32 / 8) = 4 vector ops needed
for v in range(num_vecs):
    # Each iteration emits a load/store at offset (v * vlen_elements)
    c_chunk, vl_op = get_safe_chunk_avl(rr.result, v, vlen, idx, ib)
    # ... emit vle32/vse32 with correct pointer offsets
```

`get_safe_chunk_avl` dynamically computes the AVL to protect against out-of-bounds access on the last (potentially partial) chunk. For MR=32 with 8 FP32 elements/vector, the generated packing loop body contains exactly **4 consecutive vector load/store pairs** with no runtime branching.

---

## 7. The C++ Benchmark Driver (`driver_benchmark.cpp`)

Copied from `templates/driver_benchmark.cpp` into each generated project. Key responsibilities:

### Memory Layout
All matrices use **64-byte aligned allocation** to prevent RVV hardware bus errors on `jbpi2`:
```cpp
float* alloc_aligned(size_t size_floats) {
    size_t alloc_bytes = (bytes + 63) & ~63;   // round up to 64-byte boundary
    void* ptr = aligned_alloc(64, alloc_bytes);
    return static_cast<float*>(ptr);
}
```

Pack buffers also use **canary values** (−123456.0f) in a 16-element guard zone before and after, to detect memory corruption at runtime:
```cpp
for (size_t i = 0; i < 16; ++i) {
    Ac_alloc[i] = -123456.0f;                    // front canary
    Ac_alloc[16 + ac_len + i] = -123456.0f;      // back canary
}
// ... after benchmark run ...
if (Ac_alloc[i] != -123456.0f) { /* abort */ }
```

### Dataset Format
Datasets are `.dat` files with space-separated rows. Each row defines one benchmark problem:
```
# Label   M     N     K
BERT_L_M5 384   384   64
BERT_L_F11 4096 384   1024
```
Comments (`#`) and blank lines are skipped. `lda=M`, `ldb=K`, `ldc=M` (Column-Major layout).

### Benchmark Loop (per variant per problem)
1. **Compute MC/NC/KC**: Call `get_optim_mc_nc_kc()` for this variant's `mr`/`nr`.
2. **Warmup pass**: Run the kernel once to warm caches before timing.
3. **Correctness check**: Compare output vs. `gemm_ref()` (scalar reference GEMM). Tolerance = `max(1e-5, k × 1e-6)`. Variants that fail are marked `FAIL` and excluded from timing.
4. **Timing loop**: Run `N_REPS = 5` iterations, measure `chrono::high_resolution_clock`. Compute: `GFLOPS = 2×M×N×K / (avg_time × 1e9)`.
5. **Canary check**: Verify guard zones around Ac/Bc were not corrupted.

### Output Files
Results are written to two log files inside `logs/`:

**`logs/raw_results.txt`** (one row per variant per problem):
```
Label,M,N,K,KernelName,Status,GFLOPS,RVV_Time_ms,Ref_Time_ms
BERT_L_M5,384,384,64,MR16_NR15,PASS,12.26,1.54,22.8
```

**`logs/best_results.txt`** (one row per problem, best variant only):
```
Arch,Datatype,Method,VLEN,Label,M,N,K,Best_Kernel,GFLOPS,RVV_Time_ms,Ref_Time_ms
jbpi2,fp32,AXPY_FP32_Pack,256,BERT_L_M5,384,384,64,MR16_NR15,12.26,1.54,22.8
```

The terminal also shows a live Top-3 ranking for each problem as benchmarks run.

---

## 8. Design Decisions & Runtime Robustness

### VLEN Validation
The `Makefile` includes a `check_vlen` target that reads the board's VLEN at runtime:
```makefile
check_vlen:
    @vlen=$$(cat /sys/devices/virtual/riscv_isa ... | grep -o 'v256\|v512\|v128'); \
    echo "Detected VLEN: $$vlen"
```
> [!WARNING]
> Currently `check_vlen` **only prints** the VLEN — it does not assert or abort on mismatch. Assertion logic is marked as a future enhancement. Deploying a binary compiled for VLEN=256 to a VLEN=512 board will produce silent correctness errors.

### Datatype Handling at Generation Time
The driver is re-generated from a template for each config. Types (`float` for fp32, `int8_t`/`int32_t` for int8) are **baked in at Python generation time** — there are no `#ifdef DTYPE_FP32` compile-time guards in the compiled C++.

### C++ Signature Bridge
`gemm_blis_generic` uses `int32_t` parameters (BLIS calling convention used by the driver), while individual xDSL microkernels use `size_t` pointers. The Python-generated `gemm_microkernel_v_dispatch_generic` acts as the bridge — it accepts `size_t` and passes addresses of stack-allocated alpha/beta values directly to the microkernels.

---

## 9. Head-to-Head Comparison: Reference BLIS vs. xDSL BLIS

| Feature | Reference BLIS (`gemm_blis.c`) | xDSL BLIS (`gemm_blis_xdsl`) |
| :--- | :--- | :--- |
| **Generation Strategy** | Hand-written C macros and static code. | Dynamically generated C++ via Python + xDSL IR (MLIR). |
| **Microkernel Generation** | Hand-coded assembly or intrinsic functions. | Automatically generated via xDSL → EmitC → `mlir-translate`. |
| **Packing Implementation** | Nested scalar C loops (element-by-element). | **Vectorized (RVV)**: `vle32.v` / `vse32.v` (contiguous); `vlse32.v` (strided for B). |
| **Handling Large Blocks** | Scalar `for` loop bounds handle any size natively. | **Compile-time unrolling**: `num_vecs = ceil(dim/vlen_elements)` sequential vector ops, no runtime branching. |
| **Boundary Handling** | Scalar loop naturally handles non-divisible dimensions. | **Global kernel pool**: all (1,1)..(MR,NR) pairs pre-compiled; runtime dispatch selects exact-fit kernel. |
| **Cache Blocking** | Static constants or runtime computation. | Python `cache_config.py` models L1/L2/L3 per-arch; MC/NC/KC computed per variant at benchmark time. |
| **Correctness Verification** | Manual inspection or external scripts. | Built-in per-variant correctness check against `gemm_ref()` with relative error threshold `max(1e-5, k×1e-6)`. |
| **Maintainability** | Manual porting for new VLENs or kernel sizes. | Change `vlen`/`mr_nr_family` in `config.json` and regenerate. |

---

## 10. Benchmark Results

> [!NOTE]
> **Configuration**
> - **Architecture**: `jbpi2`
> - **Datatype**: `fp32`
> - **Method**: `AXPY_FP32_Pack`
> - **Vector Length (VLEN)**: 256 bits (8 FP32 elements/vector register)
> - **Metric**: `Ker GOPs` = kernel-only compute throughput (excludes packing time); `Tot GOPs` = end-to-end including packing

| Dataset / Workload | Dimensions (M×N×K) | Best Kernel | Tot GOPs | Ker GOPs | RVV Pk (GB/s) | Scl Pk (GB/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **L1 (Edge)** | 1×1×1 | `MR7_NR5` | 0.00 | 0.02 | 0.0 | 0.0 |
| **L2 (Edge)** | 3×5×7 | `MR4_NR6` | 0.21 | 1.20 | 0.7 | 0.6 |
| **L3 (MicroTile)** | 8×8×8 | `MR16_NR8` | 0.88 | 4.10 | 1.4 | 0.9 |
| **L4 (Pack_Tall)** | 256×8×32 | `MR24_NR8` | 4.52 | 9.57 | 2.9 | 1.4 |
| **L5 (Pack_Wide)** | 8×256×32 | `MR11_NR16` | 3.85 | 8.15 | 2.5 | 1.5 |
| **L6 (KC_Cross)** | 64×64×300 | `MR16_NR13` | 8.48 | 12.61 | 2.0 | 1.3 |
| **L7 (Cache_Fit)** | 128×128×128 | `MR16_NR15` | 11.09 | 13.72 | 2.4 | 1.3 |
| **s_1024 (Square)** | 1024×1024×1024 | `MR16_NR7` | 7.78 | 9.01 | 0.3 | 0.2 |
| **B1_M123** | 1024×384×1024 | `MR32_NR7` | 7.05 | 8.55 | 0.3 | 0.3 |
| **B2_M5** | 384×384×64 | `MR16_NR15` | 11.58 | 12.96 | 2.2 | 1.3 |
| **B3_M7** | 64×384×384 | `MR16_NR11` | 7.13 | 10.86 | 0.9 | 0.3 |
| **B4_F11** | 4096×384×1024 | `MR16_NR7` | 7.31 | 8.90 | 0.3 | 0.2 |
| **B5_F13** | 1024×384×4096 | `MR32_NR7` | 6.70 | 8.28 | 0.3 | 0.3 |

### Comparison: Current xDSL AXPY vs Previous xDSL Design (`prev_xdsl`)

Based on the BERT-Large workload evaluations on the Banana Pi (`jbpi2`), we compare the current automated AXPY generation strategy (Ker GOPs) against the previous xDSL FP32 implementation (`prev_xdsl`):

| Workload | Dimensions (M×N×K) | `prev_xdsl` (GFLOPS) | `prev_xdsl` Kernel | Current Ker GOPs | Current Best Kernel | Comparison |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **S1 (Square)** | 1024×1024×1024 | ~8.5 | 32x7 | **9.01** | `MR16_NR7` | +6.0% |
| **B1 (B1_M123)** | 1024×384×1024 | ~8.0 | 32x6 | **8.55** | `MR32_NR7` | +6.9% |
| **B2 (B2_M5)** | 384×384×64 | ~12.5 | 16x15 | **12.96** | `MR16_NR15` | +3.7% |
| **B3 (B3_M7)** | 64×384×384 | ~8.8 | 16x15 | **10.86** | `MR16_NR11` | +23.4% |
| **B4 (B4_F11)** | 4096×384×1024 | ~8.5 | 32x7 | **8.90** | `MR16_NR7` | +4.7% |
| **B5 (B5_F13)** | 1024×384×4096 | ~7.5 | 32x6 | **8.28** | `MR32_NR7` | +10.4% |

**Key Findings:**
1. **Across-the-board improvement:** The new run shows Ker GOPs exceeding `prev_xdsl` on every single BERT-Large workload, with gains ranging from +3.7% (B2) to +23.4% (B3).
2. **B3 regression resolved:** Previously the generator selected `MR8_NR14` for B3 (64×384×384) yielding only 6.58 GOPs, well below `prev_xdsl`'s ~8.8 GOPs. The new run correctly selects `MR16_NR11` and reaches 10.86 GOPs — a 65% improvement over the prior run and +23% over `prev_xdsl`.
3. **Packing overhead visible:** For large memory-bandwidth-bound shapes (s_1024, B1, B4, B5), `RVV Pk` bandwidth drops to ~0.3 GB/s, meaning packing dominates total runtime and creates a gap between `Tot GOPs` and `Ker GOPs`. For cache-resident shapes (L4–L7), packing BW reaches 2–3 GB/s and the gap narrows.


---


---

## 11. Summary

The xDSL BLIS implementation delivers on five key dimensions:

1. **Orchestration** — Python-string-generated C++ (`gemm_dispatch_xdsl.cpp`) implements the BLIS 5-loop blocking structure with hardware-optimal MC/NC/KC per variant computed by `cache_config.py`.
2. **Microkernels** — xDSL IR → EmitC → `mlir-translate` yields `gemm_b0_xdsl.cpp` and `gemm_b1_xdsl.cpp` containing all 512 dimension variants. Runtime dispatch selects exact-fit kernels, including boundary tiles.
3. **Packing** — xDSL IR → `gemm_pack_xdsl.cpp` with RVV vectorization: contiguous `vle32.v`/`vse32.v` for Matrix A (col-major source), strided `vlse32.v` for Matrix B (reformat col-major → row-major).
4. **Large dimensions** — Compile-time unrolling with `num_vecs = ceil(dim/vlen_elements)` sequential vector ops per loop body; safe tail handling via `get_safe_chunk_avl`.
5. **Verification** — Built-in correctness check (relative error vs. scalar `gemm_ref`), canary memory guards on pack buffers, and dual log files for raw per-variant and best-per-problem results.

---

## 12. Execution Commands

### Full Generation and Remote Execution Pipeline

Run from the **project root** (`RVV_code_gen_via_MLIR_xDSL/`):

```bash
source .venv/bin/activate && \
  REMOTE_PASS="" REMOTE_HOST="fej" REMOTE_PORT="3322" \
  ./tests/pure_xdsl_blis_rvv/compile_macro_bench.sh \
  --jump --riscv-user jlei --riscv-ip jbpi2 --riscv-port 22
```

**What this does (in order):**
1. Activates the Python virtual environment
2. Runs `generate_pure_blis_project.py --config config.json` → generates `pj_gen/`
3. `rsync` the entire `pj_gen/`, `datasets/`, and `modelLevel/` directories to `jbpi2`
4. SSH: `make clean && make all` — compiles on the board with `-march=rv64gcv -O3`
5. SSH: `make run` — executes against configured datasets, writes CSV logs
6. `rsync` logs back to `tests/pure_xdsl_blis_rvv/logs/{config}_{timestamp}/`

### Rerun Without Recompiling

To re-execute an already-compiled binary on the board (skips generation and recompile):

**Via jump host:**
```bash
sshpass -p "" ssh -p 3322 -t jlei@fej \
  "ssh -p 22 -t jbpi2 \
    'cd xdsl_rvv_microkernel/tests/pure_xdsl_blis_rvv/pj_gen/fp32_AXPY_FP32_Pack_vlen256 \
     && make run'"
```

**Directly on `jbpi2`:**
```bash
cd ~/xdsl_rvv_microkernel/tests/pure_xdsl_blis_rvv/pj_gen/fp32_AXPY_FP32_Pack_vlen256
make run
# Or run with a specific dataset:
./test_gemm.x ../../datasets/project_sample.dat
```

---

## 13. Architecture Diagram

```mermaid
graph TD
    classDef generator fill:#1f77b4,stroke:#fff,stroke-width:2px,color:#fff;
    classDef output fill:#2ca02c,stroke:#fff,stroke-width:2px,color:#fff;
    classDef pyout fill:#17becf,stroke:#fff,stroke-width:2px,color:#fff;
    classDef orchestrator fill:#ff7f0e,stroke:#fff,stroke-width:2px,color:#fff;
    classDef target fill:#d62728,stroke:#fff,stroke-width:2px,color:#fff;
    classDef plot fill:#9467bd,stroke:#fff,stroke-width:2px,color:#fff;
    classDef config fill:#8c564b,stroke:#fff,stroke-width:2px,color:#fff;

    subgraph "1. Host Machine - Project Generation"
        CFG["config.json"]:::config
        M1["generate_pure_blis_project.py"]:::generator
        CFG --> M1

        XDSL["xDSL Framework and mlir-translate"]:::generator
        M1 -->|Expand 2D family and build IR| XDSL

        subgraph "pj_gen/ (and compile_order.txt)"
            ORDER["compile_order.txt"]:::pyout
            subgraph "{datatype}_{method}_vlen{vlen}/"
                C1["gemm_b0_xdsl_{method}.cpp"]:::output
                C2["gemm_b1_xdsl_{method}.cpp"]:::output
                C3["gemm_pack_xdsl_{method}.cpp"]:::output
                C4["gemm_dispatch_xdsl_{method}.cpp"]:::pyout
                HDR["gemm_headers_{method}.h"]:::pyout
                DRV["driver_benchmark_{method}.cpp"]:::pyout
                MK["Makefile"]:::pyout
            end
        end
        M1 -->|Writes execution sequence| ORDER
        XDSL -->|mlir-translate| C1
        XDSL -->|mlir-translate| C2
        XDSL -->|mlir-translate| C3
        M1 -->|Python string gen| C4
        M1 -->|Python gen and template copy| HDR
        M1 -->|Python gen and template copy| DRV
        M1 -->|template and dataset patch| MK
    end

    subgraph "2. compile_macro_bench.sh"
        BASH["rsync pj_gen & datasets; Loop compile_order.txt"]:::orchestrator
    end

    subgraph "3. RISC-V Board - jbpi2"
        DIR["Synced project directory"]:::target
        MAKE_ALL["make all g++ -march=rv64gcv -O2"]:::target
        BIN["test_gemm_{datatype}_{method}.x"]:::target
        RUN["make run - executes dataset benchmarks"]:::target
        LOGS["logs raw_results.txt and best_results.txt"]:::output
    end

    subgraph "4. Analysis"
        PLOT["plot_perf.py and plot_comparison.py"]:::plot
    end

    MK -.->|triggers| BASH
    ORDER -.->|guides loop| BASH
    BASH -->|rsync entire directory| DIR
    DIR --> MAKE_ALL
    MAKE_ALL --> BIN
    BIN --> RUN
    RUN --> LOGS
    LOGS -->|rsync back to host| PLOT
```
