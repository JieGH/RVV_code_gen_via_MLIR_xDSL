# RVV Code Generation via MLIR & xDSL

Generates high-performance RISC-V Vector (RVV) microkernels and macrokernels for BLIS-family GEMM using xDSL and MLIR pipelines, replacing static handwritten C benchmarks with a parameterized compiler-driven project generator.

## Language

### Hardware & Architecture

**Hardware Target**:
The RISC-V physical evaluation board executing the compiled kernels (specifically BananaPi M2 Pro / `jbpi2`).

**VLEN**:
Hardware vector register length in bits (256 bits on BananaPi M2 Pro). Parameterizes vector register tiling and `vsetvli` instruction emission.

**Analytical Cache Model**:
The hardware cache sizing model (`model_level`) that calculates optimal blocking dimensions (`MC`, `NC`, `KC`) based on L1/L2/L3 cache capacities and associativity.
_Avoid_: cache heuristic, manual tuning

### Matrix Operations & BLIS Structure

**Microkernel**:
The innermost register-tiled GEMM compute loop computing an $MR \times NR$ tile over $KC$, operating directly on vector registers and cache.
_Avoid_: inner kernel, sub-kernel

**Macrokernel**:
The loop hierarchy that partitions full matrices into $MC \times NC$ blocks and orchestrates packing before dispatching microkernels.

**Fringe Tile**:
A partial $MR \times NR$ tile at the edge of an $MC \times NC$ block, when the block size is not a multiple of $MR$ or $NR$. Our Macrokernel runs the full Microkernel on it through a `tmp_C` buffer; the Reference Design calls a Microkernel sized to the tile. The extra multiply-adds are quoted next to any comparison (`ref_baseline/README.md`).

**Packer**:
The transformation routine that reorders non-contiguous matrix panels into contiguous, cache-aligned buffers ($Ac$ and $Bc$) formatted for microkernel vector loads.
_Avoid_: reshaper, reorderer, buffer copier

**Scalar Packing**:
Packing routine implemented with scalar loads and stores, executing without vector instructions.

**RVV Packing**:
Packing routine accelerated with RISC-V Vector unit-stride or strided load/store instructions (`vlse32`/`vse32`).

### Framework & Build Pipeline

**Project Generator**:
The Python/xDSL framework (`generate_pure_blis_project.py`) that emits a self-contained benchmark suite directory (`pj_gen/`) containing the entire kernel family, headers, and standalone Makefile.

**Kernel Family**:
A cohesive set of $(MR, NR)$ dimension variants sharing an identical datatype, packing strategy, cache configuration, and VLEN.

**Reference Design**:
The legacy, baseline handwritten C implementation (`gemm_blis.c` or `.ext/RVV_ukernels_benchmark`) used for correctness verification and performance baselines.

**Reference Protocol**:
The Reference Design's way of timing a GEMM (`test_gemm.c`): no warm-up, reps run until `tmin` seconds have passed, one clock read per rep, correctness checked after timing. The default in the uninstrumented build. The alternative is the **Warm-up Protocol** (`--no-ref-protocol`): warm-up calls first, then a fixed number of reps.
_Avoid_: default protocol (ambiguous: the default changed)
