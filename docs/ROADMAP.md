# Project Roadmap & Research Tracks

High-level development roadmap for `RVV_code_gen_via_MLIR_xDSL`. Tracks upcoming research, compiler dialect additions, and hardware optimizations.

Each track is synchronized with an active GitHub issue. When an issue is worked on, the agent updates both the issue on GitHub and this document.

---

## Active Tracks

### Track 1: Fixed-Dimension Evaluation & Packing/Compute Breakdown
- **GitHub Issue:** [#2](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/2)
- **Status:** `results-landed` — all 10 cases ran on `jbpi` 2026-09-16 with the new
  `Pack_Time_ms`/`Compute_Time_ms` breakdown; correctness not yet verified at these
  shapes (`skip_ref_run=true` throughout) and a Square-shape perf regression vs. the
  scalar-pack reference is still open. See `.scratch/issues/0002-fixed-dim-pack-compute-breakdown.md`.
- **Objective:** Systematically evaluate the pure xDSL BLIS pipeline with fixed $(MR, NR)$ dimensions to measure:
  1. Total GFLOPs
  2. Isolated time breakdown: Packing ($Ac$ and $Bc$) vs. Microkernel compute execution
- **Target Workload:** 10-case Square (`S1-S5`, 1000-5000 square, all `32x7`) + BERT (`B1-B5`, `32x6`/`16x15`/`16x15`/`32x7`/`32x6`) matrix — see Issue #2 for the exact m/n/k/MRxNR table.
- **Related Investigation:** Linked directly with Issue [#1 (B3_M7 RVV Packing Instability)](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/1).
- **Workflow Skill:** `/implement` (re-scoped from `/diagnosing-bugs` → `/implement`: this
  track turned out to be a build/measurement task, not a bug hunt).

---

### Track 2: Mixed-Precision Vectorized AXPY Kernels
- **GitHub Issue:** [#3](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/3)
- **Status:** `needs-triage`
- **Objective:** Expand the project generator (`generate_pure_blis_project.py`) and packing modules to support mixed-precision compute:
  - INT8 inputs with INT32 accumulation (`i8i8i32`)
  - BFloat16 / FP16 inputs with FP32 accumulation
- **Impact:** Aligns compiler generator with quantized deep learning inference workloads.
- **Workflow Skill:** `/to-spec` → `/to-tickets` → `/implement`

---

### Track 3: Google Highway xDSL Dialect (Cross-Platform Portability)
- **GitHub Issue:** [#4](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/4)
- **Status:** `needs-triage`
- **Objective:** Introduce a new xDSL dialect targeting [Google Highway](https://github.com/google/highway) to emit portable SIMD C++ for microkernels and packing.
- **Impact:** Generates a single portable kernel specification that compiles natively across RISC-V RVV, x86 AVX-512, and ARM Neon/SVE without maintaining separate compiler backends.
- **Workflow Skill:** `/wayfinder` (large-scope architectural initiative)

---

### Track 4: Banana Pi Accelerator Assembly & GCC 16 Intrinsics Spike
- **GitHub Issue:** [#5](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/5)
- **Status:** `needs-triage`
- **Objective:** Research spike to evaluate accelerator capabilities on Banana Pi:
  - Test if the accelerator extensions support C vector intrinsics with the GCC 16 release.
  - Explore direct assembly code generation for unsupported instructions.
- **Workflow Skill:** `/prototype` & `/research`
