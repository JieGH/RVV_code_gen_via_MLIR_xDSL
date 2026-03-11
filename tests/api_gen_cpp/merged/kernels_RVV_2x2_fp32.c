#include "kernels_RVV_2x2_fp32.h"
#include <riscv_vector.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-03-11 10:07:23
//   MR         : 1
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================
// Merged from xdsl_api_gemm_1x1_b0.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-03-11 10:07:23
//   MR         : 1
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_1x1_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v21;
  v21 = v20;
  for (size_t i22 = v11; i22 < v1; i22 += v12) {
    size_t v23 = i22 * v4;
    size_t v24 = 1;
    float* v25 = &v3[v23];
    vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v24);
    size_t v27 = i22 * v6;
    vfloat32m1_t v28 = v21;
    float v29 = v5[v27];
    vfloat32m1_t v30 = __riscv_vfmacc_vf_f32m1(v28, v29, v26, v24);
    v21 = v30;
  }
  vfloat32m1_t v31 = v21;
  float* v32 = &v8[v11];
  __riscv_vse32_v_f32m1(v32, v31, v12);
  return;
}



// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-03-11 10:07:23
//   MR         : 1
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================
// Merged from xdsl_api_gemm_1x1_b1.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-03-11 10:07:23
//   MR         : 1
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_1x1_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v12);
  vfloat32m1_t v22;
  v22 = v21;
  for (size_t i23 = v11; i23 < v1; i23 += v12) {
    size_t v24 = i23 * v4;
    size_t v25 = 1;
    float* v26 = &v3[v24];
    vfloat32m1_t v27 = __riscv_vle32_v_f32m1(v26, v25);
    size_t v28 = i23 * v6;
    vfloat32m1_t v29 = v22;
    float v30 = v5[v28];
    vfloat32m1_t v31 = __riscv_vfmacc_vf_f32m1(v29, v30, v27, v25);
    v22 = v31;
  }
  vfloat32m1_t v32 = v22;
  float* v33 = &v8[v11];
  __riscv_vse32_v_f32m1(v33, v32, v12);
  return;
}



// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-03-11 10:07:23
//   MR         : 1
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================
// Merged from xdsl_api_gemm_1x2_b0.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-03-11 10:07:23
//   MR         : 1
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_1x2_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v22;
  v22 = v20;
  vfloat32m1_t v23;
  v23 = v21;
  for (size_t i24 = v11; i24 < v1; i24 += v12) {
    size_t v25 = i24 * v4;
    size_t v26 = 1;
    float* v27 = &v3[v25];
    vfloat32m1_t v28 = __riscv_vle32_v_f32m1(v27, v26);
    size_t v29 = i24 * v6;
    size_t v30 = v29 + v12;
    vfloat32m1_t v31 = v22;
    float v32 = v5[v29];
    vfloat32m1_t v33 = __riscv_vfmacc_vf_f32m1(v31, v32, v28, v26);
    vfloat32m1_t v34 = v23;
    float v35 = v5[v30];
    vfloat32m1_t v36 = __riscv_vfmacc_vf_f32m1(v34, v35, v28, v26);
    v22 = v33;
    v23 = v36;
  }
  vfloat32m1_t v37 = v22;
  vfloat32m1_t v38 = v23;
  float* v39 = &v8[v11];
  __riscv_vse32_v_f32m1(v39, v37, v12);
  float* v40 = &v8[v9];
  __riscv_vse32_v_f32m1(v40, v38, v12);
  return;
}



// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-03-11 10:07:23
//   MR         : 1
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================
// Merged from xdsl_api_gemm_1x2_b1.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-03-11 10:07:23
//   MR         : 1
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_1x2_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v12);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v12);
  vfloat32m1_t v24;
  v24 = v21;
  vfloat32m1_t v25;
  v25 = v23;
  for (size_t i26 = v11; i26 < v1; i26 += v12) {
    size_t v27 = i26 * v4;
    size_t v28 = 1;
    float* v29 = &v3[v27];
    vfloat32m1_t v30 = __riscv_vle32_v_f32m1(v29, v28);
    size_t v31 = i26 * v6;
    size_t v32 = v31 + v12;
    vfloat32m1_t v33 = v24;
    float v34 = v5[v31];
    vfloat32m1_t v35 = __riscv_vfmacc_vf_f32m1(v33, v34, v30, v28);
    vfloat32m1_t v36 = v25;
    float v37 = v5[v32];
    vfloat32m1_t v38 = __riscv_vfmacc_vf_f32m1(v36, v37, v30, v28);
    v24 = v35;
    v25 = v38;
  }
  vfloat32m1_t v39 = v24;
  vfloat32m1_t v40 = v25;
  float* v41 = &v8[v11];
  __riscv_vse32_v_f32m1(v41, v39, v12);
  float* v42 = &v8[v9];
  __riscv_vse32_v_f32m1(v42, v40, v12);
  return;
}



// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-03-11 10:07:23
//   MR         : 2
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================
// Merged from xdsl_api_gemm_2x1_b0.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-03-11 10:07:23
//   MR         : 2
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_2x1_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v21;
  v21 = v20;
  for (size_t i22 = v11; i22 < v1; i22 += v12) {
    size_t v23 = i22 * v4;
    size_t v24 = 2;
    float* v25 = &v3[v23];
    vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v24);
    size_t v27 = i22 * v6;
    vfloat32m1_t v28 = v21;
    float v29 = v5[v27];
    vfloat32m1_t v30 = __riscv_vfmacc_vf_f32m1(v28, v29, v26, v24);
    v21 = v30;
  }
  vfloat32m1_t v31 = v21;
  float* v32 = &v8[v11];
  __riscv_vse32_v_f32m1(v32, v31, v13);
  return;
}



// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-03-11 10:07:23
//   MR         : 2
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================
// Merged from xdsl_api_gemm_2x1_b1.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-03-11 10:07:23
//   MR         : 2
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_2x1_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v13);
  vfloat32m1_t v22;
  v22 = v21;
  for (size_t i23 = v11; i23 < v1; i23 += v12) {
    size_t v24 = i23 * v4;
    size_t v25 = 2;
    float* v26 = &v3[v24];
    vfloat32m1_t v27 = __riscv_vle32_v_f32m1(v26, v25);
    size_t v28 = i23 * v6;
    vfloat32m1_t v29 = v22;
    float v30 = v5[v28];
    vfloat32m1_t v31 = __riscv_vfmacc_vf_f32m1(v29, v30, v27, v25);
    v22 = v31;
  }
  vfloat32m1_t v32 = v22;
  float* v33 = &v8[v11];
  __riscv_vse32_v_f32m1(v33, v32, v13);
  return;
}



// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-03-11 10:07:23
//   MR         : 2
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================
// Merged from xdsl_api_gemm_2x2_b0.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-03-11 10:07:23
//   MR         : 2
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_2x2_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v22;
  v22 = v20;
  vfloat32m1_t v23;
  v23 = v21;
  for (size_t i24 = v11; i24 < v1; i24 += v12) {
    size_t v25 = i24 * v4;
    size_t v26 = 2;
    float* v27 = &v3[v25];
    vfloat32m1_t v28 = __riscv_vle32_v_f32m1(v27, v26);
    size_t v29 = i24 * v6;
    size_t v30 = v29 + v12;
    vfloat32m1_t v31 = v22;
    float v32 = v5[v29];
    vfloat32m1_t v33 = __riscv_vfmacc_vf_f32m1(v31, v32, v28, v26);
    vfloat32m1_t v34 = v23;
    float v35 = v5[v30];
    vfloat32m1_t v36 = __riscv_vfmacc_vf_f32m1(v34, v35, v28, v26);
    v22 = v33;
    v23 = v36;
  }
  vfloat32m1_t v37 = v22;
  vfloat32m1_t v38 = v23;
  float* v39 = &v8[v11];
  __riscv_vse32_v_f32m1(v39, v37, v13);
  float* v40 = &v8[v9];
  __riscv_vse32_v_f32m1(v40, v38, v13);
  return;
}



// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-03-11 10:07:23
//   MR         : 2
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================
// Merged from xdsl_api_gemm_2x2_b1.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-03-11 10:07:23
//   MR         : 2
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_2x2_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v13);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v13);
  vfloat32m1_t v24;
  v24 = v21;
  vfloat32m1_t v25;
  v25 = v23;
  for (size_t i26 = v11; i26 < v1; i26 += v12) {
    size_t v27 = i26 * v4;
    size_t v28 = 2;
    float* v29 = &v3[v27];
    vfloat32m1_t v30 = __riscv_vle32_v_f32m1(v29, v28);
    size_t v31 = i26 * v6;
    size_t v32 = v31 + v12;
    vfloat32m1_t v33 = v24;
    float v34 = v5[v31];
    vfloat32m1_t v35 = __riscv_vfmacc_vf_f32m1(v33, v34, v30, v28);
    vfloat32m1_t v36 = v25;
    float v37 = v5[v32];
    vfloat32m1_t v38 = __riscv_vfmacc_vf_f32m1(v36, v37, v30, v28);
    v24 = v35;
    v25 = v38;
  }
  vfloat32m1_t v39 = v24;
  vfloat32m1_t v40 = v25;
  float* v41 = &v8[v11];
  __riscv_vse32_v_f32m1(v41, v39, v13);
  float* v42 = &v8[v9];
  __riscv_vse32_v_f32m1(v42, v40, v13);
  return;
}



