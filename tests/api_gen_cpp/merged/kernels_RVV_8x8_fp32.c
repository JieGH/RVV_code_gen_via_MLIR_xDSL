// ============================================================
// xDSL Generated Merged Kernels
// ------------------------------------------------------------
// Precision : fp32
// Types     : A=float, B=float, C=float
// Generation: 2026-04-17 10:24:59
// ============================================================
#include "kernels_RVV_8x8_fp32.h"
#include <riscv_vector.h>

// ------------------------------------------------------------
// Merged from xdsl_api_gemm_1x1_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 1
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
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



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_1x1_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 1
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
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



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_1x2_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 1
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
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



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_1x2_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 1
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
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



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_1x3_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 1
//   NR         : 3
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_1x3_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v23;
  v23 = v20;
  vfloat32m1_t v24;
  v24 = v21;
  vfloat32m1_t v25;
  v25 = v22;
  for (size_t i26 = v11; i26 < v1; i26 += v12) {
    size_t v27 = i26 * v4;
    size_t v28 = 1;
    float* v29 = &v3[v27];
    vfloat32m1_t v30 = __riscv_vle32_v_f32m1(v29, v28);
    size_t v31 = i26 * v6;
    size_t v32 = v31 + v12;
    size_t v33 = v31 + v13;
    vfloat32m1_t v34 = v23;
    float v35 = v5[v31];
    vfloat32m1_t v36 = __riscv_vfmacc_vf_f32m1(v34, v35, v30, v28);
    vfloat32m1_t v37 = v24;
    float v38 = v5[v32];
    vfloat32m1_t v39 = __riscv_vfmacc_vf_f32m1(v37, v38, v30, v28);
    vfloat32m1_t v40 = v25;
    float v41 = v5[v33];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v30, v28);
    v23 = v36;
    v24 = v39;
    v25 = v42;
  }
  vfloat32m1_t v43 = v23;
  vfloat32m1_t v44 = v24;
  vfloat32m1_t v45 = v25;
  float* v46 = &v8[v11];
  __riscv_vse32_v_f32m1(v46, v43, v12);
  float* v47 = &v8[v9];
  __riscv_vse32_v_f32m1(v47, v44, v12);
  size_t v48 = v13 * v9;
  float* v49 = &v8[v48];
  __riscv_vse32_v_f32m1(v49, v45, v12);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_1x3_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 1
//   NR         : 3
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_1x3_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v12);
  vfloat32m1_t v27;
  v27 = v21;
  vfloat32m1_t v28;
  v28 = v23;
  vfloat32m1_t v29;
  v29 = v26;
  for (size_t i30 = v11; i30 < v1; i30 += v12) {
    size_t v31 = i30 * v4;
    size_t v32 = 1;
    float* v33 = &v3[v31];
    vfloat32m1_t v34 = __riscv_vle32_v_f32m1(v33, v32);
    size_t v35 = i30 * v6;
    size_t v36 = v35 + v12;
    size_t v37 = v35 + v13;
    vfloat32m1_t v38 = v27;
    float v39 = v5[v35];
    vfloat32m1_t v40 = __riscv_vfmacc_vf_f32m1(v38, v39, v34, v32);
    vfloat32m1_t v41 = v28;
    float v42 = v5[v36];
    vfloat32m1_t v43 = __riscv_vfmacc_vf_f32m1(v41, v42, v34, v32);
    vfloat32m1_t v44 = v29;
    float v45 = v5[v37];
    vfloat32m1_t v46 = __riscv_vfmacc_vf_f32m1(v44, v45, v34, v32);
    v27 = v40;
    v28 = v43;
    v29 = v46;
  }
  vfloat32m1_t v47 = v27;
  vfloat32m1_t v48 = v28;
  vfloat32m1_t v49 = v29;
  float* v50 = &v8[v11];
  __riscv_vse32_v_f32m1(v50, v47, v12);
  float* v51 = &v8[v9];
  __riscv_vse32_v_f32m1(v51, v48, v12);
  size_t v52 = v13 * v9;
  float* v53 = &v8[v52];
  __riscv_vse32_v_f32m1(v53, v49, v12);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_1x4_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 1
//   NR         : 4
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_1x4_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v24;
  v24 = v20;
  vfloat32m1_t v25;
  v25 = v21;
  vfloat32m1_t v26;
  v26 = v22;
  vfloat32m1_t v27;
  v27 = v23;
  for (size_t i28 = v11; i28 < v1; i28 += v12) {
    size_t v29 = i28 * v4;
    size_t v30 = 1;
    float* v31 = &v3[v29];
    vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v30);
    size_t v33 = i28 * v6;
    size_t v34 = v33 + v12;
    size_t v35 = v33 + v13;
    size_t v36 = v33 + v14;
    vfloat32m1_t v37 = v24;
    float v38 = v5[v33];
    vfloat32m1_t v39 = __riscv_vfmacc_vf_f32m1(v37, v38, v32, v30);
    vfloat32m1_t v40 = v25;
    float v41 = v5[v34];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v32, v30);
    vfloat32m1_t v43 = v26;
    float v44 = v5[v35];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v32, v30);
    vfloat32m1_t v46 = v27;
    float v47 = v5[v36];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v32, v30);
    v24 = v39;
    v25 = v42;
    v26 = v45;
    v27 = v48;
  }
  vfloat32m1_t v49 = v24;
  vfloat32m1_t v50 = v25;
  vfloat32m1_t v51 = v26;
  vfloat32m1_t v52 = v27;
  float* v53 = &v8[v11];
  __riscv_vse32_v_f32m1(v53, v49, v12);
  float* v54 = &v8[v9];
  __riscv_vse32_v_f32m1(v54, v50, v12);
  size_t v55 = v13 * v9;
  float* v56 = &v8[v55];
  __riscv_vse32_v_f32m1(v56, v51, v12);
  size_t v57 = v14 * v9;
  float* v58 = &v8[v57];
  __riscv_vse32_v_f32m1(v58, v52, v12);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_1x4_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 1
//   NR         : 4
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_1x4_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v12);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v12);
  vfloat32m1_t v30;
  v30 = v21;
  vfloat32m1_t v31;
  v31 = v23;
  vfloat32m1_t v32;
  v32 = v26;
  vfloat32m1_t v33;
  v33 = v29;
  for (size_t i34 = v11; i34 < v1; i34 += v12) {
    size_t v35 = i34 * v4;
    size_t v36 = 1;
    float* v37 = &v3[v35];
    vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v36);
    size_t v39 = i34 * v6;
    size_t v40 = v39 + v12;
    size_t v41 = v39 + v13;
    size_t v42 = v39 + v14;
    vfloat32m1_t v43 = v30;
    float v44 = v5[v39];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v38, v36);
    vfloat32m1_t v46 = v31;
    float v47 = v5[v40];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v38, v36);
    vfloat32m1_t v49 = v32;
    float v50 = v5[v41];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v38, v36);
    vfloat32m1_t v52 = v33;
    float v53 = v5[v42];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v38, v36);
    v30 = v45;
    v31 = v48;
    v32 = v51;
    v33 = v54;
  }
  vfloat32m1_t v55 = v30;
  vfloat32m1_t v56 = v31;
  vfloat32m1_t v57 = v32;
  vfloat32m1_t v58 = v33;
  float* v59 = &v8[v11];
  __riscv_vse32_v_f32m1(v59, v55, v12);
  float* v60 = &v8[v9];
  __riscv_vse32_v_f32m1(v60, v56, v12);
  size_t v61 = v13 * v9;
  float* v62 = &v8[v61];
  __riscv_vse32_v_f32m1(v62, v57, v12);
  size_t v63 = v14 * v9;
  float* v64 = &v8[v63];
  __riscv_vse32_v_f32m1(v64, v58, v12);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_1x5_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 1
//   NR         : 5
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_1x5_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v24 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v25;
  v25 = v20;
  vfloat32m1_t v26;
  v26 = v21;
  vfloat32m1_t v27;
  v27 = v22;
  vfloat32m1_t v28;
  v28 = v23;
  vfloat32m1_t v29;
  v29 = v24;
  for (size_t i30 = v11; i30 < v1; i30 += v12) {
    size_t v31 = i30 * v4;
    size_t v32 = 1;
    float* v33 = &v3[v31];
    vfloat32m1_t v34 = __riscv_vle32_v_f32m1(v33, v32);
    size_t v35 = i30 * v6;
    size_t v36 = v35 + v12;
    size_t v37 = v35 + v13;
    size_t v38 = v35 + v14;
    size_t v39 = v35 + v15;
    vfloat32m1_t v40 = v25;
    float v41 = v5[v35];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v34, v32);
    vfloat32m1_t v43 = v26;
    float v44 = v5[v36];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v34, v32);
    vfloat32m1_t v46 = v27;
    float v47 = v5[v37];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v34, v32);
    vfloat32m1_t v49 = v28;
    float v50 = v5[v38];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v34, v32);
    vfloat32m1_t v52 = v29;
    float v53 = v5[v39];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v34, v32);
    v25 = v42;
    v26 = v45;
    v27 = v48;
    v28 = v51;
    v29 = v54;
  }
  vfloat32m1_t v55 = v25;
  vfloat32m1_t v56 = v26;
  vfloat32m1_t v57 = v27;
  vfloat32m1_t v58 = v28;
  vfloat32m1_t v59 = v29;
  float* v60 = &v8[v11];
  __riscv_vse32_v_f32m1(v60, v55, v12);
  float* v61 = &v8[v9];
  __riscv_vse32_v_f32m1(v61, v56, v12);
  size_t v62 = v13 * v9;
  float* v63 = &v8[v62];
  __riscv_vse32_v_f32m1(v63, v57, v12);
  size_t v64 = v14 * v9;
  float* v65 = &v8[v64];
  __riscv_vse32_v_f32m1(v65, v58, v12);
  size_t v66 = v15 * v9;
  float* v67 = &v8[v66];
  __riscv_vse32_v_f32m1(v67, v59, v12);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_1x5_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 1
//   NR         : 5
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_1x5_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v12);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v12);
  size_t v30 = v15 * v9;
  float* v31 = &v8[v30];
  vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v12);
  vfloat32m1_t v33;
  v33 = v21;
  vfloat32m1_t v34;
  v34 = v23;
  vfloat32m1_t v35;
  v35 = v26;
  vfloat32m1_t v36;
  v36 = v29;
  vfloat32m1_t v37;
  v37 = v32;
  for (size_t i38 = v11; i38 < v1; i38 += v12) {
    size_t v39 = i38 * v4;
    size_t v40 = 1;
    float* v41 = &v3[v39];
    vfloat32m1_t v42 = __riscv_vle32_v_f32m1(v41, v40);
    size_t v43 = i38 * v6;
    size_t v44 = v43 + v12;
    size_t v45 = v43 + v13;
    size_t v46 = v43 + v14;
    size_t v47 = v43 + v15;
    vfloat32m1_t v48 = v33;
    float v49 = v5[v43];
    vfloat32m1_t v50 = __riscv_vfmacc_vf_f32m1(v48, v49, v42, v40);
    vfloat32m1_t v51 = v34;
    float v52 = v5[v44];
    vfloat32m1_t v53 = __riscv_vfmacc_vf_f32m1(v51, v52, v42, v40);
    vfloat32m1_t v54 = v35;
    float v55 = v5[v45];
    vfloat32m1_t v56 = __riscv_vfmacc_vf_f32m1(v54, v55, v42, v40);
    vfloat32m1_t v57 = v36;
    float v58 = v5[v46];
    vfloat32m1_t v59 = __riscv_vfmacc_vf_f32m1(v57, v58, v42, v40);
    vfloat32m1_t v60 = v37;
    float v61 = v5[v47];
    vfloat32m1_t v62 = __riscv_vfmacc_vf_f32m1(v60, v61, v42, v40);
    v33 = v50;
    v34 = v53;
    v35 = v56;
    v36 = v59;
    v37 = v62;
  }
  vfloat32m1_t v63 = v33;
  vfloat32m1_t v64 = v34;
  vfloat32m1_t v65 = v35;
  vfloat32m1_t v66 = v36;
  vfloat32m1_t v67 = v37;
  float* v68 = &v8[v11];
  __riscv_vse32_v_f32m1(v68, v63, v12);
  float* v69 = &v8[v9];
  __riscv_vse32_v_f32m1(v69, v64, v12);
  size_t v70 = v13 * v9;
  float* v71 = &v8[v70];
  __riscv_vse32_v_f32m1(v71, v65, v12);
  size_t v72 = v14 * v9;
  float* v73 = &v8[v72];
  __riscv_vse32_v_f32m1(v73, v66, v12);
  size_t v74 = v15 * v9;
  float* v75 = &v8[v74];
  __riscv_vse32_v_f32m1(v75, v67, v12);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_1x6_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 1
//   NR         : 6
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_1x6_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v24 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v25 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v26;
  v26 = v20;
  vfloat32m1_t v27;
  v27 = v21;
  vfloat32m1_t v28;
  v28 = v22;
  vfloat32m1_t v29;
  v29 = v23;
  vfloat32m1_t v30;
  v30 = v24;
  vfloat32m1_t v31;
  v31 = v25;
  for (size_t i32 = v11; i32 < v1; i32 += v12) {
    size_t v33 = i32 * v4;
    size_t v34 = 1;
    float* v35 = &v3[v33];
    vfloat32m1_t v36 = __riscv_vle32_v_f32m1(v35, v34);
    size_t v37 = i32 * v6;
    size_t v38 = v37 + v12;
    size_t v39 = v37 + v13;
    size_t v40 = v37 + v14;
    size_t v41 = v37 + v15;
    size_t v42 = v37 + v16;
    vfloat32m1_t v43 = v26;
    float v44 = v5[v37];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v36, v34);
    vfloat32m1_t v46 = v27;
    float v47 = v5[v38];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v36, v34);
    vfloat32m1_t v49 = v28;
    float v50 = v5[v39];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v36, v34);
    vfloat32m1_t v52 = v29;
    float v53 = v5[v40];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v36, v34);
    vfloat32m1_t v55 = v30;
    float v56 = v5[v41];
    vfloat32m1_t v57 = __riscv_vfmacc_vf_f32m1(v55, v56, v36, v34);
    vfloat32m1_t v58 = v31;
    float v59 = v5[v42];
    vfloat32m1_t v60 = __riscv_vfmacc_vf_f32m1(v58, v59, v36, v34);
    v26 = v45;
    v27 = v48;
    v28 = v51;
    v29 = v54;
    v30 = v57;
    v31 = v60;
  }
  vfloat32m1_t v61 = v26;
  vfloat32m1_t v62 = v27;
  vfloat32m1_t v63 = v28;
  vfloat32m1_t v64 = v29;
  vfloat32m1_t v65 = v30;
  vfloat32m1_t v66 = v31;
  float* v67 = &v8[v11];
  __riscv_vse32_v_f32m1(v67, v61, v12);
  float* v68 = &v8[v9];
  __riscv_vse32_v_f32m1(v68, v62, v12);
  size_t v69 = v13 * v9;
  float* v70 = &v8[v69];
  __riscv_vse32_v_f32m1(v70, v63, v12);
  size_t v71 = v14 * v9;
  float* v72 = &v8[v71];
  __riscv_vse32_v_f32m1(v72, v64, v12);
  size_t v73 = v15 * v9;
  float* v74 = &v8[v73];
  __riscv_vse32_v_f32m1(v74, v65, v12);
  size_t v75 = v16 * v9;
  float* v76 = &v8[v75];
  __riscv_vse32_v_f32m1(v76, v66, v12);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_1x6_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 1
//   NR         : 6
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_1x6_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v12);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v12);
  size_t v30 = v15 * v9;
  float* v31 = &v8[v30];
  vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v12);
  size_t v33 = v16 * v9;
  float* v34 = &v8[v33];
  vfloat32m1_t v35 = __riscv_vle32_v_f32m1(v34, v12);
  vfloat32m1_t v36;
  v36 = v21;
  vfloat32m1_t v37;
  v37 = v23;
  vfloat32m1_t v38;
  v38 = v26;
  vfloat32m1_t v39;
  v39 = v29;
  vfloat32m1_t v40;
  v40 = v32;
  vfloat32m1_t v41;
  v41 = v35;
  for (size_t i42 = v11; i42 < v1; i42 += v12) {
    size_t v43 = i42 * v4;
    size_t v44 = 1;
    float* v45 = &v3[v43];
    vfloat32m1_t v46 = __riscv_vle32_v_f32m1(v45, v44);
    size_t v47 = i42 * v6;
    size_t v48 = v47 + v12;
    size_t v49 = v47 + v13;
    size_t v50 = v47 + v14;
    size_t v51 = v47 + v15;
    size_t v52 = v47 + v16;
    vfloat32m1_t v53 = v36;
    float v54 = v5[v47];
    vfloat32m1_t v55 = __riscv_vfmacc_vf_f32m1(v53, v54, v46, v44);
    vfloat32m1_t v56 = v37;
    float v57 = v5[v48];
    vfloat32m1_t v58 = __riscv_vfmacc_vf_f32m1(v56, v57, v46, v44);
    vfloat32m1_t v59 = v38;
    float v60 = v5[v49];
    vfloat32m1_t v61 = __riscv_vfmacc_vf_f32m1(v59, v60, v46, v44);
    vfloat32m1_t v62 = v39;
    float v63 = v5[v50];
    vfloat32m1_t v64 = __riscv_vfmacc_vf_f32m1(v62, v63, v46, v44);
    vfloat32m1_t v65 = v40;
    float v66 = v5[v51];
    vfloat32m1_t v67 = __riscv_vfmacc_vf_f32m1(v65, v66, v46, v44);
    vfloat32m1_t v68 = v41;
    float v69 = v5[v52];
    vfloat32m1_t v70 = __riscv_vfmacc_vf_f32m1(v68, v69, v46, v44);
    v36 = v55;
    v37 = v58;
    v38 = v61;
    v39 = v64;
    v40 = v67;
    v41 = v70;
  }
  vfloat32m1_t v71 = v36;
  vfloat32m1_t v72 = v37;
  vfloat32m1_t v73 = v38;
  vfloat32m1_t v74 = v39;
  vfloat32m1_t v75 = v40;
  vfloat32m1_t v76 = v41;
  float* v77 = &v8[v11];
  __riscv_vse32_v_f32m1(v77, v71, v12);
  float* v78 = &v8[v9];
  __riscv_vse32_v_f32m1(v78, v72, v12);
  size_t v79 = v13 * v9;
  float* v80 = &v8[v79];
  __riscv_vse32_v_f32m1(v80, v73, v12);
  size_t v81 = v14 * v9;
  float* v82 = &v8[v81];
  __riscv_vse32_v_f32m1(v82, v74, v12);
  size_t v83 = v15 * v9;
  float* v84 = &v8[v83];
  __riscv_vse32_v_f32m1(v84, v75, v12);
  size_t v85 = v16 * v9;
  float* v86 = &v8[v85];
  __riscv_vse32_v_f32m1(v86, v76, v12);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_1x7_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 1
//   NR         : 7
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_1x7_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v24 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v25 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v26 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v27;
  v27 = v20;
  vfloat32m1_t v28;
  v28 = v21;
  vfloat32m1_t v29;
  v29 = v22;
  vfloat32m1_t v30;
  v30 = v23;
  vfloat32m1_t v31;
  v31 = v24;
  vfloat32m1_t v32;
  v32 = v25;
  vfloat32m1_t v33;
  v33 = v26;
  for (size_t i34 = v11; i34 < v1; i34 += v12) {
    size_t v35 = i34 * v4;
    size_t v36 = 1;
    float* v37 = &v3[v35];
    vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v36);
    size_t v39 = i34 * v6;
    size_t v40 = v39 + v12;
    size_t v41 = v39 + v13;
    size_t v42 = v39 + v14;
    size_t v43 = v39 + v15;
    size_t v44 = v39 + v16;
    size_t v45 = v39 + v17;
    vfloat32m1_t v46 = v27;
    float v47 = v5[v39];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v38, v36);
    vfloat32m1_t v49 = v28;
    float v50 = v5[v40];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v38, v36);
    vfloat32m1_t v52 = v29;
    float v53 = v5[v41];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v38, v36);
    vfloat32m1_t v55 = v30;
    float v56 = v5[v42];
    vfloat32m1_t v57 = __riscv_vfmacc_vf_f32m1(v55, v56, v38, v36);
    vfloat32m1_t v58 = v31;
    float v59 = v5[v43];
    vfloat32m1_t v60 = __riscv_vfmacc_vf_f32m1(v58, v59, v38, v36);
    vfloat32m1_t v61 = v32;
    float v62 = v5[v44];
    vfloat32m1_t v63 = __riscv_vfmacc_vf_f32m1(v61, v62, v38, v36);
    vfloat32m1_t v64 = v33;
    float v65 = v5[v45];
    vfloat32m1_t v66 = __riscv_vfmacc_vf_f32m1(v64, v65, v38, v36);
    v27 = v48;
    v28 = v51;
    v29 = v54;
    v30 = v57;
    v31 = v60;
    v32 = v63;
    v33 = v66;
  }
  vfloat32m1_t v67 = v27;
  vfloat32m1_t v68 = v28;
  vfloat32m1_t v69 = v29;
  vfloat32m1_t v70 = v30;
  vfloat32m1_t v71 = v31;
  vfloat32m1_t v72 = v32;
  vfloat32m1_t v73 = v33;
  float* v74 = &v8[v11];
  __riscv_vse32_v_f32m1(v74, v67, v12);
  float* v75 = &v8[v9];
  __riscv_vse32_v_f32m1(v75, v68, v12);
  size_t v76 = v13 * v9;
  float* v77 = &v8[v76];
  __riscv_vse32_v_f32m1(v77, v69, v12);
  size_t v78 = v14 * v9;
  float* v79 = &v8[v78];
  __riscv_vse32_v_f32m1(v79, v70, v12);
  size_t v80 = v15 * v9;
  float* v81 = &v8[v80];
  __riscv_vse32_v_f32m1(v81, v71, v12);
  size_t v82 = v16 * v9;
  float* v83 = &v8[v82];
  __riscv_vse32_v_f32m1(v83, v72, v12);
  size_t v84 = v17 * v9;
  float* v85 = &v8[v84];
  __riscv_vse32_v_f32m1(v85, v73, v12);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_1x7_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 1
//   NR         : 7
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_1x7_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v12);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v12);
  size_t v30 = v15 * v9;
  float* v31 = &v8[v30];
  vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v12);
  size_t v33 = v16 * v9;
  float* v34 = &v8[v33];
  vfloat32m1_t v35 = __riscv_vle32_v_f32m1(v34, v12);
  size_t v36 = v17 * v9;
  float* v37 = &v8[v36];
  vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v12);
  vfloat32m1_t v39;
  v39 = v21;
  vfloat32m1_t v40;
  v40 = v23;
  vfloat32m1_t v41;
  v41 = v26;
  vfloat32m1_t v42;
  v42 = v29;
  vfloat32m1_t v43;
  v43 = v32;
  vfloat32m1_t v44;
  v44 = v35;
  vfloat32m1_t v45;
  v45 = v38;
  for (size_t i46 = v11; i46 < v1; i46 += v12) {
    size_t v47 = i46 * v4;
    size_t v48 = 1;
    float* v49 = &v3[v47];
    vfloat32m1_t v50 = __riscv_vle32_v_f32m1(v49, v48);
    size_t v51 = i46 * v6;
    size_t v52 = v51 + v12;
    size_t v53 = v51 + v13;
    size_t v54 = v51 + v14;
    size_t v55 = v51 + v15;
    size_t v56 = v51 + v16;
    size_t v57 = v51 + v17;
    vfloat32m1_t v58 = v39;
    float v59 = v5[v51];
    vfloat32m1_t v60 = __riscv_vfmacc_vf_f32m1(v58, v59, v50, v48);
    vfloat32m1_t v61 = v40;
    float v62 = v5[v52];
    vfloat32m1_t v63 = __riscv_vfmacc_vf_f32m1(v61, v62, v50, v48);
    vfloat32m1_t v64 = v41;
    float v65 = v5[v53];
    vfloat32m1_t v66 = __riscv_vfmacc_vf_f32m1(v64, v65, v50, v48);
    vfloat32m1_t v67 = v42;
    float v68 = v5[v54];
    vfloat32m1_t v69 = __riscv_vfmacc_vf_f32m1(v67, v68, v50, v48);
    vfloat32m1_t v70 = v43;
    float v71 = v5[v55];
    vfloat32m1_t v72 = __riscv_vfmacc_vf_f32m1(v70, v71, v50, v48);
    vfloat32m1_t v73 = v44;
    float v74 = v5[v56];
    vfloat32m1_t v75 = __riscv_vfmacc_vf_f32m1(v73, v74, v50, v48);
    vfloat32m1_t v76 = v45;
    float v77 = v5[v57];
    vfloat32m1_t v78 = __riscv_vfmacc_vf_f32m1(v76, v77, v50, v48);
    v39 = v60;
    v40 = v63;
    v41 = v66;
    v42 = v69;
    v43 = v72;
    v44 = v75;
    v45 = v78;
  }
  vfloat32m1_t v79 = v39;
  vfloat32m1_t v80 = v40;
  vfloat32m1_t v81 = v41;
  vfloat32m1_t v82 = v42;
  vfloat32m1_t v83 = v43;
  vfloat32m1_t v84 = v44;
  vfloat32m1_t v85 = v45;
  float* v86 = &v8[v11];
  __riscv_vse32_v_f32m1(v86, v79, v12);
  float* v87 = &v8[v9];
  __riscv_vse32_v_f32m1(v87, v80, v12);
  size_t v88 = v13 * v9;
  float* v89 = &v8[v88];
  __riscv_vse32_v_f32m1(v89, v81, v12);
  size_t v90 = v14 * v9;
  float* v91 = &v8[v90];
  __riscv_vse32_v_f32m1(v91, v82, v12);
  size_t v92 = v15 * v9;
  float* v93 = &v8[v92];
  __riscv_vse32_v_f32m1(v93, v83, v12);
  size_t v94 = v16 * v9;
  float* v95 = &v8[v94];
  __riscv_vse32_v_f32m1(v95, v84, v12);
  size_t v96 = v17 * v9;
  float* v97 = &v8[v96];
  __riscv_vse32_v_f32m1(v97, v85, v12);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_1x8_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 1
//   NR         : 8
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_1x8_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v24 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v25 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v26 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v27 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v28;
  v28 = v20;
  vfloat32m1_t v29;
  v29 = v21;
  vfloat32m1_t v30;
  v30 = v22;
  vfloat32m1_t v31;
  v31 = v23;
  vfloat32m1_t v32;
  v32 = v24;
  vfloat32m1_t v33;
  v33 = v25;
  vfloat32m1_t v34;
  v34 = v26;
  vfloat32m1_t v35;
  v35 = v27;
  for (size_t i36 = v11; i36 < v1; i36 += v12) {
    size_t v37 = i36 * v4;
    size_t v38 = 1;
    float* v39 = &v3[v37];
    vfloat32m1_t v40 = __riscv_vle32_v_f32m1(v39, v38);
    size_t v41 = i36 * v6;
    size_t v42 = v41 + v12;
    size_t v43 = v41 + v13;
    size_t v44 = v41 + v14;
    size_t v45 = v41 + v15;
    size_t v46 = v41 + v16;
    size_t v47 = v41 + v17;
    size_t v48 = v41 + v18;
    vfloat32m1_t v49 = v28;
    float v50 = v5[v41];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v40, v38);
    vfloat32m1_t v52 = v29;
    float v53 = v5[v42];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v40, v38);
    vfloat32m1_t v55 = v30;
    float v56 = v5[v43];
    vfloat32m1_t v57 = __riscv_vfmacc_vf_f32m1(v55, v56, v40, v38);
    vfloat32m1_t v58 = v31;
    float v59 = v5[v44];
    vfloat32m1_t v60 = __riscv_vfmacc_vf_f32m1(v58, v59, v40, v38);
    vfloat32m1_t v61 = v32;
    float v62 = v5[v45];
    vfloat32m1_t v63 = __riscv_vfmacc_vf_f32m1(v61, v62, v40, v38);
    vfloat32m1_t v64 = v33;
    float v65 = v5[v46];
    vfloat32m1_t v66 = __riscv_vfmacc_vf_f32m1(v64, v65, v40, v38);
    vfloat32m1_t v67 = v34;
    float v68 = v5[v47];
    vfloat32m1_t v69 = __riscv_vfmacc_vf_f32m1(v67, v68, v40, v38);
    vfloat32m1_t v70 = v35;
    float v71 = v5[v48];
    vfloat32m1_t v72 = __riscv_vfmacc_vf_f32m1(v70, v71, v40, v38);
    v28 = v51;
    v29 = v54;
    v30 = v57;
    v31 = v60;
    v32 = v63;
    v33 = v66;
    v34 = v69;
    v35 = v72;
  }
  vfloat32m1_t v73 = v28;
  vfloat32m1_t v74 = v29;
  vfloat32m1_t v75 = v30;
  vfloat32m1_t v76 = v31;
  vfloat32m1_t v77 = v32;
  vfloat32m1_t v78 = v33;
  vfloat32m1_t v79 = v34;
  vfloat32m1_t v80 = v35;
  float* v81 = &v8[v11];
  __riscv_vse32_v_f32m1(v81, v73, v12);
  float* v82 = &v8[v9];
  __riscv_vse32_v_f32m1(v82, v74, v12);
  size_t v83 = v13 * v9;
  float* v84 = &v8[v83];
  __riscv_vse32_v_f32m1(v84, v75, v12);
  size_t v85 = v14 * v9;
  float* v86 = &v8[v85];
  __riscv_vse32_v_f32m1(v86, v76, v12);
  size_t v87 = v15 * v9;
  float* v88 = &v8[v87];
  __riscv_vse32_v_f32m1(v88, v77, v12);
  size_t v89 = v16 * v9;
  float* v90 = &v8[v89];
  __riscv_vse32_v_f32m1(v90, v78, v12);
  size_t v91 = v17 * v9;
  float* v92 = &v8[v91];
  __riscv_vse32_v_f32m1(v92, v79, v12);
  size_t v93 = v18 * v9;
  float* v94 = &v8[v93];
  __riscv_vse32_v_f32m1(v94, v80, v12);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_1x8_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 1
//   NR         : 8
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_1x8_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v12);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v12);
  size_t v30 = v15 * v9;
  float* v31 = &v8[v30];
  vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v12);
  size_t v33 = v16 * v9;
  float* v34 = &v8[v33];
  vfloat32m1_t v35 = __riscv_vle32_v_f32m1(v34, v12);
  size_t v36 = v17 * v9;
  float* v37 = &v8[v36];
  vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v12);
  size_t v39 = v18 * v9;
  float* v40 = &v8[v39];
  vfloat32m1_t v41 = __riscv_vle32_v_f32m1(v40, v12);
  vfloat32m1_t v42;
  v42 = v21;
  vfloat32m1_t v43;
  v43 = v23;
  vfloat32m1_t v44;
  v44 = v26;
  vfloat32m1_t v45;
  v45 = v29;
  vfloat32m1_t v46;
  v46 = v32;
  vfloat32m1_t v47;
  v47 = v35;
  vfloat32m1_t v48;
  v48 = v38;
  vfloat32m1_t v49;
  v49 = v41;
  for (size_t i50 = v11; i50 < v1; i50 += v12) {
    size_t v51 = i50 * v4;
    size_t v52 = 1;
    float* v53 = &v3[v51];
    vfloat32m1_t v54 = __riscv_vle32_v_f32m1(v53, v52);
    size_t v55 = i50 * v6;
    size_t v56 = v55 + v12;
    size_t v57 = v55 + v13;
    size_t v58 = v55 + v14;
    size_t v59 = v55 + v15;
    size_t v60 = v55 + v16;
    size_t v61 = v55 + v17;
    size_t v62 = v55 + v18;
    vfloat32m1_t v63 = v42;
    float v64 = v5[v55];
    vfloat32m1_t v65 = __riscv_vfmacc_vf_f32m1(v63, v64, v54, v52);
    vfloat32m1_t v66 = v43;
    float v67 = v5[v56];
    vfloat32m1_t v68 = __riscv_vfmacc_vf_f32m1(v66, v67, v54, v52);
    vfloat32m1_t v69 = v44;
    float v70 = v5[v57];
    vfloat32m1_t v71 = __riscv_vfmacc_vf_f32m1(v69, v70, v54, v52);
    vfloat32m1_t v72 = v45;
    float v73 = v5[v58];
    vfloat32m1_t v74 = __riscv_vfmacc_vf_f32m1(v72, v73, v54, v52);
    vfloat32m1_t v75 = v46;
    float v76 = v5[v59];
    vfloat32m1_t v77 = __riscv_vfmacc_vf_f32m1(v75, v76, v54, v52);
    vfloat32m1_t v78 = v47;
    float v79 = v5[v60];
    vfloat32m1_t v80 = __riscv_vfmacc_vf_f32m1(v78, v79, v54, v52);
    vfloat32m1_t v81 = v48;
    float v82 = v5[v61];
    vfloat32m1_t v83 = __riscv_vfmacc_vf_f32m1(v81, v82, v54, v52);
    vfloat32m1_t v84 = v49;
    float v85 = v5[v62];
    vfloat32m1_t v86 = __riscv_vfmacc_vf_f32m1(v84, v85, v54, v52);
    v42 = v65;
    v43 = v68;
    v44 = v71;
    v45 = v74;
    v46 = v77;
    v47 = v80;
    v48 = v83;
    v49 = v86;
  }
  vfloat32m1_t v87 = v42;
  vfloat32m1_t v88 = v43;
  vfloat32m1_t v89 = v44;
  vfloat32m1_t v90 = v45;
  vfloat32m1_t v91 = v46;
  vfloat32m1_t v92 = v47;
  vfloat32m1_t v93 = v48;
  vfloat32m1_t v94 = v49;
  float* v95 = &v8[v11];
  __riscv_vse32_v_f32m1(v95, v87, v12);
  float* v96 = &v8[v9];
  __riscv_vse32_v_f32m1(v96, v88, v12);
  size_t v97 = v13 * v9;
  float* v98 = &v8[v97];
  __riscv_vse32_v_f32m1(v98, v89, v12);
  size_t v99 = v14 * v9;
  float* v100 = &v8[v99];
  __riscv_vse32_v_f32m1(v100, v90, v12);
  size_t v101 = v15 * v9;
  float* v102 = &v8[v101];
  __riscv_vse32_v_f32m1(v102, v91, v12);
  size_t v103 = v16 * v9;
  float* v104 = &v8[v103];
  __riscv_vse32_v_f32m1(v104, v92, v12);
  size_t v105 = v17 * v9;
  float* v106 = &v8[v105];
  __riscv_vse32_v_f32m1(v106, v93, v12);
  size_t v107 = v18 * v9;
  float* v108 = &v8[v107];
  __riscv_vse32_v_f32m1(v108, v94, v12);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_2x1_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 2
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
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



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_2x1_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 2
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
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



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_2x2_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 2
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
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



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_2x2_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 2
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
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



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_2x3_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 2
//   NR         : 3
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_2x3_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v23;
  v23 = v20;
  vfloat32m1_t v24;
  v24 = v21;
  vfloat32m1_t v25;
  v25 = v22;
  for (size_t i26 = v11; i26 < v1; i26 += v12) {
    size_t v27 = i26 * v4;
    size_t v28 = 2;
    float* v29 = &v3[v27];
    vfloat32m1_t v30 = __riscv_vle32_v_f32m1(v29, v28);
    size_t v31 = i26 * v6;
    size_t v32 = v31 + v12;
    size_t v33 = v31 + v13;
    vfloat32m1_t v34 = v23;
    float v35 = v5[v31];
    vfloat32m1_t v36 = __riscv_vfmacc_vf_f32m1(v34, v35, v30, v28);
    vfloat32m1_t v37 = v24;
    float v38 = v5[v32];
    vfloat32m1_t v39 = __riscv_vfmacc_vf_f32m1(v37, v38, v30, v28);
    vfloat32m1_t v40 = v25;
    float v41 = v5[v33];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v30, v28);
    v23 = v36;
    v24 = v39;
    v25 = v42;
  }
  vfloat32m1_t v43 = v23;
  vfloat32m1_t v44 = v24;
  vfloat32m1_t v45 = v25;
  float* v46 = &v8[v11];
  __riscv_vse32_v_f32m1(v46, v43, v13);
  float* v47 = &v8[v9];
  __riscv_vse32_v_f32m1(v47, v44, v13);
  size_t v48 = v13 * v9;
  float* v49 = &v8[v48];
  __riscv_vse32_v_f32m1(v49, v45, v13);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_2x3_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 2
//   NR         : 3
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_2x3_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v13);
  vfloat32m1_t v27;
  v27 = v21;
  vfloat32m1_t v28;
  v28 = v23;
  vfloat32m1_t v29;
  v29 = v26;
  for (size_t i30 = v11; i30 < v1; i30 += v12) {
    size_t v31 = i30 * v4;
    size_t v32 = 2;
    float* v33 = &v3[v31];
    vfloat32m1_t v34 = __riscv_vle32_v_f32m1(v33, v32);
    size_t v35 = i30 * v6;
    size_t v36 = v35 + v12;
    size_t v37 = v35 + v13;
    vfloat32m1_t v38 = v27;
    float v39 = v5[v35];
    vfloat32m1_t v40 = __riscv_vfmacc_vf_f32m1(v38, v39, v34, v32);
    vfloat32m1_t v41 = v28;
    float v42 = v5[v36];
    vfloat32m1_t v43 = __riscv_vfmacc_vf_f32m1(v41, v42, v34, v32);
    vfloat32m1_t v44 = v29;
    float v45 = v5[v37];
    vfloat32m1_t v46 = __riscv_vfmacc_vf_f32m1(v44, v45, v34, v32);
    v27 = v40;
    v28 = v43;
    v29 = v46;
  }
  vfloat32m1_t v47 = v27;
  vfloat32m1_t v48 = v28;
  vfloat32m1_t v49 = v29;
  float* v50 = &v8[v11];
  __riscv_vse32_v_f32m1(v50, v47, v13);
  float* v51 = &v8[v9];
  __riscv_vse32_v_f32m1(v51, v48, v13);
  size_t v52 = v13 * v9;
  float* v53 = &v8[v52];
  __riscv_vse32_v_f32m1(v53, v49, v13);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_2x4_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 2
//   NR         : 4
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_2x4_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v24;
  v24 = v20;
  vfloat32m1_t v25;
  v25 = v21;
  vfloat32m1_t v26;
  v26 = v22;
  vfloat32m1_t v27;
  v27 = v23;
  for (size_t i28 = v11; i28 < v1; i28 += v12) {
    size_t v29 = i28 * v4;
    size_t v30 = 2;
    float* v31 = &v3[v29];
    vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v30);
    size_t v33 = i28 * v6;
    size_t v34 = v33 + v12;
    size_t v35 = v33 + v13;
    size_t v36 = v33 + v14;
    vfloat32m1_t v37 = v24;
    float v38 = v5[v33];
    vfloat32m1_t v39 = __riscv_vfmacc_vf_f32m1(v37, v38, v32, v30);
    vfloat32m1_t v40 = v25;
    float v41 = v5[v34];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v32, v30);
    vfloat32m1_t v43 = v26;
    float v44 = v5[v35];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v32, v30);
    vfloat32m1_t v46 = v27;
    float v47 = v5[v36];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v32, v30);
    v24 = v39;
    v25 = v42;
    v26 = v45;
    v27 = v48;
  }
  vfloat32m1_t v49 = v24;
  vfloat32m1_t v50 = v25;
  vfloat32m1_t v51 = v26;
  vfloat32m1_t v52 = v27;
  float* v53 = &v8[v11];
  __riscv_vse32_v_f32m1(v53, v49, v13);
  float* v54 = &v8[v9];
  __riscv_vse32_v_f32m1(v54, v50, v13);
  size_t v55 = v13 * v9;
  float* v56 = &v8[v55];
  __riscv_vse32_v_f32m1(v56, v51, v13);
  size_t v57 = v14 * v9;
  float* v58 = &v8[v57];
  __riscv_vse32_v_f32m1(v58, v52, v13);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_2x4_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 2
//   NR         : 4
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_2x4_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v13);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v13);
  vfloat32m1_t v30;
  v30 = v21;
  vfloat32m1_t v31;
  v31 = v23;
  vfloat32m1_t v32;
  v32 = v26;
  vfloat32m1_t v33;
  v33 = v29;
  for (size_t i34 = v11; i34 < v1; i34 += v12) {
    size_t v35 = i34 * v4;
    size_t v36 = 2;
    float* v37 = &v3[v35];
    vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v36);
    size_t v39 = i34 * v6;
    size_t v40 = v39 + v12;
    size_t v41 = v39 + v13;
    size_t v42 = v39 + v14;
    vfloat32m1_t v43 = v30;
    float v44 = v5[v39];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v38, v36);
    vfloat32m1_t v46 = v31;
    float v47 = v5[v40];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v38, v36);
    vfloat32m1_t v49 = v32;
    float v50 = v5[v41];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v38, v36);
    vfloat32m1_t v52 = v33;
    float v53 = v5[v42];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v38, v36);
    v30 = v45;
    v31 = v48;
    v32 = v51;
    v33 = v54;
  }
  vfloat32m1_t v55 = v30;
  vfloat32m1_t v56 = v31;
  vfloat32m1_t v57 = v32;
  vfloat32m1_t v58 = v33;
  float* v59 = &v8[v11];
  __riscv_vse32_v_f32m1(v59, v55, v13);
  float* v60 = &v8[v9];
  __riscv_vse32_v_f32m1(v60, v56, v13);
  size_t v61 = v13 * v9;
  float* v62 = &v8[v61];
  __riscv_vse32_v_f32m1(v62, v57, v13);
  size_t v63 = v14 * v9;
  float* v64 = &v8[v63];
  __riscv_vse32_v_f32m1(v64, v58, v13);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_2x5_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 2
//   NR         : 5
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_2x5_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v24 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v25;
  v25 = v20;
  vfloat32m1_t v26;
  v26 = v21;
  vfloat32m1_t v27;
  v27 = v22;
  vfloat32m1_t v28;
  v28 = v23;
  vfloat32m1_t v29;
  v29 = v24;
  for (size_t i30 = v11; i30 < v1; i30 += v12) {
    size_t v31 = i30 * v4;
    size_t v32 = 2;
    float* v33 = &v3[v31];
    vfloat32m1_t v34 = __riscv_vle32_v_f32m1(v33, v32);
    size_t v35 = i30 * v6;
    size_t v36 = v35 + v12;
    size_t v37 = v35 + v13;
    size_t v38 = v35 + v14;
    size_t v39 = v35 + v15;
    vfloat32m1_t v40 = v25;
    float v41 = v5[v35];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v34, v32);
    vfloat32m1_t v43 = v26;
    float v44 = v5[v36];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v34, v32);
    vfloat32m1_t v46 = v27;
    float v47 = v5[v37];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v34, v32);
    vfloat32m1_t v49 = v28;
    float v50 = v5[v38];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v34, v32);
    vfloat32m1_t v52 = v29;
    float v53 = v5[v39];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v34, v32);
    v25 = v42;
    v26 = v45;
    v27 = v48;
    v28 = v51;
    v29 = v54;
  }
  vfloat32m1_t v55 = v25;
  vfloat32m1_t v56 = v26;
  vfloat32m1_t v57 = v27;
  vfloat32m1_t v58 = v28;
  vfloat32m1_t v59 = v29;
  float* v60 = &v8[v11];
  __riscv_vse32_v_f32m1(v60, v55, v13);
  float* v61 = &v8[v9];
  __riscv_vse32_v_f32m1(v61, v56, v13);
  size_t v62 = v13 * v9;
  float* v63 = &v8[v62];
  __riscv_vse32_v_f32m1(v63, v57, v13);
  size_t v64 = v14 * v9;
  float* v65 = &v8[v64];
  __riscv_vse32_v_f32m1(v65, v58, v13);
  size_t v66 = v15 * v9;
  float* v67 = &v8[v66];
  __riscv_vse32_v_f32m1(v67, v59, v13);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_2x5_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 2
//   NR         : 5
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_2x5_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v13);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v13);
  size_t v30 = v15 * v9;
  float* v31 = &v8[v30];
  vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v13);
  vfloat32m1_t v33;
  v33 = v21;
  vfloat32m1_t v34;
  v34 = v23;
  vfloat32m1_t v35;
  v35 = v26;
  vfloat32m1_t v36;
  v36 = v29;
  vfloat32m1_t v37;
  v37 = v32;
  for (size_t i38 = v11; i38 < v1; i38 += v12) {
    size_t v39 = i38 * v4;
    size_t v40 = 2;
    float* v41 = &v3[v39];
    vfloat32m1_t v42 = __riscv_vle32_v_f32m1(v41, v40);
    size_t v43 = i38 * v6;
    size_t v44 = v43 + v12;
    size_t v45 = v43 + v13;
    size_t v46 = v43 + v14;
    size_t v47 = v43 + v15;
    vfloat32m1_t v48 = v33;
    float v49 = v5[v43];
    vfloat32m1_t v50 = __riscv_vfmacc_vf_f32m1(v48, v49, v42, v40);
    vfloat32m1_t v51 = v34;
    float v52 = v5[v44];
    vfloat32m1_t v53 = __riscv_vfmacc_vf_f32m1(v51, v52, v42, v40);
    vfloat32m1_t v54 = v35;
    float v55 = v5[v45];
    vfloat32m1_t v56 = __riscv_vfmacc_vf_f32m1(v54, v55, v42, v40);
    vfloat32m1_t v57 = v36;
    float v58 = v5[v46];
    vfloat32m1_t v59 = __riscv_vfmacc_vf_f32m1(v57, v58, v42, v40);
    vfloat32m1_t v60 = v37;
    float v61 = v5[v47];
    vfloat32m1_t v62 = __riscv_vfmacc_vf_f32m1(v60, v61, v42, v40);
    v33 = v50;
    v34 = v53;
    v35 = v56;
    v36 = v59;
    v37 = v62;
  }
  vfloat32m1_t v63 = v33;
  vfloat32m1_t v64 = v34;
  vfloat32m1_t v65 = v35;
  vfloat32m1_t v66 = v36;
  vfloat32m1_t v67 = v37;
  float* v68 = &v8[v11];
  __riscv_vse32_v_f32m1(v68, v63, v13);
  float* v69 = &v8[v9];
  __riscv_vse32_v_f32m1(v69, v64, v13);
  size_t v70 = v13 * v9;
  float* v71 = &v8[v70];
  __riscv_vse32_v_f32m1(v71, v65, v13);
  size_t v72 = v14 * v9;
  float* v73 = &v8[v72];
  __riscv_vse32_v_f32m1(v73, v66, v13);
  size_t v74 = v15 * v9;
  float* v75 = &v8[v74];
  __riscv_vse32_v_f32m1(v75, v67, v13);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_2x6_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 2
//   NR         : 6
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_2x6_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v24 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v25 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v26;
  v26 = v20;
  vfloat32m1_t v27;
  v27 = v21;
  vfloat32m1_t v28;
  v28 = v22;
  vfloat32m1_t v29;
  v29 = v23;
  vfloat32m1_t v30;
  v30 = v24;
  vfloat32m1_t v31;
  v31 = v25;
  for (size_t i32 = v11; i32 < v1; i32 += v12) {
    size_t v33 = i32 * v4;
    size_t v34 = 2;
    float* v35 = &v3[v33];
    vfloat32m1_t v36 = __riscv_vle32_v_f32m1(v35, v34);
    size_t v37 = i32 * v6;
    size_t v38 = v37 + v12;
    size_t v39 = v37 + v13;
    size_t v40 = v37 + v14;
    size_t v41 = v37 + v15;
    size_t v42 = v37 + v16;
    vfloat32m1_t v43 = v26;
    float v44 = v5[v37];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v36, v34);
    vfloat32m1_t v46 = v27;
    float v47 = v5[v38];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v36, v34);
    vfloat32m1_t v49 = v28;
    float v50 = v5[v39];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v36, v34);
    vfloat32m1_t v52 = v29;
    float v53 = v5[v40];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v36, v34);
    vfloat32m1_t v55 = v30;
    float v56 = v5[v41];
    vfloat32m1_t v57 = __riscv_vfmacc_vf_f32m1(v55, v56, v36, v34);
    vfloat32m1_t v58 = v31;
    float v59 = v5[v42];
    vfloat32m1_t v60 = __riscv_vfmacc_vf_f32m1(v58, v59, v36, v34);
    v26 = v45;
    v27 = v48;
    v28 = v51;
    v29 = v54;
    v30 = v57;
    v31 = v60;
  }
  vfloat32m1_t v61 = v26;
  vfloat32m1_t v62 = v27;
  vfloat32m1_t v63 = v28;
  vfloat32m1_t v64 = v29;
  vfloat32m1_t v65 = v30;
  vfloat32m1_t v66 = v31;
  float* v67 = &v8[v11];
  __riscv_vse32_v_f32m1(v67, v61, v13);
  float* v68 = &v8[v9];
  __riscv_vse32_v_f32m1(v68, v62, v13);
  size_t v69 = v13 * v9;
  float* v70 = &v8[v69];
  __riscv_vse32_v_f32m1(v70, v63, v13);
  size_t v71 = v14 * v9;
  float* v72 = &v8[v71];
  __riscv_vse32_v_f32m1(v72, v64, v13);
  size_t v73 = v15 * v9;
  float* v74 = &v8[v73];
  __riscv_vse32_v_f32m1(v74, v65, v13);
  size_t v75 = v16 * v9;
  float* v76 = &v8[v75];
  __riscv_vse32_v_f32m1(v76, v66, v13);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_2x6_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 2
//   NR         : 6
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_2x6_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v13);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v13);
  size_t v30 = v15 * v9;
  float* v31 = &v8[v30];
  vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v13);
  size_t v33 = v16 * v9;
  float* v34 = &v8[v33];
  vfloat32m1_t v35 = __riscv_vle32_v_f32m1(v34, v13);
  vfloat32m1_t v36;
  v36 = v21;
  vfloat32m1_t v37;
  v37 = v23;
  vfloat32m1_t v38;
  v38 = v26;
  vfloat32m1_t v39;
  v39 = v29;
  vfloat32m1_t v40;
  v40 = v32;
  vfloat32m1_t v41;
  v41 = v35;
  for (size_t i42 = v11; i42 < v1; i42 += v12) {
    size_t v43 = i42 * v4;
    size_t v44 = 2;
    float* v45 = &v3[v43];
    vfloat32m1_t v46 = __riscv_vle32_v_f32m1(v45, v44);
    size_t v47 = i42 * v6;
    size_t v48 = v47 + v12;
    size_t v49 = v47 + v13;
    size_t v50 = v47 + v14;
    size_t v51 = v47 + v15;
    size_t v52 = v47 + v16;
    vfloat32m1_t v53 = v36;
    float v54 = v5[v47];
    vfloat32m1_t v55 = __riscv_vfmacc_vf_f32m1(v53, v54, v46, v44);
    vfloat32m1_t v56 = v37;
    float v57 = v5[v48];
    vfloat32m1_t v58 = __riscv_vfmacc_vf_f32m1(v56, v57, v46, v44);
    vfloat32m1_t v59 = v38;
    float v60 = v5[v49];
    vfloat32m1_t v61 = __riscv_vfmacc_vf_f32m1(v59, v60, v46, v44);
    vfloat32m1_t v62 = v39;
    float v63 = v5[v50];
    vfloat32m1_t v64 = __riscv_vfmacc_vf_f32m1(v62, v63, v46, v44);
    vfloat32m1_t v65 = v40;
    float v66 = v5[v51];
    vfloat32m1_t v67 = __riscv_vfmacc_vf_f32m1(v65, v66, v46, v44);
    vfloat32m1_t v68 = v41;
    float v69 = v5[v52];
    vfloat32m1_t v70 = __riscv_vfmacc_vf_f32m1(v68, v69, v46, v44);
    v36 = v55;
    v37 = v58;
    v38 = v61;
    v39 = v64;
    v40 = v67;
    v41 = v70;
  }
  vfloat32m1_t v71 = v36;
  vfloat32m1_t v72 = v37;
  vfloat32m1_t v73 = v38;
  vfloat32m1_t v74 = v39;
  vfloat32m1_t v75 = v40;
  vfloat32m1_t v76 = v41;
  float* v77 = &v8[v11];
  __riscv_vse32_v_f32m1(v77, v71, v13);
  float* v78 = &v8[v9];
  __riscv_vse32_v_f32m1(v78, v72, v13);
  size_t v79 = v13 * v9;
  float* v80 = &v8[v79];
  __riscv_vse32_v_f32m1(v80, v73, v13);
  size_t v81 = v14 * v9;
  float* v82 = &v8[v81];
  __riscv_vse32_v_f32m1(v82, v74, v13);
  size_t v83 = v15 * v9;
  float* v84 = &v8[v83];
  __riscv_vse32_v_f32m1(v84, v75, v13);
  size_t v85 = v16 * v9;
  float* v86 = &v8[v85];
  __riscv_vse32_v_f32m1(v86, v76, v13);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_2x7_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 2
//   NR         : 7
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_2x7_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v24 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v25 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v26 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v27;
  v27 = v20;
  vfloat32m1_t v28;
  v28 = v21;
  vfloat32m1_t v29;
  v29 = v22;
  vfloat32m1_t v30;
  v30 = v23;
  vfloat32m1_t v31;
  v31 = v24;
  vfloat32m1_t v32;
  v32 = v25;
  vfloat32m1_t v33;
  v33 = v26;
  for (size_t i34 = v11; i34 < v1; i34 += v12) {
    size_t v35 = i34 * v4;
    size_t v36 = 2;
    float* v37 = &v3[v35];
    vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v36);
    size_t v39 = i34 * v6;
    size_t v40 = v39 + v12;
    size_t v41 = v39 + v13;
    size_t v42 = v39 + v14;
    size_t v43 = v39 + v15;
    size_t v44 = v39 + v16;
    size_t v45 = v39 + v17;
    vfloat32m1_t v46 = v27;
    float v47 = v5[v39];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v38, v36);
    vfloat32m1_t v49 = v28;
    float v50 = v5[v40];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v38, v36);
    vfloat32m1_t v52 = v29;
    float v53 = v5[v41];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v38, v36);
    vfloat32m1_t v55 = v30;
    float v56 = v5[v42];
    vfloat32m1_t v57 = __riscv_vfmacc_vf_f32m1(v55, v56, v38, v36);
    vfloat32m1_t v58 = v31;
    float v59 = v5[v43];
    vfloat32m1_t v60 = __riscv_vfmacc_vf_f32m1(v58, v59, v38, v36);
    vfloat32m1_t v61 = v32;
    float v62 = v5[v44];
    vfloat32m1_t v63 = __riscv_vfmacc_vf_f32m1(v61, v62, v38, v36);
    vfloat32m1_t v64 = v33;
    float v65 = v5[v45];
    vfloat32m1_t v66 = __riscv_vfmacc_vf_f32m1(v64, v65, v38, v36);
    v27 = v48;
    v28 = v51;
    v29 = v54;
    v30 = v57;
    v31 = v60;
    v32 = v63;
    v33 = v66;
  }
  vfloat32m1_t v67 = v27;
  vfloat32m1_t v68 = v28;
  vfloat32m1_t v69 = v29;
  vfloat32m1_t v70 = v30;
  vfloat32m1_t v71 = v31;
  vfloat32m1_t v72 = v32;
  vfloat32m1_t v73 = v33;
  float* v74 = &v8[v11];
  __riscv_vse32_v_f32m1(v74, v67, v13);
  float* v75 = &v8[v9];
  __riscv_vse32_v_f32m1(v75, v68, v13);
  size_t v76 = v13 * v9;
  float* v77 = &v8[v76];
  __riscv_vse32_v_f32m1(v77, v69, v13);
  size_t v78 = v14 * v9;
  float* v79 = &v8[v78];
  __riscv_vse32_v_f32m1(v79, v70, v13);
  size_t v80 = v15 * v9;
  float* v81 = &v8[v80];
  __riscv_vse32_v_f32m1(v81, v71, v13);
  size_t v82 = v16 * v9;
  float* v83 = &v8[v82];
  __riscv_vse32_v_f32m1(v83, v72, v13);
  size_t v84 = v17 * v9;
  float* v85 = &v8[v84];
  __riscv_vse32_v_f32m1(v85, v73, v13);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_2x7_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 2
//   NR         : 7
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_2x7_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v13);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v13);
  size_t v30 = v15 * v9;
  float* v31 = &v8[v30];
  vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v13);
  size_t v33 = v16 * v9;
  float* v34 = &v8[v33];
  vfloat32m1_t v35 = __riscv_vle32_v_f32m1(v34, v13);
  size_t v36 = v17 * v9;
  float* v37 = &v8[v36];
  vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v13);
  vfloat32m1_t v39;
  v39 = v21;
  vfloat32m1_t v40;
  v40 = v23;
  vfloat32m1_t v41;
  v41 = v26;
  vfloat32m1_t v42;
  v42 = v29;
  vfloat32m1_t v43;
  v43 = v32;
  vfloat32m1_t v44;
  v44 = v35;
  vfloat32m1_t v45;
  v45 = v38;
  for (size_t i46 = v11; i46 < v1; i46 += v12) {
    size_t v47 = i46 * v4;
    size_t v48 = 2;
    float* v49 = &v3[v47];
    vfloat32m1_t v50 = __riscv_vle32_v_f32m1(v49, v48);
    size_t v51 = i46 * v6;
    size_t v52 = v51 + v12;
    size_t v53 = v51 + v13;
    size_t v54 = v51 + v14;
    size_t v55 = v51 + v15;
    size_t v56 = v51 + v16;
    size_t v57 = v51 + v17;
    vfloat32m1_t v58 = v39;
    float v59 = v5[v51];
    vfloat32m1_t v60 = __riscv_vfmacc_vf_f32m1(v58, v59, v50, v48);
    vfloat32m1_t v61 = v40;
    float v62 = v5[v52];
    vfloat32m1_t v63 = __riscv_vfmacc_vf_f32m1(v61, v62, v50, v48);
    vfloat32m1_t v64 = v41;
    float v65 = v5[v53];
    vfloat32m1_t v66 = __riscv_vfmacc_vf_f32m1(v64, v65, v50, v48);
    vfloat32m1_t v67 = v42;
    float v68 = v5[v54];
    vfloat32m1_t v69 = __riscv_vfmacc_vf_f32m1(v67, v68, v50, v48);
    vfloat32m1_t v70 = v43;
    float v71 = v5[v55];
    vfloat32m1_t v72 = __riscv_vfmacc_vf_f32m1(v70, v71, v50, v48);
    vfloat32m1_t v73 = v44;
    float v74 = v5[v56];
    vfloat32m1_t v75 = __riscv_vfmacc_vf_f32m1(v73, v74, v50, v48);
    vfloat32m1_t v76 = v45;
    float v77 = v5[v57];
    vfloat32m1_t v78 = __riscv_vfmacc_vf_f32m1(v76, v77, v50, v48);
    v39 = v60;
    v40 = v63;
    v41 = v66;
    v42 = v69;
    v43 = v72;
    v44 = v75;
    v45 = v78;
  }
  vfloat32m1_t v79 = v39;
  vfloat32m1_t v80 = v40;
  vfloat32m1_t v81 = v41;
  vfloat32m1_t v82 = v42;
  vfloat32m1_t v83 = v43;
  vfloat32m1_t v84 = v44;
  vfloat32m1_t v85 = v45;
  float* v86 = &v8[v11];
  __riscv_vse32_v_f32m1(v86, v79, v13);
  float* v87 = &v8[v9];
  __riscv_vse32_v_f32m1(v87, v80, v13);
  size_t v88 = v13 * v9;
  float* v89 = &v8[v88];
  __riscv_vse32_v_f32m1(v89, v81, v13);
  size_t v90 = v14 * v9;
  float* v91 = &v8[v90];
  __riscv_vse32_v_f32m1(v91, v82, v13);
  size_t v92 = v15 * v9;
  float* v93 = &v8[v92];
  __riscv_vse32_v_f32m1(v93, v83, v13);
  size_t v94 = v16 * v9;
  float* v95 = &v8[v94];
  __riscv_vse32_v_f32m1(v95, v84, v13);
  size_t v96 = v17 * v9;
  float* v97 = &v8[v96];
  __riscv_vse32_v_f32m1(v97, v85, v13);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_2x8_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 2
//   NR         : 8
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_2x8_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v24 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v25 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v26 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v27 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v28;
  v28 = v20;
  vfloat32m1_t v29;
  v29 = v21;
  vfloat32m1_t v30;
  v30 = v22;
  vfloat32m1_t v31;
  v31 = v23;
  vfloat32m1_t v32;
  v32 = v24;
  vfloat32m1_t v33;
  v33 = v25;
  vfloat32m1_t v34;
  v34 = v26;
  vfloat32m1_t v35;
  v35 = v27;
  for (size_t i36 = v11; i36 < v1; i36 += v12) {
    size_t v37 = i36 * v4;
    size_t v38 = 2;
    float* v39 = &v3[v37];
    vfloat32m1_t v40 = __riscv_vle32_v_f32m1(v39, v38);
    size_t v41 = i36 * v6;
    size_t v42 = v41 + v12;
    size_t v43 = v41 + v13;
    size_t v44 = v41 + v14;
    size_t v45 = v41 + v15;
    size_t v46 = v41 + v16;
    size_t v47 = v41 + v17;
    size_t v48 = v41 + v18;
    vfloat32m1_t v49 = v28;
    float v50 = v5[v41];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v40, v38);
    vfloat32m1_t v52 = v29;
    float v53 = v5[v42];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v40, v38);
    vfloat32m1_t v55 = v30;
    float v56 = v5[v43];
    vfloat32m1_t v57 = __riscv_vfmacc_vf_f32m1(v55, v56, v40, v38);
    vfloat32m1_t v58 = v31;
    float v59 = v5[v44];
    vfloat32m1_t v60 = __riscv_vfmacc_vf_f32m1(v58, v59, v40, v38);
    vfloat32m1_t v61 = v32;
    float v62 = v5[v45];
    vfloat32m1_t v63 = __riscv_vfmacc_vf_f32m1(v61, v62, v40, v38);
    vfloat32m1_t v64 = v33;
    float v65 = v5[v46];
    vfloat32m1_t v66 = __riscv_vfmacc_vf_f32m1(v64, v65, v40, v38);
    vfloat32m1_t v67 = v34;
    float v68 = v5[v47];
    vfloat32m1_t v69 = __riscv_vfmacc_vf_f32m1(v67, v68, v40, v38);
    vfloat32m1_t v70 = v35;
    float v71 = v5[v48];
    vfloat32m1_t v72 = __riscv_vfmacc_vf_f32m1(v70, v71, v40, v38);
    v28 = v51;
    v29 = v54;
    v30 = v57;
    v31 = v60;
    v32 = v63;
    v33 = v66;
    v34 = v69;
    v35 = v72;
  }
  vfloat32m1_t v73 = v28;
  vfloat32m1_t v74 = v29;
  vfloat32m1_t v75 = v30;
  vfloat32m1_t v76 = v31;
  vfloat32m1_t v77 = v32;
  vfloat32m1_t v78 = v33;
  vfloat32m1_t v79 = v34;
  vfloat32m1_t v80 = v35;
  float* v81 = &v8[v11];
  __riscv_vse32_v_f32m1(v81, v73, v13);
  float* v82 = &v8[v9];
  __riscv_vse32_v_f32m1(v82, v74, v13);
  size_t v83 = v13 * v9;
  float* v84 = &v8[v83];
  __riscv_vse32_v_f32m1(v84, v75, v13);
  size_t v85 = v14 * v9;
  float* v86 = &v8[v85];
  __riscv_vse32_v_f32m1(v86, v76, v13);
  size_t v87 = v15 * v9;
  float* v88 = &v8[v87];
  __riscv_vse32_v_f32m1(v88, v77, v13);
  size_t v89 = v16 * v9;
  float* v90 = &v8[v89];
  __riscv_vse32_v_f32m1(v90, v78, v13);
  size_t v91 = v17 * v9;
  float* v92 = &v8[v91];
  __riscv_vse32_v_f32m1(v92, v79, v13);
  size_t v93 = v18 * v9;
  float* v94 = &v8[v93];
  __riscv_vse32_v_f32m1(v94, v80, v13);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_2x8_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 2
//   NR         : 8
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_2x8_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v13);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v13);
  size_t v30 = v15 * v9;
  float* v31 = &v8[v30];
  vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v13);
  size_t v33 = v16 * v9;
  float* v34 = &v8[v33];
  vfloat32m1_t v35 = __riscv_vle32_v_f32m1(v34, v13);
  size_t v36 = v17 * v9;
  float* v37 = &v8[v36];
  vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v13);
  size_t v39 = v18 * v9;
  float* v40 = &v8[v39];
  vfloat32m1_t v41 = __riscv_vle32_v_f32m1(v40, v13);
  vfloat32m1_t v42;
  v42 = v21;
  vfloat32m1_t v43;
  v43 = v23;
  vfloat32m1_t v44;
  v44 = v26;
  vfloat32m1_t v45;
  v45 = v29;
  vfloat32m1_t v46;
  v46 = v32;
  vfloat32m1_t v47;
  v47 = v35;
  vfloat32m1_t v48;
  v48 = v38;
  vfloat32m1_t v49;
  v49 = v41;
  for (size_t i50 = v11; i50 < v1; i50 += v12) {
    size_t v51 = i50 * v4;
    size_t v52 = 2;
    float* v53 = &v3[v51];
    vfloat32m1_t v54 = __riscv_vle32_v_f32m1(v53, v52);
    size_t v55 = i50 * v6;
    size_t v56 = v55 + v12;
    size_t v57 = v55 + v13;
    size_t v58 = v55 + v14;
    size_t v59 = v55 + v15;
    size_t v60 = v55 + v16;
    size_t v61 = v55 + v17;
    size_t v62 = v55 + v18;
    vfloat32m1_t v63 = v42;
    float v64 = v5[v55];
    vfloat32m1_t v65 = __riscv_vfmacc_vf_f32m1(v63, v64, v54, v52);
    vfloat32m1_t v66 = v43;
    float v67 = v5[v56];
    vfloat32m1_t v68 = __riscv_vfmacc_vf_f32m1(v66, v67, v54, v52);
    vfloat32m1_t v69 = v44;
    float v70 = v5[v57];
    vfloat32m1_t v71 = __riscv_vfmacc_vf_f32m1(v69, v70, v54, v52);
    vfloat32m1_t v72 = v45;
    float v73 = v5[v58];
    vfloat32m1_t v74 = __riscv_vfmacc_vf_f32m1(v72, v73, v54, v52);
    vfloat32m1_t v75 = v46;
    float v76 = v5[v59];
    vfloat32m1_t v77 = __riscv_vfmacc_vf_f32m1(v75, v76, v54, v52);
    vfloat32m1_t v78 = v47;
    float v79 = v5[v60];
    vfloat32m1_t v80 = __riscv_vfmacc_vf_f32m1(v78, v79, v54, v52);
    vfloat32m1_t v81 = v48;
    float v82 = v5[v61];
    vfloat32m1_t v83 = __riscv_vfmacc_vf_f32m1(v81, v82, v54, v52);
    vfloat32m1_t v84 = v49;
    float v85 = v5[v62];
    vfloat32m1_t v86 = __riscv_vfmacc_vf_f32m1(v84, v85, v54, v52);
    v42 = v65;
    v43 = v68;
    v44 = v71;
    v45 = v74;
    v46 = v77;
    v47 = v80;
    v48 = v83;
    v49 = v86;
  }
  vfloat32m1_t v87 = v42;
  vfloat32m1_t v88 = v43;
  vfloat32m1_t v89 = v44;
  vfloat32m1_t v90 = v45;
  vfloat32m1_t v91 = v46;
  vfloat32m1_t v92 = v47;
  vfloat32m1_t v93 = v48;
  vfloat32m1_t v94 = v49;
  float* v95 = &v8[v11];
  __riscv_vse32_v_f32m1(v95, v87, v13);
  float* v96 = &v8[v9];
  __riscv_vse32_v_f32m1(v96, v88, v13);
  size_t v97 = v13 * v9;
  float* v98 = &v8[v97];
  __riscv_vse32_v_f32m1(v98, v89, v13);
  size_t v99 = v14 * v9;
  float* v100 = &v8[v99];
  __riscv_vse32_v_f32m1(v100, v90, v13);
  size_t v101 = v15 * v9;
  float* v102 = &v8[v101];
  __riscv_vse32_v_f32m1(v102, v91, v13);
  size_t v103 = v16 * v9;
  float* v104 = &v8[v103];
  __riscv_vse32_v_f32m1(v104, v92, v13);
  size_t v105 = v17 * v9;
  float* v106 = &v8[v105];
  __riscv_vse32_v_f32m1(v106, v93, v13);
  size_t v107 = v18 * v9;
  float* v108 = &v8[v107];
  __riscv_vse32_v_f32m1(v108, v94, v13);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_3x1_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 3
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_3x1_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v21;
  v21 = v20;
  for (size_t i22 = v11; i22 < v1; i22 += v12) {
    size_t v23 = i22 * v4;
    size_t v24 = 3;
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
  __riscv_vse32_v_f32m1(v32, v31, v14);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_3x1_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 3
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_3x1_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v14);
  vfloat32m1_t v22;
  v22 = v21;
  for (size_t i23 = v11; i23 < v1; i23 += v12) {
    size_t v24 = i23 * v4;
    size_t v25 = 3;
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
  __riscv_vse32_v_f32m1(v33, v32, v14);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_3x2_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 3
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_3x2_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v22;
  v22 = v20;
  vfloat32m1_t v23;
  v23 = v21;
  for (size_t i24 = v11; i24 < v1; i24 += v12) {
    size_t v25 = i24 * v4;
    size_t v26 = 3;
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
  __riscv_vse32_v_f32m1(v39, v37, v14);
  float* v40 = &v8[v9];
  __riscv_vse32_v_f32m1(v40, v38, v14);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_3x2_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 3
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_3x2_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v14);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v14);
  vfloat32m1_t v24;
  v24 = v21;
  vfloat32m1_t v25;
  v25 = v23;
  for (size_t i26 = v11; i26 < v1; i26 += v12) {
    size_t v27 = i26 * v4;
    size_t v28 = 3;
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
  __riscv_vse32_v_f32m1(v41, v39, v14);
  float* v42 = &v8[v9];
  __riscv_vse32_v_f32m1(v42, v40, v14);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_3x3_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 3
//   NR         : 3
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_3x3_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v23;
  v23 = v20;
  vfloat32m1_t v24;
  v24 = v21;
  vfloat32m1_t v25;
  v25 = v22;
  for (size_t i26 = v11; i26 < v1; i26 += v12) {
    size_t v27 = i26 * v4;
    size_t v28 = 3;
    float* v29 = &v3[v27];
    vfloat32m1_t v30 = __riscv_vle32_v_f32m1(v29, v28);
    size_t v31 = i26 * v6;
    size_t v32 = v31 + v12;
    size_t v33 = v31 + v13;
    vfloat32m1_t v34 = v23;
    float v35 = v5[v31];
    vfloat32m1_t v36 = __riscv_vfmacc_vf_f32m1(v34, v35, v30, v28);
    vfloat32m1_t v37 = v24;
    float v38 = v5[v32];
    vfloat32m1_t v39 = __riscv_vfmacc_vf_f32m1(v37, v38, v30, v28);
    vfloat32m1_t v40 = v25;
    float v41 = v5[v33];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v30, v28);
    v23 = v36;
    v24 = v39;
    v25 = v42;
  }
  vfloat32m1_t v43 = v23;
  vfloat32m1_t v44 = v24;
  vfloat32m1_t v45 = v25;
  float* v46 = &v8[v11];
  __riscv_vse32_v_f32m1(v46, v43, v14);
  float* v47 = &v8[v9];
  __riscv_vse32_v_f32m1(v47, v44, v14);
  size_t v48 = v13 * v9;
  float* v49 = &v8[v48];
  __riscv_vse32_v_f32m1(v49, v45, v14);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_3x3_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 3
//   NR         : 3
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_3x3_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v14);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v14);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v14);
  vfloat32m1_t v27;
  v27 = v21;
  vfloat32m1_t v28;
  v28 = v23;
  vfloat32m1_t v29;
  v29 = v26;
  for (size_t i30 = v11; i30 < v1; i30 += v12) {
    size_t v31 = i30 * v4;
    size_t v32 = 3;
    float* v33 = &v3[v31];
    vfloat32m1_t v34 = __riscv_vle32_v_f32m1(v33, v32);
    size_t v35 = i30 * v6;
    size_t v36 = v35 + v12;
    size_t v37 = v35 + v13;
    vfloat32m1_t v38 = v27;
    float v39 = v5[v35];
    vfloat32m1_t v40 = __riscv_vfmacc_vf_f32m1(v38, v39, v34, v32);
    vfloat32m1_t v41 = v28;
    float v42 = v5[v36];
    vfloat32m1_t v43 = __riscv_vfmacc_vf_f32m1(v41, v42, v34, v32);
    vfloat32m1_t v44 = v29;
    float v45 = v5[v37];
    vfloat32m1_t v46 = __riscv_vfmacc_vf_f32m1(v44, v45, v34, v32);
    v27 = v40;
    v28 = v43;
    v29 = v46;
  }
  vfloat32m1_t v47 = v27;
  vfloat32m1_t v48 = v28;
  vfloat32m1_t v49 = v29;
  float* v50 = &v8[v11];
  __riscv_vse32_v_f32m1(v50, v47, v14);
  float* v51 = &v8[v9];
  __riscv_vse32_v_f32m1(v51, v48, v14);
  size_t v52 = v13 * v9;
  float* v53 = &v8[v52];
  __riscv_vse32_v_f32m1(v53, v49, v14);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_3x4_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 3
//   NR         : 4
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_3x4_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v24;
  v24 = v20;
  vfloat32m1_t v25;
  v25 = v21;
  vfloat32m1_t v26;
  v26 = v22;
  vfloat32m1_t v27;
  v27 = v23;
  for (size_t i28 = v11; i28 < v1; i28 += v12) {
    size_t v29 = i28 * v4;
    size_t v30 = 3;
    float* v31 = &v3[v29];
    vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v30);
    size_t v33 = i28 * v6;
    size_t v34 = v33 + v12;
    size_t v35 = v33 + v13;
    size_t v36 = v33 + v14;
    vfloat32m1_t v37 = v24;
    float v38 = v5[v33];
    vfloat32m1_t v39 = __riscv_vfmacc_vf_f32m1(v37, v38, v32, v30);
    vfloat32m1_t v40 = v25;
    float v41 = v5[v34];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v32, v30);
    vfloat32m1_t v43 = v26;
    float v44 = v5[v35];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v32, v30);
    vfloat32m1_t v46 = v27;
    float v47 = v5[v36];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v32, v30);
    v24 = v39;
    v25 = v42;
    v26 = v45;
    v27 = v48;
  }
  vfloat32m1_t v49 = v24;
  vfloat32m1_t v50 = v25;
  vfloat32m1_t v51 = v26;
  vfloat32m1_t v52 = v27;
  float* v53 = &v8[v11];
  __riscv_vse32_v_f32m1(v53, v49, v14);
  float* v54 = &v8[v9];
  __riscv_vse32_v_f32m1(v54, v50, v14);
  size_t v55 = v13 * v9;
  float* v56 = &v8[v55];
  __riscv_vse32_v_f32m1(v56, v51, v14);
  size_t v57 = v14 * v9;
  float* v58 = &v8[v57];
  __riscv_vse32_v_f32m1(v58, v52, v14);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_3x4_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 3
//   NR         : 4
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_3x4_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v14);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v14);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v14);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v14);
  vfloat32m1_t v30;
  v30 = v21;
  vfloat32m1_t v31;
  v31 = v23;
  vfloat32m1_t v32;
  v32 = v26;
  vfloat32m1_t v33;
  v33 = v29;
  for (size_t i34 = v11; i34 < v1; i34 += v12) {
    size_t v35 = i34 * v4;
    size_t v36 = 3;
    float* v37 = &v3[v35];
    vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v36);
    size_t v39 = i34 * v6;
    size_t v40 = v39 + v12;
    size_t v41 = v39 + v13;
    size_t v42 = v39 + v14;
    vfloat32m1_t v43 = v30;
    float v44 = v5[v39];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v38, v36);
    vfloat32m1_t v46 = v31;
    float v47 = v5[v40];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v38, v36);
    vfloat32m1_t v49 = v32;
    float v50 = v5[v41];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v38, v36);
    vfloat32m1_t v52 = v33;
    float v53 = v5[v42];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v38, v36);
    v30 = v45;
    v31 = v48;
    v32 = v51;
    v33 = v54;
  }
  vfloat32m1_t v55 = v30;
  vfloat32m1_t v56 = v31;
  vfloat32m1_t v57 = v32;
  vfloat32m1_t v58 = v33;
  float* v59 = &v8[v11];
  __riscv_vse32_v_f32m1(v59, v55, v14);
  float* v60 = &v8[v9];
  __riscv_vse32_v_f32m1(v60, v56, v14);
  size_t v61 = v13 * v9;
  float* v62 = &v8[v61];
  __riscv_vse32_v_f32m1(v62, v57, v14);
  size_t v63 = v14 * v9;
  float* v64 = &v8[v63];
  __riscv_vse32_v_f32m1(v64, v58, v14);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_3x5_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 3
//   NR         : 5
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_3x5_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v24 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v25;
  v25 = v20;
  vfloat32m1_t v26;
  v26 = v21;
  vfloat32m1_t v27;
  v27 = v22;
  vfloat32m1_t v28;
  v28 = v23;
  vfloat32m1_t v29;
  v29 = v24;
  for (size_t i30 = v11; i30 < v1; i30 += v12) {
    size_t v31 = i30 * v4;
    size_t v32 = 3;
    float* v33 = &v3[v31];
    vfloat32m1_t v34 = __riscv_vle32_v_f32m1(v33, v32);
    size_t v35 = i30 * v6;
    size_t v36 = v35 + v12;
    size_t v37 = v35 + v13;
    size_t v38 = v35 + v14;
    size_t v39 = v35 + v15;
    vfloat32m1_t v40 = v25;
    float v41 = v5[v35];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v34, v32);
    vfloat32m1_t v43 = v26;
    float v44 = v5[v36];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v34, v32);
    vfloat32m1_t v46 = v27;
    float v47 = v5[v37];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v34, v32);
    vfloat32m1_t v49 = v28;
    float v50 = v5[v38];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v34, v32);
    vfloat32m1_t v52 = v29;
    float v53 = v5[v39];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v34, v32);
    v25 = v42;
    v26 = v45;
    v27 = v48;
    v28 = v51;
    v29 = v54;
  }
  vfloat32m1_t v55 = v25;
  vfloat32m1_t v56 = v26;
  vfloat32m1_t v57 = v27;
  vfloat32m1_t v58 = v28;
  vfloat32m1_t v59 = v29;
  float* v60 = &v8[v11];
  __riscv_vse32_v_f32m1(v60, v55, v14);
  float* v61 = &v8[v9];
  __riscv_vse32_v_f32m1(v61, v56, v14);
  size_t v62 = v13 * v9;
  float* v63 = &v8[v62];
  __riscv_vse32_v_f32m1(v63, v57, v14);
  size_t v64 = v14 * v9;
  float* v65 = &v8[v64];
  __riscv_vse32_v_f32m1(v65, v58, v14);
  size_t v66 = v15 * v9;
  float* v67 = &v8[v66];
  __riscv_vse32_v_f32m1(v67, v59, v14);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_3x5_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 3
//   NR         : 5
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_3x5_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v14);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v14);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v14);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v14);
  size_t v30 = v15 * v9;
  float* v31 = &v8[v30];
  vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v14);
  vfloat32m1_t v33;
  v33 = v21;
  vfloat32m1_t v34;
  v34 = v23;
  vfloat32m1_t v35;
  v35 = v26;
  vfloat32m1_t v36;
  v36 = v29;
  vfloat32m1_t v37;
  v37 = v32;
  for (size_t i38 = v11; i38 < v1; i38 += v12) {
    size_t v39 = i38 * v4;
    size_t v40 = 3;
    float* v41 = &v3[v39];
    vfloat32m1_t v42 = __riscv_vle32_v_f32m1(v41, v40);
    size_t v43 = i38 * v6;
    size_t v44 = v43 + v12;
    size_t v45 = v43 + v13;
    size_t v46 = v43 + v14;
    size_t v47 = v43 + v15;
    vfloat32m1_t v48 = v33;
    float v49 = v5[v43];
    vfloat32m1_t v50 = __riscv_vfmacc_vf_f32m1(v48, v49, v42, v40);
    vfloat32m1_t v51 = v34;
    float v52 = v5[v44];
    vfloat32m1_t v53 = __riscv_vfmacc_vf_f32m1(v51, v52, v42, v40);
    vfloat32m1_t v54 = v35;
    float v55 = v5[v45];
    vfloat32m1_t v56 = __riscv_vfmacc_vf_f32m1(v54, v55, v42, v40);
    vfloat32m1_t v57 = v36;
    float v58 = v5[v46];
    vfloat32m1_t v59 = __riscv_vfmacc_vf_f32m1(v57, v58, v42, v40);
    vfloat32m1_t v60 = v37;
    float v61 = v5[v47];
    vfloat32m1_t v62 = __riscv_vfmacc_vf_f32m1(v60, v61, v42, v40);
    v33 = v50;
    v34 = v53;
    v35 = v56;
    v36 = v59;
    v37 = v62;
  }
  vfloat32m1_t v63 = v33;
  vfloat32m1_t v64 = v34;
  vfloat32m1_t v65 = v35;
  vfloat32m1_t v66 = v36;
  vfloat32m1_t v67 = v37;
  float* v68 = &v8[v11];
  __riscv_vse32_v_f32m1(v68, v63, v14);
  float* v69 = &v8[v9];
  __riscv_vse32_v_f32m1(v69, v64, v14);
  size_t v70 = v13 * v9;
  float* v71 = &v8[v70];
  __riscv_vse32_v_f32m1(v71, v65, v14);
  size_t v72 = v14 * v9;
  float* v73 = &v8[v72];
  __riscv_vse32_v_f32m1(v73, v66, v14);
  size_t v74 = v15 * v9;
  float* v75 = &v8[v74];
  __riscv_vse32_v_f32m1(v75, v67, v14);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_3x6_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 3
//   NR         : 6
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_3x6_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v24 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v25 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v26;
  v26 = v20;
  vfloat32m1_t v27;
  v27 = v21;
  vfloat32m1_t v28;
  v28 = v22;
  vfloat32m1_t v29;
  v29 = v23;
  vfloat32m1_t v30;
  v30 = v24;
  vfloat32m1_t v31;
  v31 = v25;
  for (size_t i32 = v11; i32 < v1; i32 += v12) {
    size_t v33 = i32 * v4;
    size_t v34 = 3;
    float* v35 = &v3[v33];
    vfloat32m1_t v36 = __riscv_vle32_v_f32m1(v35, v34);
    size_t v37 = i32 * v6;
    size_t v38 = v37 + v12;
    size_t v39 = v37 + v13;
    size_t v40 = v37 + v14;
    size_t v41 = v37 + v15;
    size_t v42 = v37 + v16;
    vfloat32m1_t v43 = v26;
    float v44 = v5[v37];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v36, v34);
    vfloat32m1_t v46 = v27;
    float v47 = v5[v38];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v36, v34);
    vfloat32m1_t v49 = v28;
    float v50 = v5[v39];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v36, v34);
    vfloat32m1_t v52 = v29;
    float v53 = v5[v40];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v36, v34);
    vfloat32m1_t v55 = v30;
    float v56 = v5[v41];
    vfloat32m1_t v57 = __riscv_vfmacc_vf_f32m1(v55, v56, v36, v34);
    vfloat32m1_t v58 = v31;
    float v59 = v5[v42];
    vfloat32m1_t v60 = __riscv_vfmacc_vf_f32m1(v58, v59, v36, v34);
    v26 = v45;
    v27 = v48;
    v28 = v51;
    v29 = v54;
    v30 = v57;
    v31 = v60;
  }
  vfloat32m1_t v61 = v26;
  vfloat32m1_t v62 = v27;
  vfloat32m1_t v63 = v28;
  vfloat32m1_t v64 = v29;
  vfloat32m1_t v65 = v30;
  vfloat32m1_t v66 = v31;
  float* v67 = &v8[v11];
  __riscv_vse32_v_f32m1(v67, v61, v14);
  float* v68 = &v8[v9];
  __riscv_vse32_v_f32m1(v68, v62, v14);
  size_t v69 = v13 * v9;
  float* v70 = &v8[v69];
  __riscv_vse32_v_f32m1(v70, v63, v14);
  size_t v71 = v14 * v9;
  float* v72 = &v8[v71];
  __riscv_vse32_v_f32m1(v72, v64, v14);
  size_t v73 = v15 * v9;
  float* v74 = &v8[v73];
  __riscv_vse32_v_f32m1(v74, v65, v14);
  size_t v75 = v16 * v9;
  float* v76 = &v8[v75];
  __riscv_vse32_v_f32m1(v76, v66, v14);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_3x6_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 3
//   NR         : 6
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_3x6_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v14);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v14);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v14);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v14);
  size_t v30 = v15 * v9;
  float* v31 = &v8[v30];
  vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v14);
  size_t v33 = v16 * v9;
  float* v34 = &v8[v33];
  vfloat32m1_t v35 = __riscv_vle32_v_f32m1(v34, v14);
  vfloat32m1_t v36;
  v36 = v21;
  vfloat32m1_t v37;
  v37 = v23;
  vfloat32m1_t v38;
  v38 = v26;
  vfloat32m1_t v39;
  v39 = v29;
  vfloat32m1_t v40;
  v40 = v32;
  vfloat32m1_t v41;
  v41 = v35;
  for (size_t i42 = v11; i42 < v1; i42 += v12) {
    size_t v43 = i42 * v4;
    size_t v44 = 3;
    float* v45 = &v3[v43];
    vfloat32m1_t v46 = __riscv_vle32_v_f32m1(v45, v44);
    size_t v47 = i42 * v6;
    size_t v48 = v47 + v12;
    size_t v49 = v47 + v13;
    size_t v50 = v47 + v14;
    size_t v51 = v47 + v15;
    size_t v52 = v47 + v16;
    vfloat32m1_t v53 = v36;
    float v54 = v5[v47];
    vfloat32m1_t v55 = __riscv_vfmacc_vf_f32m1(v53, v54, v46, v44);
    vfloat32m1_t v56 = v37;
    float v57 = v5[v48];
    vfloat32m1_t v58 = __riscv_vfmacc_vf_f32m1(v56, v57, v46, v44);
    vfloat32m1_t v59 = v38;
    float v60 = v5[v49];
    vfloat32m1_t v61 = __riscv_vfmacc_vf_f32m1(v59, v60, v46, v44);
    vfloat32m1_t v62 = v39;
    float v63 = v5[v50];
    vfloat32m1_t v64 = __riscv_vfmacc_vf_f32m1(v62, v63, v46, v44);
    vfloat32m1_t v65 = v40;
    float v66 = v5[v51];
    vfloat32m1_t v67 = __riscv_vfmacc_vf_f32m1(v65, v66, v46, v44);
    vfloat32m1_t v68 = v41;
    float v69 = v5[v52];
    vfloat32m1_t v70 = __riscv_vfmacc_vf_f32m1(v68, v69, v46, v44);
    v36 = v55;
    v37 = v58;
    v38 = v61;
    v39 = v64;
    v40 = v67;
    v41 = v70;
  }
  vfloat32m1_t v71 = v36;
  vfloat32m1_t v72 = v37;
  vfloat32m1_t v73 = v38;
  vfloat32m1_t v74 = v39;
  vfloat32m1_t v75 = v40;
  vfloat32m1_t v76 = v41;
  float* v77 = &v8[v11];
  __riscv_vse32_v_f32m1(v77, v71, v14);
  float* v78 = &v8[v9];
  __riscv_vse32_v_f32m1(v78, v72, v14);
  size_t v79 = v13 * v9;
  float* v80 = &v8[v79];
  __riscv_vse32_v_f32m1(v80, v73, v14);
  size_t v81 = v14 * v9;
  float* v82 = &v8[v81];
  __riscv_vse32_v_f32m1(v82, v74, v14);
  size_t v83 = v15 * v9;
  float* v84 = &v8[v83];
  __riscv_vse32_v_f32m1(v84, v75, v14);
  size_t v85 = v16 * v9;
  float* v86 = &v8[v85];
  __riscv_vse32_v_f32m1(v86, v76, v14);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_3x7_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 3
//   NR         : 7
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_3x7_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v24 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v25 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v26 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v27;
  v27 = v20;
  vfloat32m1_t v28;
  v28 = v21;
  vfloat32m1_t v29;
  v29 = v22;
  vfloat32m1_t v30;
  v30 = v23;
  vfloat32m1_t v31;
  v31 = v24;
  vfloat32m1_t v32;
  v32 = v25;
  vfloat32m1_t v33;
  v33 = v26;
  for (size_t i34 = v11; i34 < v1; i34 += v12) {
    size_t v35 = i34 * v4;
    size_t v36 = 3;
    float* v37 = &v3[v35];
    vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v36);
    size_t v39 = i34 * v6;
    size_t v40 = v39 + v12;
    size_t v41 = v39 + v13;
    size_t v42 = v39 + v14;
    size_t v43 = v39 + v15;
    size_t v44 = v39 + v16;
    size_t v45 = v39 + v17;
    vfloat32m1_t v46 = v27;
    float v47 = v5[v39];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v38, v36);
    vfloat32m1_t v49 = v28;
    float v50 = v5[v40];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v38, v36);
    vfloat32m1_t v52 = v29;
    float v53 = v5[v41];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v38, v36);
    vfloat32m1_t v55 = v30;
    float v56 = v5[v42];
    vfloat32m1_t v57 = __riscv_vfmacc_vf_f32m1(v55, v56, v38, v36);
    vfloat32m1_t v58 = v31;
    float v59 = v5[v43];
    vfloat32m1_t v60 = __riscv_vfmacc_vf_f32m1(v58, v59, v38, v36);
    vfloat32m1_t v61 = v32;
    float v62 = v5[v44];
    vfloat32m1_t v63 = __riscv_vfmacc_vf_f32m1(v61, v62, v38, v36);
    vfloat32m1_t v64 = v33;
    float v65 = v5[v45];
    vfloat32m1_t v66 = __riscv_vfmacc_vf_f32m1(v64, v65, v38, v36);
    v27 = v48;
    v28 = v51;
    v29 = v54;
    v30 = v57;
    v31 = v60;
    v32 = v63;
    v33 = v66;
  }
  vfloat32m1_t v67 = v27;
  vfloat32m1_t v68 = v28;
  vfloat32m1_t v69 = v29;
  vfloat32m1_t v70 = v30;
  vfloat32m1_t v71 = v31;
  vfloat32m1_t v72 = v32;
  vfloat32m1_t v73 = v33;
  float* v74 = &v8[v11];
  __riscv_vse32_v_f32m1(v74, v67, v14);
  float* v75 = &v8[v9];
  __riscv_vse32_v_f32m1(v75, v68, v14);
  size_t v76 = v13 * v9;
  float* v77 = &v8[v76];
  __riscv_vse32_v_f32m1(v77, v69, v14);
  size_t v78 = v14 * v9;
  float* v79 = &v8[v78];
  __riscv_vse32_v_f32m1(v79, v70, v14);
  size_t v80 = v15 * v9;
  float* v81 = &v8[v80];
  __riscv_vse32_v_f32m1(v81, v71, v14);
  size_t v82 = v16 * v9;
  float* v83 = &v8[v82];
  __riscv_vse32_v_f32m1(v83, v72, v14);
  size_t v84 = v17 * v9;
  float* v85 = &v8[v84];
  __riscv_vse32_v_f32m1(v85, v73, v14);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_3x7_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 3
//   NR         : 7
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_3x7_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v14);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v14);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v14);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v14);
  size_t v30 = v15 * v9;
  float* v31 = &v8[v30];
  vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v14);
  size_t v33 = v16 * v9;
  float* v34 = &v8[v33];
  vfloat32m1_t v35 = __riscv_vle32_v_f32m1(v34, v14);
  size_t v36 = v17 * v9;
  float* v37 = &v8[v36];
  vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v14);
  vfloat32m1_t v39;
  v39 = v21;
  vfloat32m1_t v40;
  v40 = v23;
  vfloat32m1_t v41;
  v41 = v26;
  vfloat32m1_t v42;
  v42 = v29;
  vfloat32m1_t v43;
  v43 = v32;
  vfloat32m1_t v44;
  v44 = v35;
  vfloat32m1_t v45;
  v45 = v38;
  for (size_t i46 = v11; i46 < v1; i46 += v12) {
    size_t v47 = i46 * v4;
    size_t v48 = 3;
    float* v49 = &v3[v47];
    vfloat32m1_t v50 = __riscv_vle32_v_f32m1(v49, v48);
    size_t v51 = i46 * v6;
    size_t v52 = v51 + v12;
    size_t v53 = v51 + v13;
    size_t v54 = v51 + v14;
    size_t v55 = v51 + v15;
    size_t v56 = v51 + v16;
    size_t v57 = v51 + v17;
    vfloat32m1_t v58 = v39;
    float v59 = v5[v51];
    vfloat32m1_t v60 = __riscv_vfmacc_vf_f32m1(v58, v59, v50, v48);
    vfloat32m1_t v61 = v40;
    float v62 = v5[v52];
    vfloat32m1_t v63 = __riscv_vfmacc_vf_f32m1(v61, v62, v50, v48);
    vfloat32m1_t v64 = v41;
    float v65 = v5[v53];
    vfloat32m1_t v66 = __riscv_vfmacc_vf_f32m1(v64, v65, v50, v48);
    vfloat32m1_t v67 = v42;
    float v68 = v5[v54];
    vfloat32m1_t v69 = __riscv_vfmacc_vf_f32m1(v67, v68, v50, v48);
    vfloat32m1_t v70 = v43;
    float v71 = v5[v55];
    vfloat32m1_t v72 = __riscv_vfmacc_vf_f32m1(v70, v71, v50, v48);
    vfloat32m1_t v73 = v44;
    float v74 = v5[v56];
    vfloat32m1_t v75 = __riscv_vfmacc_vf_f32m1(v73, v74, v50, v48);
    vfloat32m1_t v76 = v45;
    float v77 = v5[v57];
    vfloat32m1_t v78 = __riscv_vfmacc_vf_f32m1(v76, v77, v50, v48);
    v39 = v60;
    v40 = v63;
    v41 = v66;
    v42 = v69;
    v43 = v72;
    v44 = v75;
    v45 = v78;
  }
  vfloat32m1_t v79 = v39;
  vfloat32m1_t v80 = v40;
  vfloat32m1_t v81 = v41;
  vfloat32m1_t v82 = v42;
  vfloat32m1_t v83 = v43;
  vfloat32m1_t v84 = v44;
  vfloat32m1_t v85 = v45;
  float* v86 = &v8[v11];
  __riscv_vse32_v_f32m1(v86, v79, v14);
  float* v87 = &v8[v9];
  __riscv_vse32_v_f32m1(v87, v80, v14);
  size_t v88 = v13 * v9;
  float* v89 = &v8[v88];
  __riscv_vse32_v_f32m1(v89, v81, v14);
  size_t v90 = v14 * v9;
  float* v91 = &v8[v90];
  __riscv_vse32_v_f32m1(v91, v82, v14);
  size_t v92 = v15 * v9;
  float* v93 = &v8[v92];
  __riscv_vse32_v_f32m1(v93, v83, v14);
  size_t v94 = v16 * v9;
  float* v95 = &v8[v94];
  __riscv_vse32_v_f32m1(v95, v84, v14);
  size_t v96 = v17 * v9;
  float* v97 = &v8[v96];
  __riscv_vse32_v_f32m1(v97, v85, v14);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_3x8_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 3
//   NR         : 8
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_3x8_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v24 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v25 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v26 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v27 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v28;
  v28 = v20;
  vfloat32m1_t v29;
  v29 = v21;
  vfloat32m1_t v30;
  v30 = v22;
  vfloat32m1_t v31;
  v31 = v23;
  vfloat32m1_t v32;
  v32 = v24;
  vfloat32m1_t v33;
  v33 = v25;
  vfloat32m1_t v34;
  v34 = v26;
  vfloat32m1_t v35;
  v35 = v27;
  for (size_t i36 = v11; i36 < v1; i36 += v12) {
    size_t v37 = i36 * v4;
    size_t v38 = 3;
    float* v39 = &v3[v37];
    vfloat32m1_t v40 = __riscv_vle32_v_f32m1(v39, v38);
    size_t v41 = i36 * v6;
    size_t v42 = v41 + v12;
    size_t v43 = v41 + v13;
    size_t v44 = v41 + v14;
    size_t v45 = v41 + v15;
    size_t v46 = v41 + v16;
    size_t v47 = v41 + v17;
    size_t v48 = v41 + v18;
    vfloat32m1_t v49 = v28;
    float v50 = v5[v41];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v40, v38);
    vfloat32m1_t v52 = v29;
    float v53 = v5[v42];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v40, v38);
    vfloat32m1_t v55 = v30;
    float v56 = v5[v43];
    vfloat32m1_t v57 = __riscv_vfmacc_vf_f32m1(v55, v56, v40, v38);
    vfloat32m1_t v58 = v31;
    float v59 = v5[v44];
    vfloat32m1_t v60 = __riscv_vfmacc_vf_f32m1(v58, v59, v40, v38);
    vfloat32m1_t v61 = v32;
    float v62 = v5[v45];
    vfloat32m1_t v63 = __riscv_vfmacc_vf_f32m1(v61, v62, v40, v38);
    vfloat32m1_t v64 = v33;
    float v65 = v5[v46];
    vfloat32m1_t v66 = __riscv_vfmacc_vf_f32m1(v64, v65, v40, v38);
    vfloat32m1_t v67 = v34;
    float v68 = v5[v47];
    vfloat32m1_t v69 = __riscv_vfmacc_vf_f32m1(v67, v68, v40, v38);
    vfloat32m1_t v70 = v35;
    float v71 = v5[v48];
    vfloat32m1_t v72 = __riscv_vfmacc_vf_f32m1(v70, v71, v40, v38);
    v28 = v51;
    v29 = v54;
    v30 = v57;
    v31 = v60;
    v32 = v63;
    v33 = v66;
    v34 = v69;
    v35 = v72;
  }
  vfloat32m1_t v73 = v28;
  vfloat32m1_t v74 = v29;
  vfloat32m1_t v75 = v30;
  vfloat32m1_t v76 = v31;
  vfloat32m1_t v77 = v32;
  vfloat32m1_t v78 = v33;
  vfloat32m1_t v79 = v34;
  vfloat32m1_t v80 = v35;
  float* v81 = &v8[v11];
  __riscv_vse32_v_f32m1(v81, v73, v14);
  float* v82 = &v8[v9];
  __riscv_vse32_v_f32m1(v82, v74, v14);
  size_t v83 = v13 * v9;
  float* v84 = &v8[v83];
  __riscv_vse32_v_f32m1(v84, v75, v14);
  size_t v85 = v14 * v9;
  float* v86 = &v8[v85];
  __riscv_vse32_v_f32m1(v86, v76, v14);
  size_t v87 = v15 * v9;
  float* v88 = &v8[v87];
  __riscv_vse32_v_f32m1(v88, v77, v14);
  size_t v89 = v16 * v9;
  float* v90 = &v8[v89];
  __riscv_vse32_v_f32m1(v90, v78, v14);
  size_t v91 = v17 * v9;
  float* v92 = &v8[v91];
  __riscv_vse32_v_f32m1(v92, v79, v14);
  size_t v93 = v18 * v9;
  float* v94 = &v8[v93];
  __riscv_vse32_v_f32m1(v94, v80, v14);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_3x8_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 3
//   NR         : 8
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_3x8_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v14);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v14);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v14);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v14);
  size_t v30 = v15 * v9;
  float* v31 = &v8[v30];
  vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v14);
  size_t v33 = v16 * v9;
  float* v34 = &v8[v33];
  vfloat32m1_t v35 = __riscv_vle32_v_f32m1(v34, v14);
  size_t v36 = v17 * v9;
  float* v37 = &v8[v36];
  vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v14);
  size_t v39 = v18 * v9;
  float* v40 = &v8[v39];
  vfloat32m1_t v41 = __riscv_vle32_v_f32m1(v40, v14);
  vfloat32m1_t v42;
  v42 = v21;
  vfloat32m1_t v43;
  v43 = v23;
  vfloat32m1_t v44;
  v44 = v26;
  vfloat32m1_t v45;
  v45 = v29;
  vfloat32m1_t v46;
  v46 = v32;
  vfloat32m1_t v47;
  v47 = v35;
  vfloat32m1_t v48;
  v48 = v38;
  vfloat32m1_t v49;
  v49 = v41;
  for (size_t i50 = v11; i50 < v1; i50 += v12) {
    size_t v51 = i50 * v4;
    size_t v52 = 3;
    float* v53 = &v3[v51];
    vfloat32m1_t v54 = __riscv_vle32_v_f32m1(v53, v52);
    size_t v55 = i50 * v6;
    size_t v56 = v55 + v12;
    size_t v57 = v55 + v13;
    size_t v58 = v55 + v14;
    size_t v59 = v55 + v15;
    size_t v60 = v55 + v16;
    size_t v61 = v55 + v17;
    size_t v62 = v55 + v18;
    vfloat32m1_t v63 = v42;
    float v64 = v5[v55];
    vfloat32m1_t v65 = __riscv_vfmacc_vf_f32m1(v63, v64, v54, v52);
    vfloat32m1_t v66 = v43;
    float v67 = v5[v56];
    vfloat32m1_t v68 = __riscv_vfmacc_vf_f32m1(v66, v67, v54, v52);
    vfloat32m1_t v69 = v44;
    float v70 = v5[v57];
    vfloat32m1_t v71 = __riscv_vfmacc_vf_f32m1(v69, v70, v54, v52);
    vfloat32m1_t v72 = v45;
    float v73 = v5[v58];
    vfloat32m1_t v74 = __riscv_vfmacc_vf_f32m1(v72, v73, v54, v52);
    vfloat32m1_t v75 = v46;
    float v76 = v5[v59];
    vfloat32m1_t v77 = __riscv_vfmacc_vf_f32m1(v75, v76, v54, v52);
    vfloat32m1_t v78 = v47;
    float v79 = v5[v60];
    vfloat32m1_t v80 = __riscv_vfmacc_vf_f32m1(v78, v79, v54, v52);
    vfloat32m1_t v81 = v48;
    float v82 = v5[v61];
    vfloat32m1_t v83 = __riscv_vfmacc_vf_f32m1(v81, v82, v54, v52);
    vfloat32m1_t v84 = v49;
    float v85 = v5[v62];
    vfloat32m1_t v86 = __riscv_vfmacc_vf_f32m1(v84, v85, v54, v52);
    v42 = v65;
    v43 = v68;
    v44 = v71;
    v45 = v74;
    v46 = v77;
    v47 = v80;
    v48 = v83;
    v49 = v86;
  }
  vfloat32m1_t v87 = v42;
  vfloat32m1_t v88 = v43;
  vfloat32m1_t v89 = v44;
  vfloat32m1_t v90 = v45;
  vfloat32m1_t v91 = v46;
  vfloat32m1_t v92 = v47;
  vfloat32m1_t v93 = v48;
  vfloat32m1_t v94 = v49;
  float* v95 = &v8[v11];
  __riscv_vse32_v_f32m1(v95, v87, v14);
  float* v96 = &v8[v9];
  __riscv_vse32_v_f32m1(v96, v88, v14);
  size_t v97 = v13 * v9;
  float* v98 = &v8[v97];
  __riscv_vse32_v_f32m1(v98, v89, v14);
  size_t v99 = v14 * v9;
  float* v100 = &v8[v99];
  __riscv_vse32_v_f32m1(v100, v90, v14);
  size_t v101 = v15 * v9;
  float* v102 = &v8[v101];
  __riscv_vse32_v_f32m1(v102, v91, v14);
  size_t v103 = v16 * v9;
  float* v104 = &v8[v103];
  __riscv_vse32_v_f32m1(v104, v92, v14);
  size_t v105 = v17 * v9;
  float* v106 = &v8[v105];
  __riscv_vse32_v_f32m1(v106, v93, v14);
  size_t v107 = v18 * v9;
  float* v108 = &v8[v107];
  __riscv_vse32_v_f32m1(v108, v94, v14);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_4x1_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 4
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_4x1_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v21;
  v21 = v20;
  for (size_t i22 = v11; i22 < v1; i22 += v12) {
    size_t v23 = i22 * v4;
    size_t v24 = 4;
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
  __riscv_vse32_v_f32m1(v32, v31, v15);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_4x1_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 4
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_4x1_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v15);
  vfloat32m1_t v22;
  v22 = v21;
  for (size_t i23 = v11; i23 < v1; i23 += v12) {
    size_t v24 = i23 * v4;
    size_t v25 = 4;
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
  __riscv_vse32_v_f32m1(v33, v32, v15);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_4x2_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 4
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_4x2_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v22;
  v22 = v20;
  vfloat32m1_t v23;
  v23 = v21;
  for (size_t i24 = v11; i24 < v1; i24 += v12) {
    size_t v25 = i24 * v4;
    size_t v26 = 4;
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
  __riscv_vse32_v_f32m1(v39, v37, v15);
  float* v40 = &v8[v9];
  __riscv_vse32_v_f32m1(v40, v38, v15);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_4x2_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:58
//   MR         : 4
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_4x2_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v15);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v15);
  vfloat32m1_t v24;
  v24 = v21;
  vfloat32m1_t v25;
  v25 = v23;
  for (size_t i26 = v11; i26 < v1; i26 += v12) {
    size_t v27 = i26 * v4;
    size_t v28 = 4;
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
  __riscv_vse32_v_f32m1(v41, v39, v15);
  float* v42 = &v8[v9];
  __riscv_vse32_v_f32m1(v42, v40, v15);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_4x3_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 4
//   NR         : 3
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_4x3_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v23;
  v23 = v20;
  vfloat32m1_t v24;
  v24 = v21;
  vfloat32m1_t v25;
  v25 = v22;
  for (size_t i26 = v11; i26 < v1; i26 += v12) {
    size_t v27 = i26 * v4;
    size_t v28 = 4;
    float* v29 = &v3[v27];
    vfloat32m1_t v30 = __riscv_vle32_v_f32m1(v29, v28);
    size_t v31 = i26 * v6;
    size_t v32 = v31 + v12;
    size_t v33 = v31 + v13;
    vfloat32m1_t v34 = v23;
    float v35 = v5[v31];
    vfloat32m1_t v36 = __riscv_vfmacc_vf_f32m1(v34, v35, v30, v28);
    vfloat32m1_t v37 = v24;
    float v38 = v5[v32];
    vfloat32m1_t v39 = __riscv_vfmacc_vf_f32m1(v37, v38, v30, v28);
    vfloat32m1_t v40 = v25;
    float v41 = v5[v33];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v30, v28);
    v23 = v36;
    v24 = v39;
    v25 = v42;
  }
  vfloat32m1_t v43 = v23;
  vfloat32m1_t v44 = v24;
  vfloat32m1_t v45 = v25;
  float* v46 = &v8[v11];
  __riscv_vse32_v_f32m1(v46, v43, v15);
  float* v47 = &v8[v9];
  __riscv_vse32_v_f32m1(v47, v44, v15);
  size_t v48 = v13 * v9;
  float* v49 = &v8[v48];
  __riscv_vse32_v_f32m1(v49, v45, v15);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_4x3_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 4
//   NR         : 3
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_4x3_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v15);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v15);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v15);
  vfloat32m1_t v27;
  v27 = v21;
  vfloat32m1_t v28;
  v28 = v23;
  vfloat32m1_t v29;
  v29 = v26;
  for (size_t i30 = v11; i30 < v1; i30 += v12) {
    size_t v31 = i30 * v4;
    size_t v32 = 4;
    float* v33 = &v3[v31];
    vfloat32m1_t v34 = __riscv_vle32_v_f32m1(v33, v32);
    size_t v35 = i30 * v6;
    size_t v36 = v35 + v12;
    size_t v37 = v35 + v13;
    vfloat32m1_t v38 = v27;
    float v39 = v5[v35];
    vfloat32m1_t v40 = __riscv_vfmacc_vf_f32m1(v38, v39, v34, v32);
    vfloat32m1_t v41 = v28;
    float v42 = v5[v36];
    vfloat32m1_t v43 = __riscv_vfmacc_vf_f32m1(v41, v42, v34, v32);
    vfloat32m1_t v44 = v29;
    float v45 = v5[v37];
    vfloat32m1_t v46 = __riscv_vfmacc_vf_f32m1(v44, v45, v34, v32);
    v27 = v40;
    v28 = v43;
    v29 = v46;
  }
  vfloat32m1_t v47 = v27;
  vfloat32m1_t v48 = v28;
  vfloat32m1_t v49 = v29;
  float* v50 = &v8[v11];
  __riscv_vse32_v_f32m1(v50, v47, v15);
  float* v51 = &v8[v9];
  __riscv_vse32_v_f32m1(v51, v48, v15);
  size_t v52 = v13 * v9;
  float* v53 = &v8[v52];
  __riscv_vse32_v_f32m1(v53, v49, v15);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_4x4_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 4
//   NR         : 4
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_4x4_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v24;
  v24 = v20;
  vfloat32m1_t v25;
  v25 = v21;
  vfloat32m1_t v26;
  v26 = v22;
  vfloat32m1_t v27;
  v27 = v23;
  for (size_t i28 = v11; i28 < v1; i28 += v12) {
    size_t v29 = i28 * v4;
    size_t v30 = 4;
    float* v31 = &v3[v29];
    vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v30);
    size_t v33 = i28 * v6;
    size_t v34 = v33 + v12;
    size_t v35 = v33 + v13;
    size_t v36 = v33 + v14;
    vfloat32m1_t v37 = v24;
    float v38 = v5[v33];
    vfloat32m1_t v39 = __riscv_vfmacc_vf_f32m1(v37, v38, v32, v30);
    vfloat32m1_t v40 = v25;
    float v41 = v5[v34];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v32, v30);
    vfloat32m1_t v43 = v26;
    float v44 = v5[v35];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v32, v30);
    vfloat32m1_t v46 = v27;
    float v47 = v5[v36];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v32, v30);
    v24 = v39;
    v25 = v42;
    v26 = v45;
    v27 = v48;
  }
  vfloat32m1_t v49 = v24;
  vfloat32m1_t v50 = v25;
  vfloat32m1_t v51 = v26;
  vfloat32m1_t v52 = v27;
  float* v53 = &v8[v11];
  __riscv_vse32_v_f32m1(v53, v49, v15);
  float* v54 = &v8[v9];
  __riscv_vse32_v_f32m1(v54, v50, v15);
  size_t v55 = v13 * v9;
  float* v56 = &v8[v55];
  __riscv_vse32_v_f32m1(v56, v51, v15);
  size_t v57 = v14 * v9;
  float* v58 = &v8[v57];
  __riscv_vse32_v_f32m1(v58, v52, v15);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_4x4_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 4
//   NR         : 4
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_4x4_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v15);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v15);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v15);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v15);
  vfloat32m1_t v30;
  v30 = v21;
  vfloat32m1_t v31;
  v31 = v23;
  vfloat32m1_t v32;
  v32 = v26;
  vfloat32m1_t v33;
  v33 = v29;
  for (size_t i34 = v11; i34 < v1; i34 += v12) {
    size_t v35 = i34 * v4;
    size_t v36 = 4;
    float* v37 = &v3[v35];
    vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v36);
    size_t v39 = i34 * v6;
    size_t v40 = v39 + v12;
    size_t v41 = v39 + v13;
    size_t v42 = v39 + v14;
    vfloat32m1_t v43 = v30;
    float v44 = v5[v39];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v38, v36);
    vfloat32m1_t v46 = v31;
    float v47 = v5[v40];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v38, v36);
    vfloat32m1_t v49 = v32;
    float v50 = v5[v41];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v38, v36);
    vfloat32m1_t v52 = v33;
    float v53 = v5[v42];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v38, v36);
    v30 = v45;
    v31 = v48;
    v32 = v51;
    v33 = v54;
  }
  vfloat32m1_t v55 = v30;
  vfloat32m1_t v56 = v31;
  vfloat32m1_t v57 = v32;
  vfloat32m1_t v58 = v33;
  float* v59 = &v8[v11];
  __riscv_vse32_v_f32m1(v59, v55, v15);
  float* v60 = &v8[v9];
  __riscv_vse32_v_f32m1(v60, v56, v15);
  size_t v61 = v13 * v9;
  float* v62 = &v8[v61];
  __riscv_vse32_v_f32m1(v62, v57, v15);
  size_t v63 = v14 * v9;
  float* v64 = &v8[v63];
  __riscv_vse32_v_f32m1(v64, v58, v15);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_4x5_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 4
//   NR         : 5
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_4x5_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v24 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v25;
  v25 = v20;
  vfloat32m1_t v26;
  v26 = v21;
  vfloat32m1_t v27;
  v27 = v22;
  vfloat32m1_t v28;
  v28 = v23;
  vfloat32m1_t v29;
  v29 = v24;
  for (size_t i30 = v11; i30 < v1; i30 += v12) {
    size_t v31 = i30 * v4;
    size_t v32 = 4;
    float* v33 = &v3[v31];
    vfloat32m1_t v34 = __riscv_vle32_v_f32m1(v33, v32);
    size_t v35 = i30 * v6;
    size_t v36 = v35 + v12;
    size_t v37 = v35 + v13;
    size_t v38 = v35 + v14;
    size_t v39 = v35 + v15;
    vfloat32m1_t v40 = v25;
    float v41 = v5[v35];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v34, v32);
    vfloat32m1_t v43 = v26;
    float v44 = v5[v36];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v34, v32);
    vfloat32m1_t v46 = v27;
    float v47 = v5[v37];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v34, v32);
    vfloat32m1_t v49 = v28;
    float v50 = v5[v38];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v34, v32);
    vfloat32m1_t v52 = v29;
    float v53 = v5[v39];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v34, v32);
    v25 = v42;
    v26 = v45;
    v27 = v48;
    v28 = v51;
    v29 = v54;
  }
  vfloat32m1_t v55 = v25;
  vfloat32m1_t v56 = v26;
  vfloat32m1_t v57 = v27;
  vfloat32m1_t v58 = v28;
  vfloat32m1_t v59 = v29;
  float* v60 = &v8[v11];
  __riscv_vse32_v_f32m1(v60, v55, v15);
  float* v61 = &v8[v9];
  __riscv_vse32_v_f32m1(v61, v56, v15);
  size_t v62 = v13 * v9;
  float* v63 = &v8[v62];
  __riscv_vse32_v_f32m1(v63, v57, v15);
  size_t v64 = v14 * v9;
  float* v65 = &v8[v64];
  __riscv_vse32_v_f32m1(v65, v58, v15);
  size_t v66 = v15 * v9;
  float* v67 = &v8[v66];
  __riscv_vse32_v_f32m1(v67, v59, v15);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_4x5_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 4
//   NR         : 5
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_4x5_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v15);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v15);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v15);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v15);
  size_t v30 = v15 * v9;
  float* v31 = &v8[v30];
  vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v15);
  vfloat32m1_t v33;
  v33 = v21;
  vfloat32m1_t v34;
  v34 = v23;
  vfloat32m1_t v35;
  v35 = v26;
  vfloat32m1_t v36;
  v36 = v29;
  vfloat32m1_t v37;
  v37 = v32;
  for (size_t i38 = v11; i38 < v1; i38 += v12) {
    size_t v39 = i38 * v4;
    size_t v40 = 4;
    float* v41 = &v3[v39];
    vfloat32m1_t v42 = __riscv_vle32_v_f32m1(v41, v40);
    size_t v43 = i38 * v6;
    size_t v44 = v43 + v12;
    size_t v45 = v43 + v13;
    size_t v46 = v43 + v14;
    size_t v47 = v43 + v15;
    vfloat32m1_t v48 = v33;
    float v49 = v5[v43];
    vfloat32m1_t v50 = __riscv_vfmacc_vf_f32m1(v48, v49, v42, v40);
    vfloat32m1_t v51 = v34;
    float v52 = v5[v44];
    vfloat32m1_t v53 = __riscv_vfmacc_vf_f32m1(v51, v52, v42, v40);
    vfloat32m1_t v54 = v35;
    float v55 = v5[v45];
    vfloat32m1_t v56 = __riscv_vfmacc_vf_f32m1(v54, v55, v42, v40);
    vfloat32m1_t v57 = v36;
    float v58 = v5[v46];
    vfloat32m1_t v59 = __riscv_vfmacc_vf_f32m1(v57, v58, v42, v40);
    vfloat32m1_t v60 = v37;
    float v61 = v5[v47];
    vfloat32m1_t v62 = __riscv_vfmacc_vf_f32m1(v60, v61, v42, v40);
    v33 = v50;
    v34 = v53;
    v35 = v56;
    v36 = v59;
    v37 = v62;
  }
  vfloat32m1_t v63 = v33;
  vfloat32m1_t v64 = v34;
  vfloat32m1_t v65 = v35;
  vfloat32m1_t v66 = v36;
  vfloat32m1_t v67 = v37;
  float* v68 = &v8[v11];
  __riscv_vse32_v_f32m1(v68, v63, v15);
  float* v69 = &v8[v9];
  __riscv_vse32_v_f32m1(v69, v64, v15);
  size_t v70 = v13 * v9;
  float* v71 = &v8[v70];
  __riscv_vse32_v_f32m1(v71, v65, v15);
  size_t v72 = v14 * v9;
  float* v73 = &v8[v72];
  __riscv_vse32_v_f32m1(v73, v66, v15);
  size_t v74 = v15 * v9;
  float* v75 = &v8[v74];
  __riscv_vse32_v_f32m1(v75, v67, v15);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_4x6_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 4
//   NR         : 6
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_4x6_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v24 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v25 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v26;
  v26 = v20;
  vfloat32m1_t v27;
  v27 = v21;
  vfloat32m1_t v28;
  v28 = v22;
  vfloat32m1_t v29;
  v29 = v23;
  vfloat32m1_t v30;
  v30 = v24;
  vfloat32m1_t v31;
  v31 = v25;
  for (size_t i32 = v11; i32 < v1; i32 += v12) {
    size_t v33 = i32 * v4;
    size_t v34 = 4;
    float* v35 = &v3[v33];
    vfloat32m1_t v36 = __riscv_vle32_v_f32m1(v35, v34);
    size_t v37 = i32 * v6;
    size_t v38 = v37 + v12;
    size_t v39 = v37 + v13;
    size_t v40 = v37 + v14;
    size_t v41 = v37 + v15;
    size_t v42 = v37 + v16;
    vfloat32m1_t v43 = v26;
    float v44 = v5[v37];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v36, v34);
    vfloat32m1_t v46 = v27;
    float v47 = v5[v38];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v36, v34);
    vfloat32m1_t v49 = v28;
    float v50 = v5[v39];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v36, v34);
    vfloat32m1_t v52 = v29;
    float v53 = v5[v40];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v36, v34);
    vfloat32m1_t v55 = v30;
    float v56 = v5[v41];
    vfloat32m1_t v57 = __riscv_vfmacc_vf_f32m1(v55, v56, v36, v34);
    vfloat32m1_t v58 = v31;
    float v59 = v5[v42];
    vfloat32m1_t v60 = __riscv_vfmacc_vf_f32m1(v58, v59, v36, v34);
    v26 = v45;
    v27 = v48;
    v28 = v51;
    v29 = v54;
    v30 = v57;
    v31 = v60;
  }
  vfloat32m1_t v61 = v26;
  vfloat32m1_t v62 = v27;
  vfloat32m1_t v63 = v28;
  vfloat32m1_t v64 = v29;
  vfloat32m1_t v65 = v30;
  vfloat32m1_t v66 = v31;
  float* v67 = &v8[v11];
  __riscv_vse32_v_f32m1(v67, v61, v15);
  float* v68 = &v8[v9];
  __riscv_vse32_v_f32m1(v68, v62, v15);
  size_t v69 = v13 * v9;
  float* v70 = &v8[v69];
  __riscv_vse32_v_f32m1(v70, v63, v15);
  size_t v71 = v14 * v9;
  float* v72 = &v8[v71];
  __riscv_vse32_v_f32m1(v72, v64, v15);
  size_t v73 = v15 * v9;
  float* v74 = &v8[v73];
  __riscv_vse32_v_f32m1(v74, v65, v15);
  size_t v75 = v16 * v9;
  float* v76 = &v8[v75];
  __riscv_vse32_v_f32m1(v76, v66, v15);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_4x6_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 4
//   NR         : 6
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_4x6_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v15);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v15);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v15);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v15);
  size_t v30 = v15 * v9;
  float* v31 = &v8[v30];
  vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v15);
  size_t v33 = v16 * v9;
  float* v34 = &v8[v33];
  vfloat32m1_t v35 = __riscv_vle32_v_f32m1(v34, v15);
  vfloat32m1_t v36;
  v36 = v21;
  vfloat32m1_t v37;
  v37 = v23;
  vfloat32m1_t v38;
  v38 = v26;
  vfloat32m1_t v39;
  v39 = v29;
  vfloat32m1_t v40;
  v40 = v32;
  vfloat32m1_t v41;
  v41 = v35;
  for (size_t i42 = v11; i42 < v1; i42 += v12) {
    size_t v43 = i42 * v4;
    size_t v44 = 4;
    float* v45 = &v3[v43];
    vfloat32m1_t v46 = __riscv_vle32_v_f32m1(v45, v44);
    size_t v47 = i42 * v6;
    size_t v48 = v47 + v12;
    size_t v49 = v47 + v13;
    size_t v50 = v47 + v14;
    size_t v51 = v47 + v15;
    size_t v52 = v47 + v16;
    vfloat32m1_t v53 = v36;
    float v54 = v5[v47];
    vfloat32m1_t v55 = __riscv_vfmacc_vf_f32m1(v53, v54, v46, v44);
    vfloat32m1_t v56 = v37;
    float v57 = v5[v48];
    vfloat32m1_t v58 = __riscv_vfmacc_vf_f32m1(v56, v57, v46, v44);
    vfloat32m1_t v59 = v38;
    float v60 = v5[v49];
    vfloat32m1_t v61 = __riscv_vfmacc_vf_f32m1(v59, v60, v46, v44);
    vfloat32m1_t v62 = v39;
    float v63 = v5[v50];
    vfloat32m1_t v64 = __riscv_vfmacc_vf_f32m1(v62, v63, v46, v44);
    vfloat32m1_t v65 = v40;
    float v66 = v5[v51];
    vfloat32m1_t v67 = __riscv_vfmacc_vf_f32m1(v65, v66, v46, v44);
    vfloat32m1_t v68 = v41;
    float v69 = v5[v52];
    vfloat32m1_t v70 = __riscv_vfmacc_vf_f32m1(v68, v69, v46, v44);
    v36 = v55;
    v37 = v58;
    v38 = v61;
    v39 = v64;
    v40 = v67;
    v41 = v70;
  }
  vfloat32m1_t v71 = v36;
  vfloat32m1_t v72 = v37;
  vfloat32m1_t v73 = v38;
  vfloat32m1_t v74 = v39;
  vfloat32m1_t v75 = v40;
  vfloat32m1_t v76 = v41;
  float* v77 = &v8[v11];
  __riscv_vse32_v_f32m1(v77, v71, v15);
  float* v78 = &v8[v9];
  __riscv_vse32_v_f32m1(v78, v72, v15);
  size_t v79 = v13 * v9;
  float* v80 = &v8[v79];
  __riscv_vse32_v_f32m1(v80, v73, v15);
  size_t v81 = v14 * v9;
  float* v82 = &v8[v81];
  __riscv_vse32_v_f32m1(v82, v74, v15);
  size_t v83 = v15 * v9;
  float* v84 = &v8[v83];
  __riscv_vse32_v_f32m1(v84, v75, v15);
  size_t v85 = v16 * v9;
  float* v86 = &v8[v85];
  __riscv_vse32_v_f32m1(v86, v76, v15);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_4x7_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 4
//   NR         : 7
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_4x7_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v24 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v25 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v26 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v27;
  v27 = v20;
  vfloat32m1_t v28;
  v28 = v21;
  vfloat32m1_t v29;
  v29 = v22;
  vfloat32m1_t v30;
  v30 = v23;
  vfloat32m1_t v31;
  v31 = v24;
  vfloat32m1_t v32;
  v32 = v25;
  vfloat32m1_t v33;
  v33 = v26;
  for (size_t i34 = v11; i34 < v1; i34 += v12) {
    size_t v35 = i34 * v4;
    size_t v36 = 4;
    float* v37 = &v3[v35];
    vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v36);
    size_t v39 = i34 * v6;
    size_t v40 = v39 + v12;
    size_t v41 = v39 + v13;
    size_t v42 = v39 + v14;
    size_t v43 = v39 + v15;
    size_t v44 = v39 + v16;
    size_t v45 = v39 + v17;
    vfloat32m1_t v46 = v27;
    float v47 = v5[v39];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v38, v36);
    vfloat32m1_t v49 = v28;
    float v50 = v5[v40];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v38, v36);
    vfloat32m1_t v52 = v29;
    float v53 = v5[v41];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v38, v36);
    vfloat32m1_t v55 = v30;
    float v56 = v5[v42];
    vfloat32m1_t v57 = __riscv_vfmacc_vf_f32m1(v55, v56, v38, v36);
    vfloat32m1_t v58 = v31;
    float v59 = v5[v43];
    vfloat32m1_t v60 = __riscv_vfmacc_vf_f32m1(v58, v59, v38, v36);
    vfloat32m1_t v61 = v32;
    float v62 = v5[v44];
    vfloat32m1_t v63 = __riscv_vfmacc_vf_f32m1(v61, v62, v38, v36);
    vfloat32m1_t v64 = v33;
    float v65 = v5[v45];
    vfloat32m1_t v66 = __riscv_vfmacc_vf_f32m1(v64, v65, v38, v36);
    v27 = v48;
    v28 = v51;
    v29 = v54;
    v30 = v57;
    v31 = v60;
    v32 = v63;
    v33 = v66;
  }
  vfloat32m1_t v67 = v27;
  vfloat32m1_t v68 = v28;
  vfloat32m1_t v69 = v29;
  vfloat32m1_t v70 = v30;
  vfloat32m1_t v71 = v31;
  vfloat32m1_t v72 = v32;
  vfloat32m1_t v73 = v33;
  float* v74 = &v8[v11];
  __riscv_vse32_v_f32m1(v74, v67, v15);
  float* v75 = &v8[v9];
  __riscv_vse32_v_f32m1(v75, v68, v15);
  size_t v76 = v13 * v9;
  float* v77 = &v8[v76];
  __riscv_vse32_v_f32m1(v77, v69, v15);
  size_t v78 = v14 * v9;
  float* v79 = &v8[v78];
  __riscv_vse32_v_f32m1(v79, v70, v15);
  size_t v80 = v15 * v9;
  float* v81 = &v8[v80];
  __riscv_vse32_v_f32m1(v81, v71, v15);
  size_t v82 = v16 * v9;
  float* v83 = &v8[v82];
  __riscv_vse32_v_f32m1(v83, v72, v15);
  size_t v84 = v17 * v9;
  float* v85 = &v8[v84];
  __riscv_vse32_v_f32m1(v85, v73, v15);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_4x7_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 4
//   NR         : 7
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_4x7_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v15);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v15);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v15);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v15);
  size_t v30 = v15 * v9;
  float* v31 = &v8[v30];
  vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v15);
  size_t v33 = v16 * v9;
  float* v34 = &v8[v33];
  vfloat32m1_t v35 = __riscv_vle32_v_f32m1(v34, v15);
  size_t v36 = v17 * v9;
  float* v37 = &v8[v36];
  vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v15);
  vfloat32m1_t v39;
  v39 = v21;
  vfloat32m1_t v40;
  v40 = v23;
  vfloat32m1_t v41;
  v41 = v26;
  vfloat32m1_t v42;
  v42 = v29;
  vfloat32m1_t v43;
  v43 = v32;
  vfloat32m1_t v44;
  v44 = v35;
  vfloat32m1_t v45;
  v45 = v38;
  for (size_t i46 = v11; i46 < v1; i46 += v12) {
    size_t v47 = i46 * v4;
    size_t v48 = 4;
    float* v49 = &v3[v47];
    vfloat32m1_t v50 = __riscv_vle32_v_f32m1(v49, v48);
    size_t v51 = i46 * v6;
    size_t v52 = v51 + v12;
    size_t v53 = v51 + v13;
    size_t v54 = v51 + v14;
    size_t v55 = v51 + v15;
    size_t v56 = v51 + v16;
    size_t v57 = v51 + v17;
    vfloat32m1_t v58 = v39;
    float v59 = v5[v51];
    vfloat32m1_t v60 = __riscv_vfmacc_vf_f32m1(v58, v59, v50, v48);
    vfloat32m1_t v61 = v40;
    float v62 = v5[v52];
    vfloat32m1_t v63 = __riscv_vfmacc_vf_f32m1(v61, v62, v50, v48);
    vfloat32m1_t v64 = v41;
    float v65 = v5[v53];
    vfloat32m1_t v66 = __riscv_vfmacc_vf_f32m1(v64, v65, v50, v48);
    vfloat32m1_t v67 = v42;
    float v68 = v5[v54];
    vfloat32m1_t v69 = __riscv_vfmacc_vf_f32m1(v67, v68, v50, v48);
    vfloat32m1_t v70 = v43;
    float v71 = v5[v55];
    vfloat32m1_t v72 = __riscv_vfmacc_vf_f32m1(v70, v71, v50, v48);
    vfloat32m1_t v73 = v44;
    float v74 = v5[v56];
    vfloat32m1_t v75 = __riscv_vfmacc_vf_f32m1(v73, v74, v50, v48);
    vfloat32m1_t v76 = v45;
    float v77 = v5[v57];
    vfloat32m1_t v78 = __riscv_vfmacc_vf_f32m1(v76, v77, v50, v48);
    v39 = v60;
    v40 = v63;
    v41 = v66;
    v42 = v69;
    v43 = v72;
    v44 = v75;
    v45 = v78;
  }
  vfloat32m1_t v79 = v39;
  vfloat32m1_t v80 = v40;
  vfloat32m1_t v81 = v41;
  vfloat32m1_t v82 = v42;
  vfloat32m1_t v83 = v43;
  vfloat32m1_t v84 = v44;
  vfloat32m1_t v85 = v45;
  float* v86 = &v8[v11];
  __riscv_vse32_v_f32m1(v86, v79, v15);
  float* v87 = &v8[v9];
  __riscv_vse32_v_f32m1(v87, v80, v15);
  size_t v88 = v13 * v9;
  float* v89 = &v8[v88];
  __riscv_vse32_v_f32m1(v89, v81, v15);
  size_t v90 = v14 * v9;
  float* v91 = &v8[v90];
  __riscv_vse32_v_f32m1(v91, v82, v15);
  size_t v92 = v15 * v9;
  float* v93 = &v8[v92];
  __riscv_vse32_v_f32m1(v93, v83, v15);
  size_t v94 = v16 * v9;
  float* v95 = &v8[v94];
  __riscv_vse32_v_f32m1(v95, v84, v15);
  size_t v96 = v17 * v9;
  float* v97 = &v8[v96];
  __riscv_vse32_v_f32m1(v97, v85, v15);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_4x8_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 4
//   NR         : 8
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_4x8_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v24 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v25 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v26 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v27 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v28;
  v28 = v20;
  vfloat32m1_t v29;
  v29 = v21;
  vfloat32m1_t v30;
  v30 = v22;
  vfloat32m1_t v31;
  v31 = v23;
  vfloat32m1_t v32;
  v32 = v24;
  vfloat32m1_t v33;
  v33 = v25;
  vfloat32m1_t v34;
  v34 = v26;
  vfloat32m1_t v35;
  v35 = v27;
  for (size_t i36 = v11; i36 < v1; i36 += v12) {
    size_t v37 = i36 * v4;
    size_t v38 = 4;
    float* v39 = &v3[v37];
    vfloat32m1_t v40 = __riscv_vle32_v_f32m1(v39, v38);
    size_t v41 = i36 * v6;
    size_t v42 = v41 + v12;
    size_t v43 = v41 + v13;
    size_t v44 = v41 + v14;
    size_t v45 = v41 + v15;
    size_t v46 = v41 + v16;
    size_t v47 = v41 + v17;
    size_t v48 = v41 + v18;
    vfloat32m1_t v49 = v28;
    float v50 = v5[v41];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v40, v38);
    vfloat32m1_t v52 = v29;
    float v53 = v5[v42];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v40, v38);
    vfloat32m1_t v55 = v30;
    float v56 = v5[v43];
    vfloat32m1_t v57 = __riscv_vfmacc_vf_f32m1(v55, v56, v40, v38);
    vfloat32m1_t v58 = v31;
    float v59 = v5[v44];
    vfloat32m1_t v60 = __riscv_vfmacc_vf_f32m1(v58, v59, v40, v38);
    vfloat32m1_t v61 = v32;
    float v62 = v5[v45];
    vfloat32m1_t v63 = __riscv_vfmacc_vf_f32m1(v61, v62, v40, v38);
    vfloat32m1_t v64 = v33;
    float v65 = v5[v46];
    vfloat32m1_t v66 = __riscv_vfmacc_vf_f32m1(v64, v65, v40, v38);
    vfloat32m1_t v67 = v34;
    float v68 = v5[v47];
    vfloat32m1_t v69 = __riscv_vfmacc_vf_f32m1(v67, v68, v40, v38);
    vfloat32m1_t v70 = v35;
    float v71 = v5[v48];
    vfloat32m1_t v72 = __riscv_vfmacc_vf_f32m1(v70, v71, v40, v38);
    v28 = v51;
    v29 = v54;
    v30 = v57;
    v31 = v60;
    v32 = v63;
    v33 = v66;
    v34 = v69;
    v35 = v72;
  }
  vfloat32m1_t v73 = v28;
  vfloat32m1_t v74 = v29;
  vfloat32m1_t v75 = v30;
  vfloat32m1_t v76 = v31;
  vfloat32m1_t v77 = v32;
  vfloat32m1_t v78 = v33;
  vfloat32m1_t v79 = v34;
  vfloat32m1_t v80 = v35;
  float* v81 = &v8[v11];
  __riscv_vse32_v_f32m1(v81, v73, v15);
  float* v82 = &v8[v9];
  __riscv_vse32_v_f32m1(v82, v74, v15);
  size_t v83 = v13 * v9;
  float* v84 = &v8[v83];
  __riscv_vse32_v_f32m1(v84, v75, v15);
  size_t v85 = v14 * v9;
  float* v86 = &v8[v85];
  __riscv_vse32_v_f32m1(v86, v76, v15);
  size_t v87 = v15 * v9;
  float* v88 = &v8[v87];
  __riscv_vse32_v_f32m1(v88, v77, v15);
  size_t v89 = v16 * v9;
  float* v90 = &v8[v89];
  __riscv_vse32_v_f32m1(v90, v78, v15);
  size_t v91 = v17 * v9;
  float* v92 = &v8[v91];
  __riscv_vse32_v_f32m1(v92, v79, v15);
  size_t v93 = v18 * v9;
  float* v94 = &v8[v93];
  __riscv_vse32_v_f32m1(v94, v80, v15);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_4x8_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 4
//   NR         : 8
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_4x8_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v15);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v15);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v15);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v15);
  size_t v30 = v15 * v9;
  float* v31 = &v8[v30];
  vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v15);
  size_t v33 = v16 * v9;
  float* v34 = &v8[v33];
  vfloat32m1_t v35 = __riscv_vle32_v_f32m1(v34, v15);
  size_t v36 = v17 * v9;
  float* v37 = &v8[v36];
  vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v15);
  size_t v39 = v18 * v9;
  float* v40 = &v8[v39];
  vfloat32m1_t v41 = __riscv_vle32_v_f32m1(v40, v15);
  vfloat32m1_t v42;
  v42 = v21;
  vfloat32m1_t v43;
  v43 = v23;
  vfloat32m1_t v44;
  v44 = v26;
  vfloat32m1_t v45;
  v45 = v29;
  vfloat32m1_t v46;
  v46 = v32;
  vfloat32m1_t v47;
  v47 = v35;
  vfloat32m1_t v48;
  v48 = v38;
  vfloat32m1_t v49;
  v49 = v41;
  for (size_t i50 = v11; i50 < v1; i50 += v12) {
    size_t v51 = i50 * v4;
    size_t v52 = 4;
    float* v53 = &v3[v51];
    vfloat32m1_t v54 = __riscv_vle32_v_f32m1(v53, v52);
    size_t v55 = i50 * v6;
    size_t v56 = v55 + v12;
    size_t v57 = v55 + v13;
    size_t v58 = v55 + v14;
    size_t v59 = v55 + v15;
    size_t v60 = v55 + v16;
    size_t v61 = v55 + v17;
    size_t v62 = v55 + v18;
    vfloat32m1_t v63 = v42;
    float v64 = v5[v55];
    vfloat32m1_t v65 = __riscv_vfmacc_vf_f32m1(v63, v64, v54, v52);
    vfloat32m1_t v66 = v43;
    float v67 = v5[v56];
    vfloat32m1_t v68 = __riscv_vfmacc_vf_f32m1(v66, v67, v54, v52);
    vfloat32m1_t v69 = v44;
    float v70 = v5[v57];
    vfloat32m1_t v71 = __riscv_vfmacc_vf_f32m1(v69, v70, v54, v52);
    vfloat32m1_t v72 = v45;
    float v73 = v5[v58];
    vfloat32m1_t v74 = __riscv_vfmacc_vf_f32m1(v72, v73, v54, v52);
    vfloat32m1_t v75 = v46;
    float v76 = v5[v59];
    vfloat32m1_t v77 = __riscv_vfmacc_vf_f32m1(v75, v76, v54, v52);
    vfloat32m1_t v78 = v47;
    float v79 = v5[v60];
    vfloat32m1_t v80 = __riscv_vfmacc_vf_f32m1(v78, v79, v54, v52);
    vfloat32m1_t v81 = v48;
    float v82 = v5[v61];
    vfloat32m1_t v83 = __riscv_vfmacc_vf_f32m1(v81, v82, v54, v52);
    vfloat32m1_t v84 = v49;
    float v85 = v5[v62];
    vfloat32m1_t v86 = __riscv_vfmacc_vf_f32m1(v84, v85, v54, v52);
    v42 = v65;
    v43 = v68;
    v44 = v71;
    v45 = v74;
    v46 = v77;
    v47 = v80;
    v48 = v83;
    v49 = v86;
  }
  vfloat32m1_t v87 = v42;
  vfloat32m1_t v88 = v43;
  vfloat32m1_t v89 = v44;
  vfloat32m1_t v90 = v45;
  vfloat32m1_t v91 = v46;
  vfloat32m1_t v92 = v47;
  vfloat32m1_t v93 = v48;
  vfloat32m1_t v94 = v49;
  float* v95 = &v8[v11];
  __riscv_vse32_v_f32m1(v95, v87, v15);
  float* v96 = &v8[v9];
  __riscv_vse32_v_f32m1(v96, v88, v15);
  size_t v97 = v13 * v9;
  float* v98 = &v8[v97];
  __riscv_vse32_v_f32m1(v98, v89, v15);
  size_t v99 = v14 * v9;
  float* v100 = &v8[v99];
  __riscv_vse32_v_f32m1(v100, v90, v15);
  size_t v101 = v15 * v9;
  float* v102 = &v8[v101];
  __riscv_vse32_v_f32m1(v102, v91, v15);
  size_t v103 = v16 * v9;
  float* v104 = &v8[v103];
  __riscv_vse32_v_f32m1(v104, v92, v15);
  size_t v105 = v17 * v9;
  float* v106 = &v8[v105];
  __riscv_vse32_v_f32m1(v106, v93, v15);
  size_t v107 = v18 * v9;
  float* v108 = &v8[v107];
  __riscv_vse32_v_f32m1(v108, v94, v15);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_5x1_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 5
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_5x1_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v21;
  v21 = v20;
  for (size_t i22 = v11; i22 < v1; i22 += v12) {
    size_t v23 = i22 * v4;
    size_t v24 = 5;
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
  __riscv_vse32_v_f32m1(v32, v31, v16);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_5x1_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 5
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_5x1_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v16);
  vfloat32m1_t v22;
  v22 = v21;
  for (size_t i23 = v11; i23 < v1; i23 += v12) {
    size_t v24 = i23 * v4;
    size_t v25 = 5;
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
  __riscv_vse32_v_f32m1(v33, v32, v16);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_5x2_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 5
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_5x2_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v22;
  v22 = v20;
  vfloat32m1_t v23;
  v23 = v21;
  for (size_t i24 = v11; i24 < v1; i24 += v12) {
    size_t v25 = i24 * v4;
    size_t v26 = 5;
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
  __riscv_vse32_v_f32m1(v39, v37, v16);
  float* v40 = &v8[v9];
  __riscv_vse32_v_f32m1(v40, v38, v16);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_5x2_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 5
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_5x2_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v16);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v16);
  vfloat32m1_t v24;
  v24 = v21;
  vfloat32m1_t v25;
  v25 = v23;
  for (size_t i26 = v11; i26 < v1; i26 += v12) {
    size_t v27 = i26 * v4;
    size_t v28 = 5;
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
  __riscv_vse32_v_f32m1(v41, v39, v16);
  float* v42 = &v8[v9];
  __riscv_vse32_v_f32m1(v42, v40, v16);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_5x3_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 5
//   NR         : 3
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_5x3_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v23;
  v23 = v20;
  vfloat32m1_t v24;
  v24 = v21;
  vfloat32m1_t v25;
  v25 = v22;
  for (size_t i26 = v11; i26 < v1; i26 += v12) {
    size_t v27 = i26 * v4;
    size_t v28 = 5;
    float* v29 = &v3[v27];
    vfloat32m1_t v30 = __riscv_vle32_v_f32m1(v29, v28);
    size_t v31 = i26 * v6;
    size_t v32 = v31 + v12;
    size_t v33 = v31 + v13;
    vfloat32m1_t v34 = v23;
    float v35 = v5[v31];
    vfloat32m1_t v36 = __riscv_vfmacc_vf_f32m1(v34, v35, v30, v28);
    vfloat32m1_t v37 = v24;
    float v38 = v5[v32];
    vfloat32m1_t v39 = __riscv_vfmacc_vf_f32m1(v37, v38, v30, v28);
    vfloat32m1_t v40 = v25;
    float v41 = v5[v33];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v30, v28);
    v23 = v36;
    v24 = v39;
    v25 = v42;
  }
  vfloat32m1_t v43 = v23;
  vfloat32m1_t v44 = v24;
  vfloat32m1_t v45 = v25;
  float* v46 = &v8[v11];
  __riscv_vse32_v_f32m1(v46, v43, v16);
  float* v47 = &v8[v9];
  __riscv_vse32_v_f32m1(v47, v44, v16);
  size_t v48 = v13 * v9;
  float* v49 = &v8[v48];
  __riscv_vse32_v_f32m1(v49, v45, v16);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_5x3_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 5
//   NR         : 3
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_5x3_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v16);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v16);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v16);
  vfloat32m1_t v27;
  v27 = v21;
  vfloat32m1_t v28;
  v28 = v23;
  vfloat32m1_t v29;
  v29 = v26;
  for (size_t i30 = v11; i30 < v1; i30 += v12) {
    size_t v31 = i30 * v4;
    size_t v32 = 5;
    float* v33 = &v3[v31];
    vfloat32m1_t v34 = __riscv_vle32_v_f32m1(v33, v32);
    size_t v35 = i30 * v6;
    size_t v36 = v35 + v12;
    size_t v37 = v35 + v13;
    vfloat32m1_t v38 = v27;
    float v39 = v5[v35];
    vfloat32m1_t v40 = __riscv_vfmacc_vf_f32m1(v38, v39, v34, v32);
    vfloat32m1_t v41 = v28;
    float v42 = v5[v36];
    vfloat32m1_t v43 = __riscv_vfmacc_vf_f32m1(v41, v42, v34, v32);
    vfloat32m1_t v44 = v29;
    float v45 = v5[v37];
    vfloat32m1_t v46 = __riscv_vfmacc_vf_f32m1(v44, v45, v34, v32);
    v27 = v40;
    v28 = v43;
    v29 = v46;
  }
  vfloat32m1_t v47 = v27;
  vfloat32m1_t v48 = v28;
  vfloat32m1_t v49 = v29;
  float* v50 = &v8[v11];
  __riscv_vse32_v_f32m1(v50, v47, v16);
  float* v51 = &v8[v9];
  __riscv_vse32_v_f32m1(v51, v48, v16);
  size_t v52 = v13 * v9;
  float* v53 = &v8[v52];
  __riscv_vse32_v_f32m1(v53, v49, v16);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_5x4_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 5
//   NR         : 4
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_5x4_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v24;
  v24 = v20;
  vfloat32m1_t v25;
  v25 = v21;
  vfloat32m1_t v26;
  v26 = v22;
  vfloat32m1_t v27;
  v27 = v23;
  for (size_t i28 = v11; i28 < v1; i28 += v12) {
    size_t v29 = i28 * v4;
    size_t v30 = 5;
    float* v31 = &v3[v29];
    vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v30);
    size_t v33 = i28 * v6;
    size_t v34 = v33 + v12;
    size_t v35 = v33 + v13;
    size_t v36 = v33 + v14;
    vfloat32m1_t v37 = v24;
    float v38 = v5[v33];
    vfloat32m1_t v39 = __riscv_vfmacc_vf_f32m1(v37, v38, v32, v30);
    vfloat32m1_t v40 = v25;
    float v41 = v5[v34];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v32, v30);
    vfloat32m1_t v43 = v26;
    float v44 = v5[v35];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v32, v30);
    vfloat32m1_t v46 = v27;
    float v47 = v5[v36];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v32, v30);
    v24 = v39;
    v25 = v42;
    v26 = v45;
    v27 = v48;
  }
  vfloat32m1_t v49 = v24;
  vfloat32m1_t v50 = v25;
  vfloat32m1_t v51 = v26;
  vfloat32m1_t v52 = v27;
  float* v53 = &v8[v11];
  __riscv_vse32_v_f32m1(v53, v49, v16);
  float* v54 = &v8[v9];
  __riscv_vse32_v_f32m1(v54, v50, v16);
  size_t v55 = v13 * v9;
  float* v56 = &v8[v55];
  __riscv_vse32_v_f32m1(v56, v51, v16);
  size_t v57 = v14 * v9;
  float* v58 = &v8[v57];
  __riscv_vse32_v_f32m1(v58, v52, v16);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_5x4_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 5
//   NR         : 4
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_5x4_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v16);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v16);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v16);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v16);
  vfloat32m1_t v30;
  v30 = v21;
  vfloat32m1_t v31;
  v31 = v23;
  vfloat32m1_t v32;
  v32 = v26;
  vfloat32m1_t v33;
  v33 = v29;
  for (size_t i34 = v11; i34 < v1; i34 += v12) {
    size_t v35 = i34 * v4;
    size_t v36 = 5;
    float* v37 = &v3[v35];
    vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v36);
    size_t v39 = i34 * v6;
    size_t v40 = v39 + v12;
    size_t v41 = v39 + v13;
    size_t v42 = v39 + v14;
    vfloat32m1_t v43 = v30;
    float v44 = v5[v39];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v38, v36);
    vfloat32m1_t v46 = v31;
    float v47 = v5[v40];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v38, v36);
    vfloat32m1_t v49 = v32;
    float v50 = v5[v41];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v38, v36);
    vfloat32m1_t v52 = v33;
    float v53 = v5[v42];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v38, v36);
    v30 = v45;
    v31 = v48;
    v32 = v51;
    v33 = v54;
  }
  vfloat32m1_t v55 = v30;
  vfloat32m1_t v56 = v31;
  vfloat32m1_t v57 = v32;
  vfloat32m1_t v58 = v33;
  float* v59 = &v8[v11];
  __riscv_vse32_v_f32m1(v59, v55, v16);
  float* v60 = &v8[v9];
  __riscv_vse32_v_f32m1(v60, v56, v16);
  size_t v61 = v13 * v9;
  float* v62 = &v8[v61];
  __riscv_vse32_v_f32m1(v62, v57, v16);
  size_t v63 = v14 * v9;
  float* v64 = &v8[v63];
  __riscv_vse32_v_f32m1(v64, v58, v16);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_5x5_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 5
//   NR         : 5
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_5x5_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v24 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v25;
  v25 = v20;
  vfloat32m1_t v26;
  v26 = v21;
  vfloat32m1_t v27;
  v27 = v22;
  vfloat32m1_t v28;
  v28 = v23;
  vfloat32m1_t v29;
  v29 = v24;
  for (size_t i30 = v11; i30 < v1; i30 += v12) {
    size_t v31 = i30 * v4;
    size_t v32 = 5;
    float* v33 = &v3[v31];
    vfloat32m1_t v34 = __riscv_vle32_v_f32m1(v33, v32);
    size_t v35 = i30 * v6;
    size_t v36 = v35 + v12;
    size_t v37 = v35 + v13;
    size_t v38 = v35 + v14;
    size_t v39 = v35 + v15;
    vfloat32m1_t v40 = v25;
    float v41 = v5[v35];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v34, v32);
    vfloat32m1_t v43 = v26;
    float v44 = v5[v36];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v34, v32);
    vfloat32m1_t v46 = v27;
    float v47 = v5[v37];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v34, v32);
    vfloat32m1_t v49 = v28;
    float v50 = v5[v38];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v34, v32);
    vfloat32m1_t v52 = v29;
    float v53 = v5[v39];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v34, v32);
    v25 = v42;
    v26 = v45;
    v27 = v48;
    v28 = v51;
    v29 = v54;
  }
  vfloat32m1_t v55 = v25;
  vfloat32m1_t v56 = v26;
  vfloat32m1_t v57 = v27;
  vfloat32m1_t v58 = v28;
  vfloat32m1_t v59 = v29;
  float* v60 = &v8[v11];
  __riscv_vse32_v_f32m1(v60, v55, v16);
  float* v61 = &v8[v9];
  __riscv_vse32_v_f32m1(v61, v56, v16);
  size_t v62 = v13 * v9;
  float* v63 = &v8[v62];
  __riscv_vse32_v_f32m1(v63, v57, v16);
  size_t v64 = v14 * v9;
  float* v65 = &v8[v64];
  __riscv_vse32_v_f32m1(v65, v58, v16);
  size_t v66 = v15 * v9;
  float* v67 = &v8[v66];
  __riscv_vse32_v_f32m1(v67, v59, v16);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_5x5_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 5
//   NR         : 5
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_5x5_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v16);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v16);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v16);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v16);
  size_t v30 = v15 * v9;
  float* v31 = &v8[v30];
  vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v16);
  vfloat32m1_t v33;
  v33 = v21;
  vfloat32m1_t v34;
  v34 = v23;
  vfloat32m1_t v35;
  v35 = v26;
  vfloat32m1_t v36;
  v36 = v29;
  vfloat32m1_t v37;
  v37 = v32;
  for (size_t i38 = v11; i38 < v1; i38 += v12) {
    size_t v39 = i38 * v4;
    size_t v40 = 5;
    float* v41 = &v3[v39];
    vfloat32m1_t v42 = __riscv_vle32_v_f32m1(v41, v40);
    size_t v43 = i38 * v6;
    size_t v44 = v43 + v12;
    size_t v45 = v43 + v13;
    size_t v46 = v43 + v14;
    size_t v47 = v43 + v15;
    vfloat32m1_t v48 = v33;
    float v49 = v5[v43];
    vfloat32m1_t v50 = __riscv_vfmacc_vf_f32m1(v48, v49, v42, v40);
    vfloat32m1_t v51 = v34;
    float v52 = v5[v44];
    vfloat32m1_t v53 = __riscv_vfmacc_vf_f32m1(v51, v52, v42, v40);
    vfloat32m1_t v54 = v35;
    float v55 = v5[v45];
    vfloat32m1_t v56 = __riscv_vfmacc_vf_f32m1(v54, v55, v42, v40);
    vfloat32m1_t v57 = v36;
    float v58 = v5[v46];
    vfloat32m1_t v59 = __riscv_vfmacc_vf_f32m1(v57, v58, v42, v40);
    vfloat32m1_t v60 = v37;
    float v61 = v5[v47];
    vfloat32m1_t v62 = __riscv_vfmacc_vf_f32m1(v60, v61, v42, v40);
    v33 = v50;
    v34 = v53;
    v35 = v56;
    v36 = v59;
    v37 = v62;
  }
  vfloat32m1_t v63 = v33;
  vfloat32m1_t v64 = v34;
  vfloat32m1_t v65 = v35;
  vfloat32m1_t v66 = v36;
  vfloat32m1_t v67 = v37;
  float* v68 = &v8[v11];
  __riscv_vse32_v_f32m1(v68, v63, v16);
  float* v69 = &v8[v9];
  __riscv_vse32_v_f32m1(v69, v64, v16);
  size_t v70 = v13 * v9;
  float* v71 = &v8[v70];
  __riscv_vse32_v_f32m1(v71, v65, v16);
  size_t v72 = v14 * v9;
  float* v73 = &v8[v72];
  __riscv_vse32_v_f32m1(v73, v66, v16);
  size_t v74 = v15 * v9;
  float* v75 = &v8[v74];
  __riscv_vse32_v_f32m1(v75, v67, v16);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_5x6_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 5
//   NR         : 6
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_5x6_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v24 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v25 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v26;
  v26 = v20;
  vfloat32m1_t v27;
  v27 = v21;
  vfloat32m1_t v28;
  v28 = v22;
  vfloat32m1_t v29;
  v29 = v23;
  vfloat32m1_t v30;
  v30 = v24;
  vfloat32m1_t v31;
  v31 = v25;
  for (size_t i32 = v11; i32 < v1; i32 += v12) {
    size_t v33 = i32 * v4;
    size_t v34 = 5;
    float* v35 = &v3[v33];
    vfloat32m1_t v36 = __riscv_vle32_v_f32m1(v35, v34);
    size_t v37 = i32 * v6;
    size_t v38 = v37 + v12;
    size_t v39 = v37 + v13;
    size_t v40 = v37 + v14;
    size_t v41 = v37 + v15;
    size_t v42 = v37 + v16;
    vfloat32m1_t v43 = v26;
    float v44 = v5[v37];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v36, v34);
    vfloat32m1_t v46 = v27;
    float v47 = v5[v38];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v36, v34);
    vfloat32m1_t v49 = v28;
    float v50 = v5[v39];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v36, v34);
    vfloat32m1_t v52 = v29;
    float v53 = v5[v40];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v36, v34);
    vfloat32m1_t v55 = v30;
    float v56 = v5[v41];
    vfloat32m1_t v57 = __riscv_vfmacc_vf_f32m1(v55, v56, v36, v34);
    vfloat32m1_t v58 = v31;
    float v59 = v5[v42];
    vfloat32m1_t v60 = __riscv_vfmacc_vf_f32m1(v58, v59, v36, v34);
    v26 = v45;
    v27 = v48;
    v28 = v51;
    v29 = v54;
    v30 = v57;
    v31 = v60;
  }
  vfloat32m1_t v61 = v26;
  vfloat32m1_t v62 = v27;
  vfloat32m1_t v63 = v28;
  vfloat32m1_t v64 = v29;
  vfloat32m1_t v65 = v30;
  vfloat32m1_t v66 = v31;
  float* v67 = &v8[v11];
  __riscv_vse32_v_f32m1(v67, v61, v16);
  float* v68 = &v8[v9];
  __riscv_vse32_v_f32m1(v68, v62, v16);
  size_t v69 = v13 * v9;
  float* v70 = &v8[v69];
  __riscv_vse32_v_f32m1(v70, v63, v16);
  size_t v71 = v14 * v9;
  float* v72 = &v8[v71];
  __riscv_vse32_v_f32m1(v72, v64, v16);
  size_t v73 = v15 * v9;
  float* v74 = &v8[v73];
  __riscv_vse32_v_f32m1(v74, v65, v16);
  size_t v75 = v16 * v9;
  float* v76 = &v8[v75];
  __riscv_vse32_v_f32m1(v76, v66, v16);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_5x6_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 5
//   NR         : 6
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_5x6_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v16);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v16);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v16);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v16);
  size_t v30 = v15 * v9;
  float* v31 = &v8[v30];
  vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v16);
  size_t v33 = v16 * v9;
  float* v34 = &v8[v33];
  vfloat32m1_t v35 = __riscv_vle32_v_f32m1(v34, v16);
  vfloat32m1_t v36;
  v36 = v21;
  vfloat32m1_t v37;
  v37 = v23;
  vfloat32m1_t v38;
  v38 = v26;
  vfloat32m1_t v39;
  v39 = v29;
  vfloat32m1_t v40;
  v40 = v32;
  vfloat32m1_t v41;
  v41 = v35;
  for (size_t i42 = v11; i42 < v1; i42 += v12) {
    size_t v43 = i42 * v4;
    size_t v44 = 5;
    float* v45 = &v3[v43];
    vfloat32m1_t v46 = __riscv_vle32_v_f32m1(v45, v44);
    size_t v47 = i42 * v6;
    size_t v48 = v47 + v12;
    size_t v49 = v47 + v13;
    size_t v50 = v47 + v14;
    size_t v51 = v47 + v15;
    size_t v52 = v47 + v16;
    vfloat32m1_t v53 = v36;
    float v54 = v5[v47];
    vfloat32m1_t v55 = __riscv_vfmacc_vf_f32m1(v53, v54, v46, v44);
    vfloat32m1_t v56 = v37;
    float v57 = v5[v48];
    vfloat32m1_t v58 = __riscv_vfmacc_vf_f32m1(v56, v57, v46, v44);
    vfloat32m1_t v59 = v38;
    float v60 = v5[v49];
    vfloat32m1_t v61 = __riscv_vfmacc_vf_f32m1(v59, v60, v46, v44);
    vfloat32m1_t v62 = v39;
    float v63 = v5[v50];
    vfloat32m1_t v64 = __riscv_vfmacc_vf_f32m1(v62, v63, v46, v44);
    vfloat32m1_t v65 = v40;
    float v66 = v5[v51];
    vfloat32m1_t v67 = __riscv_vfmacc_vf_f32m1(v65, v66, v46, v44);
    vfloat32m1_t v68 = v41;
    float v69 = v5[v52];
    vfloat32m1_t v70 = __riscv_vfmacc_vf_f32m1(v68, v69, v46, v44);
    v36 = v55;
    v37 = v58;
    v38 = v61;
    v39 = v64;
    v40 = v67;
    v41 = v70;
  }
  vfloat32m1_t v71 = v36;
  vfloat32m1_t v72 = v37;
  vfloat32m1_t v73 = v38;
  vfloat32m1_t v74 = v39;
  vfloat32m1_t v75 = v40;
  vfloat32m1_t v76 = v41;
  float* v77 = &v8[v11];
  __riscv_vse32_v_f32m1(v77, v71, v16);
  float* v78 = &v8[v9];
  __riscv_vse32_v_f32m1(v78, v72, v16);
  size_t v79 = v13 * v9;
  float* v80 = &v8[v79];
  __riscv_vse32_v_f32m1(v80, v73, v16);
  size_t v81 = v14 * v9;
  float* v82 = &v8[v81];
  __riscv_vse32_v_f32m1(v82, v74, v16);
  size_t v83 = v15 * v9;
  float* v84 = &v8[v83];
  __riscv_vse32_v_f32m1(v84, v75, v16);
  size_t v85 = v16 * v9;
  float* v86 = &v8[v85];
  __riscv_vse32_v_f32m1(v86, v76, v16);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_5x7_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 5
//   NR         : 7
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_5x7_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v24 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v25 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v26 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v27;
  v27 = v20;
  vfloat32m1_t v28;
  v28 = v21;
  vfloat32m1_t v29;
  v29 = v22;
  vfloat32m1_t v30;
  v30 = v23;
  vfloat32m1_t v31;
  v31 = v24;
  vfloat32m1_t v32;
  v32 = v25;
  vfloat32m1_t v33;
  v33 = v26;
  for (size_t i34 = v11; i34 < v1; i34 += v12) {
    size_t v35 = i34 * v4;
    size_t v36 = 5;
    float* v37 = &v3[v35];
    vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v36);
    size_t v39 = i34 * v6;
    size_t v40 = v39 + v12;
    size_t v41 = v39 + v13;
    size_t v42 = v39 + v14;
    size_t v43 = v39 + v15;
    size_t v44 = v39 + v16;
    size_t v45 = v39 + v17;
    vfloat32m1_t v46 = v27;
    float v47 = v5[v39];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v38, v36);
    vfloat32m1_t v49 = v28;
    float v50 = v5[v40];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v38, v36);
    vfloat32m1_t v52 = v29;
    float v53 = v5[v41];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v38, v36);
    vfloat32m1_t v55 = v30;
    float v56 = v5[v42];
    vfloat32m1_t v57 = __riscv_vfmacc_vf_f32m1(v55, v56, v38, v36);
    vfloat32m1_t v58 = v31;
    float v59 = v5[v43];
    vfloat32m1_t v60 = __riscv_vfmacc_vf_f32m1(v58, v59, v38, v36);
    vfloat32m1_t v61 = v32;
    float v62 = v5[v44];
    vfloat32m1_t v63 = __riscv_vfmacc_vf_f32m1(v61, v62, v38, v36);
    vfloat32m1_t v64 = v33;
    float v65 = v5[v45];
    vfloat32m1_t v66 = __riscv_vfmacc_vf_f32m1(v64, v65, v38, v36);
    v27 = v48;
    v28 = v51;
    v29 = v54;
    v30 = v57;
    v31 = v60;
    v32 = v63;
    v33 = v66;
  }
  vfloat32m1_t v67 = v27;
  vfloat32m1_t v68 = v28;
  vfloat32m1_t v69 = v29;
  vfloat32m1_t v70 = v30;
  vfloat32m1_t v71 = v31;
  vfloat32m1_t v72 = v32;
  vfloat32m1_t v73 = v33;
  float* v74 = &v8[v11];
  __riscv_vse32_v_f32m1(v74, v67, v16);
  float* v75 = &v8[v9];
  __riscv_vse32_v_f32m1(v75, v68, v16);
  size_t v76 = v13 * v9;
  float* v77 = &v8[v76];
  __riscv_vse32_v_f32m1(v77, v69, v16);
  size_t v78 = v14 * v9;
  float* v79 = &v8[v78];
  __riscv_vse32_v_f32m1(v79, v70, v16);
  size_t v80 = v15 * v9;
  float* v81 = &v8[v80];
  __riscv_vse32_v_f32m1(v81, v71, v16);
  size_t v82 = v16 * v9;
  float* v83 = &v8[v82];
  __riscv_vse32_v_f32m1(v83, v72, v16);
  size_t v84 = v17 * v9;
  float* v85 = &v8[v84];
  __riscv_vse32_v_f32m1(v85, v73, v16);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_5x7_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 5
//   NR         : 7
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_5x7_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v16);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v16);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v16);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v16);
  size_t v30 = v15 * v9;
  float* v31 = &v8[v30];
  vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v16);
  size_t v33 = v16 * v9;
  float* v34 = &v8[v33];
  vfloat32m1_t v35 = __riscv_vle32_v_f32m1(v34, v16);
  size_t v36 = v17 * v9;
  float* v37 = &v8[v36];
  vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v16);
  vfloat32m1_t v39;
  v39 = v21;
  vfloat32m1_t v40;
  v40 = v23;
  vfloat32m1_t v41;
  v41 = v26;
  vfloat32m1_t v42;
  v42 = v29;
  vfloat32m1_t v43;
  v43 = v32;
  vfloat32m1_t v44;
  v44 = v35;
  vfloat32m1_t v45;
  v45 = v38;
  for (size_t i46 = v11; i46 < v1; i46 += v12) {
    size_t v47 = i46 * v4;
    size_t v48 = 5;
    float* v49 = &v3[v47];
    vfloat32m1_t v50 = __riscv_vle32_v_f32m1(v49, v48);
    size_t v51 = i46 * v6;
    size_t v52 = v51 + v12;
    size_t v53 = v51 + v13;
    size_t v54 = v51 + v14;
    size_t v55 = v51 + v15;
    size_t v56 = v51 + v16;
    size_t v57 = v51 + v17;
    vfloat32m1_t v58 = v39;
    float v59 = v5[v51];
    vfloat32m1_t v60 = __riscv_vfmacc_vf_f32m1(v58, v59, v50, v48);
    vfloat32m1_t v61 = v40;
    float v62 = v5[v52];
    vfloat32m1_t v63 = __riscv_vfmacc_vf_f32m1(v61, v62, v50, v48);
    vfloat32m1_t v64 = v41;
    float v65 = v5[v53];
    vfloat32m1_t v66 = __riscv_vfmacc_vf_f32m1(v64, v65, v50, v48);
    vfloat32m1_t v67 = v42;
    float v68 = v5[v54];
    vfloat32m1_t v69 = __riscv_vfmacc_vf_f32m1(v67, v68, v50, v48);
    vfloat32m1_t v70 = v43;
    float v71 = v5[v55];
    vfloat32m1_t v72 = __riscv_vfmacc_vf_f32m1(v70, v71, v50, v48);
    vfloat32m1_t v73 = v44;
    float v74 = v5[v56];
    vfloat32m1_t v75 = __riscv_vfmacc_vf_f32m1(v73, v74, v50, v48);
    vfloat32m1_t v76 = v45;
    float v77 = v5[v57];
    vfloat32m1_t v78 = __riscv_vfmacc_vf_f32m1(v76, v77, v50, v48);
    v39 = v60;
    v40 = v63;
    v41 = v66;
    v42 = v69;
    v43 = v72;
    v44 = v75;
    v45 = v78;
  }
  vfloat32m1_t v79 = v39;
  vfloat32m1_t v80 = v40;
  vfloat32m1_t v81 = v41;
  vfloat32m1_t v82 = v42;
  vfloat32m1_t v83 = v43;
  vfloat32m1_t v84 = v44;
  vfloat32m1_t v85 = v45;
  float* v86 = &v8[v11];
  __riscv_vse32_v_f32m1(v86, v79, v16);
  float* v87 = &v8[v9];
  __riscv_vse32_v_f32m1(v87, v80, v16);
  size_t v88 = v13 * v9;
  float* v89 = &v8[v88];
  __riscv_vse32_v_f32m1(v89, v81, v16);
  size_t v90 = v14 * v9;
  float* v91 = &v8[v90];
  __riscv_vse32_v_f32m1(v91, v82, v16);
  size_t v92 = v15 * v9;
  float* v93 = &v8[v92];
  __riscv_vse32_v_f32m1(v93, v83, v16);
  size_t v94 = v16 * v9;
  float* v95 = &v8[v94];
  __riscv_vse32_v_f32m1(v95, v84, v16);
  size_t v96 = v17 * v9;
  float* v97 = &v8[v96];
  __riscv_vse32_v_f32m1(v97, v85, v16);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_5x8_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 5
//   NR         : 8
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_5x8_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v24 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v25 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v26 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v27 = __riscv_vfmv_v_f_f32m1(v10, v16);
  vfloat32m1_t v28;
  v28 = v20;
  vfloat32m1_t v29;
  v29 = v21;
  vfloat32m1_t v30;
  v30 = v22;
  vfloat32m1_t v31;
  v31 = v23;
  vfloat32m1_t v32;
  v32 = v24;
  vfloat32m1_t v33;
  v33 = v25;
  vfloat32m1_t v34;
  v34 = v26;
  vfloat32m1_t v35;
  v35 = v27;
  for (size_t i36 = v11; i36 < v1; i36 += v12) {
    size_t v37 = i36 * v4;
    size_t v38 = 5;
    float* v39 = &v3[v37];
    vfloat32m1_t v40 = __riscv_vle32_v_f32m1(v39, v38);
    size_t v41 = i36 * v6;
    size_t v42 = v41 + v12;
    size_t v43 = v41 + v13;
    size_t v44 = v41 + v14;
    size_t v45 = v41 + v15;
    size_t v46 = v41 + v16;
    size_t v47 = v41 + v17;
    size_t v48 = v41 + v18;
    vfloat32m1_t v49 = v28;
    float v50 = v5[v41];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v40, v38);
    vfloat32m1_t v52 = v29;
    float v53 = v5[v42];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v40, v38);
    vfloat32m1_t v55 = v30;
    float v56 = v5[v43];
    vfloat32m1_t v57 = __riscv_vfmacc_vf_f32m1(v55, v56, v40, v38);
    vfloat32m1_t v58 = v31;
    float v59 = v5[v44];
    vfloat32m1_t v60 = __riscv_vfmacc_vf_f32m1(v58, v59, v40, v38);
    vfloat32m1_t v61 = v32;
    float v62 = v5[v45];
    vfloat32m1_t v63 = __riscv_vfmacc_vf_f32m1(v61, v62, v40, v38);
    vfloat32m1_t v64 = v33;
    float v65 = v5[v46];
    vfloat32m1_t v66 = __riscv_vfmacc_vf_f32m1(v64, v65, v40, v38);
    vfloat32m1_t v67 = v34;
    float v68 = v5[v47];
    vfloat32m1_t v69 = __riscv_vfmacc_vf_f32m1(v67, v68, v40, v38);
    vfloat32m1_t v70 = v35;
    float v71 = v5[v48];
    vfloat32m1_t v72 = __riscv_vfmacc_vf_f32m1(v70, v71, v40, v38);
    v28 = v51;
    v29 = v54;
    v30 = v57;
    v31 = v60;
    v32 = v63;
    v33 = v66;
    v34 = v69;
    v35 = v72;
  }
  vfloat32m1_t v73 = v28;
  vfloat32m1_t v74 = v29;
  vfloat32m1_t v75 = v30;
  vfloat32m1_t v76 = v31;
  vfloat32m1_t v77 = v32;
  vfloat32m1_t v78 = v33;
  vfloat32m1_t v79 = v34;
  vfloat32m1_t v80 = v35;
  float* v81 = &v8[v11];
  __riscv_vse32_v_f32m1(v81, v73, v16);
  float* v82 = &v8[v9];
  __riscv_vse32_v_f32m1(v82, v74, v16);
  size_t v83 = v13 * v9;
  float* v84 = &v8[v83];
  __riscv_vse32_v_f32m1(v84, v75, v16);
  size_t v85 = v14 * v9;
  float* v86 = &v8[v85];
  __riscv_vse32_v_f32m1(v86, v76, v16);
  size_t v87 = v15 * v9;
  float* v88 = &v8[v87];
  __riscv_vse32_v_f32m1(v88, v77, v16);
  size_t v89 = v16 * v9;
  float* v90 = &v8[v89];
  __riscv_vse32_v_f32m1(v90, v78, v16);
  size_t v91 = v17 * v9;
  float* v92 = &v8[v91];
  __riscv_vse32_v_f32m1(v92, v79, v16);
  size_t v93 = v18 * v9;
  float* v94 = &v8[v93];
  __riscv_vse32_v_f32m1(v94, v80, v16);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_5x8_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 5
//   NR         : 8
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_5x8_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v16);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v16);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v16);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v16);
  size_t v30 = v15 * v9;
  float* v31 = &v8[v30];
  vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v16);
  size_t v33 = v16 * v9;
  float* v34 = &v8[v33];
  vfloat32m1_t v35 = __riscv_vle32_v_f32m1(v34, v16);
  size_t v36 = v17 * v9;
  float* v37 = &v8[v36];
  vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v16);
  size_t v39 = v18 * v9;
  float* v40 = &v8[v39];
  vfloat32m1_t v41 = __riscv_vle32_v_f32m1(v40, v16);
  vfloat32m1_t v42;
  v42 = v21;
  vfloat32m1_t v43;
  v43 = v23;
  vfloat32m1_t v44;
  v44 = v26;
  vfloat32m1_t v45;
  v45 = v29;
  vfloat32m1_t v46;
  v46 = v32;
  vfloat32m1_t v47;
  v47 = v35;
  vfloat32m1_t v48;
  v48 = v38;
  vfloat32m1_t v49;
  v49 = v41;
  for (size_t i50 = v11; i50 < v1; i50 += v12) {
    size_t v51 = i50 * v4;
    size_t v52 = 5;
    float* v53 = &v3[v51];
    vfloat32m1_t v54 = __riscv_vle32_v_f32m1(v53, v52);
    size_t v55 = i50 * v6;
    size_t v56 = v55 + v12;
    size_t v57 = v55 + v13;
    size_t v58 = v55 + v14;
    size_t v59 = v55 + v15;
    size_t v60 = v55 + v16;
    size_t v61 = v55 + v17;
    size_t v62 = v55 + v18;
    vfloat32m1_t v63 = v42;
    float v64 = v5[v55];
    vfloat32m1_t v65 = __riscv_vfmacc_vf_f32m1(v63, v64, v54, v52);
    vfloat32m1_t v66 = v43;
    float v67 = v5[v56];
    vfloat32m1_t v68 = __riscv_vfmacc_vf_f32m1(v66, v67, v54, v52);
    vfloat32m1_t v69 = v44;
    float v70 = v5[v57];
    vfloat32m1_t v71 = __riscv_vfmacc_vf_f32m1(v69, v70, v54, v52);
    vfloat32m1_t v72 = v45;
    float v73 = v5[v58];
    vfloat32m1_t v74 = __riscv_vfmacc_vf_f32m1(v72, v73, v54, v52);
    vfloat32m1_t v75 = v46;
    float v76 = v5[v59];
    vfloat32m1_t v77 = __riscv_vfmacc_vf_f32m1(v75, v76, v54, v52);
    vfloat32m1_t v78 = v47;
    float v79 = v5[v60];
    vfloat32m1_t v80 = __riscv_vfmacc_vf_f32m1(v78, v79, v54, v52);
    vfloat32m1_t v81 = v48;
    float v82 = v5[v61];
    vfloat32m1_t v83 = __riscv_vfmacc_vf_f32m1(v81, v82, v54, v52);
    vfloat32m1_t v84 = v49;
    float v85 = v5[v62];
    vfloat32m1_t v86 = __riscv_vfmacc_vf_f32m1(v84, v85, v54, v52);
    v42 = v65;
    v43 = v68;
    v44 = v71;
    v45 = v74;
    v46 = v77;
    v47 = v80;
    v48 = v83;
    v49 = v86;
  }
  vfloat32m1_t v87 = v42;
  vfloat32m1_t v88 = v43;
  vfloat32m1_t v89 = v44;
  vfloat32m1_t v90 = v45;
  vfloat32m1_t v91 = v46;
  vfloat32m1_t v92 = v47;
  vfloat32m1_t v93 = v48;
  vfloat32m1_t v94 = v49;
  float* v95 = &v8[v11];
  __riscv_vse32_v_f32m1(v95, v87, v16);
  float* v96 = &v8[v9];
  __riscv_vse32_v_f32m1(v96, v88, v16);
  size_t v97 = v13 * v9;
  float* v98 = &v8[v97];
  __riscv_vse32_v_f32m1(v98, v89, v16);
  size_t v99 = v14 * v9;
  float* v100 = &v8[v99];
  __riscv_vse32_v_f32m1(v100, v90, v16);
  size_t v101 = v15 * v9;
  float* v102 = &v8[v101];
  __riscv_vse32_v_f32m1(v102, v91, v16);
  size_t v103 = v16 * v9;
  float* v104 = &v8[v103];
  __riscv_vse32_v_f32m1(v104, v92, v16);
  size_t v105 = v17 * v9;
  float* v106 = &v8[v105];
  __riscv_vse32_v_f32m1(v106, v93, v16);
  size_t v107 = v18 * v9;
  float* v108 = &v8[v107];
  __riscv_vse32_v_f32m1(v108, v94, v16);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_6x1_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 6
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_6x1_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v21;
  v21 = v20;
  for (size_t i22 = v11; i22 < v1; i22 += v12) {
    size_t v23 = i22 * v4;
    size_t v24 = 6;
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
  __riscv_vse32_v_f32m1(v32, v31, v17);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_6x1_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 6
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_6x1_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v17);
  vfloat32m1_t v22;
  v22 = v21;
  for (size_t i23 = v11; i23 < v1; i23 += v12) {
    size_t v24 = i23 * v4;
    size_t v25 = 6;
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
  __riscv_vse32_v_f32m1(v33, v32, v17);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_6x2_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 6
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_6x2_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v22;
  v22 = v20;
  vfloat32m1_t v23;
  v23 = v21;
  for (size_t i24 = v11; i24 < v1; i24 += v12) {
    size_t v25 = i24 * v4;
    size_t v26 = 6;
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
  __riscv_vse32_v_f32m1(v39, v37, v17);
  float* v40 = &v8[v9];
  __riscv_vse32_v_f32m1(v40, v38, v17);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_6x2_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 6
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_6x2_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v17);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v17);
  vfloat32m1_t v24;
  v24 = v21;
  vfloat32m1_t v25;
  v25 = v23;
  for (size_t i26 = v11; i26 < v1; i26 += v12) {
    size_t v27 = i26 * v4;
    size_t v28 = 6;
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
  __riscv_vse32_v_f32m1(v41, v39, v17);
  float* v42 = &v8[v9];
  __riscv_vse32_v_f32m1(v42, v40, v17);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_6x3_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 6
//   NR         : 3
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_6x3_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v23;
  v23 = v20;
  vfloat32m1_t v24;
  v24 = v21;
  vfloat32m1_t v25;
  v25 = v22;
  for (size_t i26 = v11; i26 < v1; i26 += v12) {
    size_t v27 = i26 * v4;
    size_t v28 = 6;
    float* v29 = &v3[v27];
    vfloat32m1_t v30 = __riscv_vle32_v_f32m1(v29, v28);
    size_t v31 = i26 * v6;
    size_t v32 = v31 + v12;
    size_t v33 = v31 + v13;
    vfloat32m1_t v34 = v23;
    float v35 = v5[v31];
    vfloat32m1_t v36 = __riscv_vfmacc_vf_f32m1(v34, v35, v30, v28);
    vfloat32m1_t v37 = v24;
    float v38 = v5[v32];
    vfloat32m1_t v39 = __riscv_vfmacc_vf_f32m1(v37, v38, v30, v28);
    vfloat32m1_t v40 = v25;
    float v41 = v5[v33];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v30, v28);
    v23 = v36;
    v24 = v39;
    v25 = v42;
  }
  vfloat32m1_t v43 = v23;
  vfloat32m1_t v44 = v24;
  vfloat32m1_t v45 = v25;
  float* v46 = &v8[v11];
  __riscv_vse32_v_f32m1(v46, v43, v17);
  float* v47 = &v8[v9];
  __riscv_vse32_v_f32m1(v47, v44, v17);
  size_t v48 = v13 * v9;
  float* v49 = &v8[v48];
  __riscv_vse32_v_f32m1(v49, v45, v17);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_6x3_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 6
//   NR         : 3
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_6x3_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v17);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v17);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v17);
  vfloat32m1_t v27;
  v27 = v21;
  vfloat32m1_t v28;
  v28 = v23;
  vfloat32m1_t v29;
  v29 = v26;
  for (size_t i30 = v11; i30 < v1; i30 += v12) {
    size_t v31 = i30 * v4;
    size_t v32 = 6;
    float* v33 = &v3[v31];
    vfloat32m1_t v34 = __riscv_vle32_v_f32m1(v33, v32);
    size_t v35 = i30 * v6;
    size_t v36 = v35 + v12;
    size_t v37 = v35 + v13;
    vfloat32m1_t v38 = v27;
    float v39 = v5[v35];
    vfloat32m1_t v40 = __riscv_vfmacc_vf_f32m1(v38, v39, v34, v32);
    vfloat32m1_t v41 = v28;
    float v42 = v5[v36];
    vfloat32m1_t v43 = __riscv_vfmacc_vf_f32m1(v41, v42, v34, v32);
    vfloat32m1_t v44 = v29;
    float v45 = v5[v37];
    vfloat32m1_t v46 = __riscv_vfmacc_vf_f32m1(v44, v45, v34, v32);
    v27 = v40;
    v28 = v43;
    v29 = v46;
  }
  vfloat32m1_t v47 = v27;
  vfloat32m1_t v48 = v28;
  vfloat32m1_t v49 = v29;
  float* v50 = &v8[v11];
  __riscv_vse32_v_f32m1(v50, v47, v17);
  float* v51 = &v8[v9];
  __riscv_vse32_v_f32m1(v51, v48, v17);
  size_t v52 = v13 * v9;
  float* v53 = &v8[v52];
  __riscv_vse32_v_f32m1(v53, v49, v17);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_6x4_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 6
//   NR         : 4
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_6x4_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v24;
  v24 = v20;
  vfloat32m1_t v25;
  v25 = v21;
  vfloat32m1_t v26;
  v26 = v22;
  vfloat32m1_t v27;
  v27 = v23;
  for (size_t i28 = v11; i28 < v1; i28 += v12) {
    size_t v29 = i28 * v4;
    size_t v30 = 6;
    float* v31 = &v3[v29];
    vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v30);
    size_t v33 = i28 * v6;
    size_t v34 = v33 + v12;
    size_t v35 = v33 + v13;
    size_t v36 = v33 + v14;
    vfloat32m1_t v37 = v24;
    float v38 = v5[v33];
    vfloat32m1_t v39 = __riscv_vfmacc_vf_f32m1(v37, v38, v32, v30);
    vfloat32m1_t v40 = v25;
    float v41 = v5[v34];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v32, v30);
    vfloat32m1_t v43 = v26;
    float v44 = v5[v35];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v32, v30);
    vfloat32m1_t v46 = v27;
    float v47 = v5[v36];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v32, v30);
    v24 = v39;
    v25 = v42;
    v26 = v45;
    v27 = v48;
  }
  vfloat32m1_t v49 = v24;
  vfloat32m1_t v50 = v25;
  vfloat32m1_t v51 = v26;
  vfloat32m1_t v52 = v27;
  float* v53 = &v8[v11];
  __riscv_vse32_v_f32m1(v53, v49, v17);
  float* v54 = &v8[v9];
  __riscv_vse32_v_f32m1(v54, v50, v17);
  size_t v55 = v13 * v9;
  float* v56 = &v8[v55];
  __riscv_vse32_v_f32m1(v56, v51, v17);
  size_t v57 = v14 * v9;
  float* v58 = &v8[v57];
  __riscv_vse32_v_f32m1(v58, v52, v17);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_6x4_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 6
//   NR         : 4
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_6x4_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v17);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v17);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v17);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v17);
  vfloat32m1_t v30;
  v30 = v21;
  vfloat32m1_t v31;
  v31 = v23;
  vfloat32m1_t v32;
  v32 = v26;
  vfloat32m1_t v33;
  v33 = v29;
  for (size_t i34 = v11; i34 < v1; i34 += v12) {
    size_t v35 = i34 * v4;
    size_t v36 = 6;
    float* v37 = &v3[v35];
    vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v36);
    size_t v39 = i34 * v6;
    size_t v40 = v39 + v12;
    size_t v41 = v39 + v13;
    size_t v42 = v39 + v14;
    vfloat32m1_t v43 = v30;
    float v44 = v5[v39];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v38, v36);
    vfloat32m1_t v46 = v31;
    float v47 = v5[v40];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v38, v36);
    vfloat32m1_t v49 = v32;
    float v50 = v5[v41];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v38, v36);
    vfloat32m1_t v52 = v33;
    float v53 = v5[v42];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v38, v36);
    v30 = v45;
    v31 = v48;
    v32 = v51;
    v33 = v54;
  }
  vfloat32m1_t v55 = v30;
  vfloat32m1_t v56 = v31;
  vfloat32m1_t v57 = v32;
  vfloat32m1_t v58 = v33;
  float* v59 = &v8[v11];
  __riscv_vse32_v_f32m1(v59, v55, v17);
  float* v60 = &v8[v9];
  __riscv_vse32_v_f32m1(v60, v56, v17);
  size_t v61 = v13 * v9;
  float* v62 = &v8[v61];
  __riscv_vse32_v_f32m1(v62, v57, v17);
  size_t v63 = v14 * v9;
  float* v64 = &v8[v63];
  __riscv_vse32_v_f32m1(v64, v58, v17);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_6x5_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 6
//   NR         : 5
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_6x5_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v24 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v25;
  v25 = v20;
  vfloat32m1_t v26;
  v26 = v21;
  vfloat32m1_t v27;
  v27 = v22;
  vfloat32m1_t v28;
  v28 = v23;
  vfloat32m1_t v29;
  v29 = v24;
  for (size_t i30 = v11; i30 < v1; i30 += v12) {
    size_t v31 = i30 * v4;
    size_t v32 = 6;
    float* v33 = &v3[v31];
    vfloat32m1_t v34 = __riscv_vle32_v_f32m1(v33, v32);
    size_t v35 = i30 * v6;
    size_t v36 = v35 + v12;
    size_t v37 = v35 + v13;
    size_t v38 = v35 + v14;
    size_t v39 = v35 + v15;
    vfloat32m1_t v40 = v25;
    float v41 = v5[v35];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v34, v32);
    vfloat32m1_t v43 = v26;
    float v44 = v5[v36];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v34, v32);
    vfloat32m1_t v46 = v27;
    float v47 = v5[v37];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v34, v32);
    vfloat32m1_t v49 = v28;
    float v50 = v5[v38];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v34, v32);
    vfloat32m1_t v52 = v29;
    float v53 = v5[v39];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v34, v32);
    v25 = v42;
    v26 = v45;
    v27 = v48;
    v28 = v51;
    v29 = v54;
  }
  vfloat32m1_t v55 = v25;
  vfloat32m1_t v56 = v26;
  vfloat32m1_t v57 = v27;
  vfloat32m1_t v58 = v28;
  vfloat32m1_t v59 = v29;
  float* v60 = &v8[v11];
  __riscv_vse32_v_f32m1(v60, v55, v17);
  float* v61 = &v8[v9];
  __riscv_vse32_v_f32m1(v61, v56, v17);
  size_t v62 = v13 * v9;
  float* v63 = &v8[v62];
  __riscv_vse32_v_f32m1(v63, v57, v17);
  size_t v64 = v14 * v9;
  float* v65 = &v8[v64];
  __riscv_vse32_v_f32m1(v65, v58, v17);
  size_t v66 = v15 * v9;
  float* v67 = &v8[v66];
  __riscv_vse32_v_f32m1(v67, v59, v17);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_6x5_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 6
//   NR         : 5
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_6x5_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v17);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v17);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v17);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v17);
  size_t v30 = v15 * v9;
  float* v31 = &v8[v30];
  vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v17);
  vfloat32m1_t v33;
  v33 = v21;
  vfloat32m1_t v34;
  v34 = v23;
  vfloat32m1_t v35;
  v35 = v26;
  vfloat32m1_t v36;
  v36 = v29;
  vfloat32m1_t v37;
  v37 = v32;
  for (size_t i38 = v11; i38 < v1; i38 += v12) {
    size_t v39 = i38 * v4;
    size_t v40 = 6;
    float* v41 = &v3[v39];
    vfloat32m1_t v42 = __riscv_vle32_v_f32m1(v41, v40);
    size_t v43 = i38 * v6;
    size_t v44 = v43 + v12;
    size_t v45 = v43 + v13;
    size_t v46 = v43 + v14;
    size_t v47 = v43 + v15;
    vfloat32m1_t v48 = v33;
    float v49 = v5[v43];
    vfloat32m1_t v50 = __riscv_vfmacc_vf_f32m1(v48, v49, v42, v40);
    vfloat32m1_t v51 = v34;
    float v52 = v5[v44];
    vfloat32m1_t v53 = __riscv_vfmacc_vf_f32m1(v51, v52, v42, v40);
    vfloat32m1_t v54 = v35;
    float v55 = v5[v45];
    vfloat32m1_t v56 = __riscv_vfmacc_vf_f32m1(v54, v55, v42, v40);
    vfloat32m1_t v57 = v36;
    float v58 = v5[v46];
    vfloat32m1_t v59 = __riscv_vfmacc_vf_f32m1(v57, v58, v42, v40);
    vfloat32m1_t v60 = v37;
    float v61 = v5[v47];
    vfloat32m1_t v62 = __riscv_vfmacc_vf_f32m1(v60, v61, v42, v40);
    v33 = v50;
    v34 = v53;
    v35 = v56;
    v36 = v59;
    v37 = v62;
  }
  vfloat32m1_t v63 = v33;
  vfloat32m1_t v64 = v34;
  vfloat32m1_t v65 = v35;
  vfloat32m1_t v66 = v36;
  vfloat32m1_t v67 = v37;
  float* v68 = &v8[v11];
  __riscv_vse32_v_f32m1(v68, v63, v17);
  float* v69 = &v8[v9];
  __riscv_vse32_v_f32m1(v69, v64, v17);
  size_t v70 = v13 * v9;
  float* v71 = &v8[v70];
  __riscv_vse32_v_f32m1(v71, v65, v17);
  size_t v72 = v14 * v9;
  float* v73 = &v8[v72];
  __riscv_vse32_v_f32m1(v73, v66, v17);
  size_t v74 = v15 * v9;
  float* v75 = &v8[v74];
  __riscv_vse32_v_f32m1(v75, v67, v17);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_6x6_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 6
//   NR         : 6
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_6x6_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v24 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v25 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v26;
  v26 = v20;
  vfloat32m1_t v27;
  v27 = v21;
  vfloat32m1_t v28;
  v28 = v22;
  vfloat32m1_t v29;
  v29 = v23;
  vfloat32m1_t v30;
  v30 = v24;
  vfloat32m1_t v31;
  v31 = v25;
  for (size_t i32 = v11; i32 < v1; i32 += v12) {
    size_t v33 = i32 * v4;
    size_t v34 = 6;
    float* v35 = &v3[v33];
    vfloat32m1_t v36 = __riscv_vle32_v_f32m1(v35, v34);
    size_t v37 = i32 * v6;
    size_t v38 = v37 + v12;
    size_t v39 = v37 + v13;
    size_t v40 = v37 + v14;
    size_t v41 = v37 + v15;
    size_t v42 = v37 + v16;
    vfloat32m1_t v43 = v26;
    float v44 = v5[v37];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v36, v34);
    vfloat32m1_t v46 = v27;
    float v47 = v5[v38];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v36, v34);
    vfloat32m1_t v49 = v28;
    float v50 = v5[v39];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v36, v34);
    vfloat32m1_t v52 = v29;
    float v53 = v5[v40];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v36, v34);
    vfloat32m1_t v55 = v30;
    float v56 = v5[v41];
    vfloat32m1_t v57 = __riscv_vfmacc_vf_f32m1(v55, v56, v36, v34);
    vfloat32m1_t v58 = v31;
    float v59 = v5[v42];
    vfloat32m1_t v60 = __riscv_vfmacc_vf_f32m1(v58, v59, v36, v34);
    v26 = v45;
    v27 = v48;
    v28 = v51;
    v29 = v54;
    v30 = v57;
    v31 = v60;
  }
  vfloat32m1_t v61 = v26;
  vfloat32m1_t v62 = v27;
  vfloat32m1_t v63 = v28;
  vfloat32m1_t v64 = v29;
  vfloat32m1_t v65 = v30;
  vfloat32m1_t v66 = v31;
  float* v67 = &v8[v11];
  __riscv_vse32_v_f32m1(v67, v61, v17);
  float* v68 = &v8[v9];
  __riscv_vse32_v_f32m1(v68, v62, v17);
  size_t v69 = v13 * v9;
  float* v70 = &v8[v69];
  __riscv_vse32_v_f32m1(v70, v63, v17);
  size_t v71 = v14 * v9;
  float* v72 = &v8[v71];
  __riscv_vse32_v_f32m1(v72, v64, v17);
  size_t v73 = v15 * v9;
  float* v74 = &v8[v73];
  __riscv_vse32_v_f32m1(v74, v65, v17);
  size_t v75 = v16 * v9;
  float* v76 = &v8[v75];
  __riscv_vse32_v_f32m1(v76, v66, v17);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_6x6_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 6
//   NR         : 6
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_6x6_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v17);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v17);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v17);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v17);
  size_t v30 = v15 * v9;
  float* v31 = &v8[v30];
  vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v17);
  size_t v33 = v16 * v9;
  float* v34 = &v8[v33];
  vfloat32m1_t v35 = __riscv_vle32_v_f32m1(v34, v17);
  vfloat32m1_t v36;
  v36 = v21;
  vfloat32m1_t v37;
  v37 = v23;
  vfloat32m1_t v38;
  v38 = v26;
  vfloat32m1_t v39;
  v39 = v29;
  vfloat32m1_t v40;
  v40 = v32;
  vfloat32m1_t v41;
  v41 = v35;
  for (size_t i42 = v11; i42 < v1; i42 += v12) {
    size_t v43 = i42 * v4;
    size_t v44 = 6;
    float* v45 = &v3[v43];
    vfloat32m1_t v46 = __riscv_vle32_v_f32m1(v45, v44);
    size_t v47 = i42 * v6;
    size_t v48 = v47 + v12;
    size_t v49 = v47 + v13;
    size_t v50 = v47 + v14;
    size_t v51 = v47 + v15;
    size_t v52 = v47 + v16;
    vfloat32m1_t v53 = v36;
    float v54 = v5[v47];
    vfloat32m1_t v55 = __riscv_vfmacc_vf_f32m1(v53, v54, v46, v44);
    vfloat32m1_t v56 = v37;
    float v57 = v5[v48];
    vfloat32m1_t v58 = __riscv_vfmacc_vf_f32m1(v56, v57, v46, v44);
    vfloat32m1_t v59 = v38;
    float v60 = v5[v49];
    vfloat32m1_t v61 = __riscv_vfmacc_vf_f32m1(v59, v60, v46, v44);
    vfloat32m1_t v62 = v39;
    float v63 = v5[v50];
    vfloat32m1_t v64 = __riscv_vfmacc_vf_f32m1(v62, v63, v46, v44);
    vfloat32m1_t v65 = v40;
    float v66 = v5[v51];
    vfloat32m1_t v67 = __riscv_vfmacc_vf_f32m1(v65, v66, v46, v44);
    vfloat32m1_t v68 = v41;
    float v69 = v5[v52];
    vfloat32m1_t v70 = __riscv_vfmacc_vf_f32m1(v68, v69, v46, v44);
    v36 = v55;
    v37 = v58;
    v38 = v61;
    v39 = v64;
    v40 = v67;
    v41 = v70;
  }
  vfloat32m1_t v71 = v36;
  vfloat32m1_t v72 = v37;
  vfloat32m1_t v73 = v38;
  vfloat32m1_t v74 = v39;
  vfloat32m1_t v75 = v40;
  vfloat32m1_t v76 = v41;
  float* v77 = &v8[v11];
  __riscv_vse32_v_f32m1(v77, v71, v17);
  float* v78 = &v8[v9];
  __riscv_vse32_v_f32m1(v78, v72, v17);
  size_t v79 = v13 * v9;
  float* v80 = &v8[v79];
  __riscv_vse32_v_f32m1(v80, v73, v17);
  size_t v81 = v14 * v9;
  float* v82 = &v8[v81];
  __riscv_vse32_v_f32m1(v82, v74, v17);
  size_t v83 = v15 * v9;
  float* v84 = &v8[v83];
  __riscv_vse32_v_f32m1(v84, v75, v17);
  size_t v85 = v16 * v9;
  float* v86 = &v8[v85];
  __riscv_vse32_v_f32m1(v86, v76, v17);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_6x7_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 6
//   NR         : 7
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_6x7_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v24 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v25 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v26 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v27;
  v27 = v20;
  vfloat32m1_t v28;
  v28 = v21;
  vfloat32m1_t v29;
  v29 = v22;
  vfloat32m1_t v30;
  v30 = v23;
  vfloat32m1_t v31;
  v31 = v24;
  vfloat32m1_t v32;
  v32 = v25;
  vfloat32m1_t v33;
  v33 = v26;
  for (size_t i34 = v11; i34 < v1; i34 += v12) {
    size_t v35 = i34 * v4;
    size_t v36 = 6;
    float* v37 = &v3[v35];
    vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v36);
    size_t v39 = i34 * v6;
    size_t v40 = v39 + v12;
    size_t v41 = v39 + v13;
    size_t v42 = v39 + v14;
    size_t v43 = v39 + v15;
    size_t v44 = v39 + v16;
    size_t v45 = v39 + v17;
    vfloat32m1_t v46 = v27;
    float v47 = v5[v39];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v38, v36);
    vfloat32m1_t v49 = v28;
    float v50 = v5[v40];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v38, v36);
    vfloat32m1_t v52 = v29;
    float v53 = v5[v41];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v38, v36);
    vfloat32m1_t v55 = v30;
    float v56 = v5[v42];
    vfloat32m1_t v57 = __riscv_vfmacc_vf_f32m1(v55, v56, v38, v36);
    vfloat32m1_t v58 = v31;
    float v59 = v5[v43];
    vfloat32m1_t v60 = __riscv_vfmacc_vf_f32m1(v58, v59, v38, v36);
    vfloat32m1_t v61 = v32;
    float v62 = v5[v44];
    vfloat32m1_t v63 = __riscv_vfmacc_vf_f32m1(v61, v62, v38, v36);
    vfloat32m1_t v64 = v33;
    float v65 = v5[v45];
    vfloat32m1_t v66 = __riscv_vfmacc_vf_f32m1(v64, v65, v38, v36);
    v27 = v48;
    v28 = v51;
    v29 = v54;
    v30 = v57;
    v31 = v60;
    v32 = v63;
    v33 = v66;
  }
  vfloat32m1_t v67 = v27;
  vfloat32m1_t v68 = v28;
  vfloat32m1_t v69 = v29;
  vfloat32m1_t v70 = v30;
  vfloat32m1_t v71 = v31;
  vfloat32m1_t v72 = v32;
  vfloat32m1_t v73 = v33;
  float* v74 = &v8[v11];
  __riscv_vse32_v_f32m1(v74, v67, v17);
  float* v75 = &v8[v9];
  __riscv_vse32_v_f32m1(v75, v68, v17);
  size_t v76 = v13 * v9;
  float* v77 = &v8[v76];
  __riscv_vse32_v_f32m1(v77, v69, v17);
  size_t v78 = v14 * v9;
  float* v79 = &v8[v78];
  __riscv_vse32_v_f32m1(v79, v70, v17);
  size_t v80 = v15 * v9;
  float* v81 = &v8[v80];
  __riscv_vse32_v_f32m1(v81, v71, v17);
  size_t v82 = v16 * v9;
  float* v83 = &v8[v82];
  __riscv_vse32_v_f32m1(v83, v72, v17);
  size_t v84 = v17 * v9;
  float* v85 = &v8[v84];
  __riscv_vse32_v_f32m1(v85, v73, v17);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_6x7_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 6
//   NR         : 7
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_6x7_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v17);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v17);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v17);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v17);
  size_t v30 = v15 * v9;
  float* v31 = &v8[v30];
  vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v17);
  size_t v33 = v16 * v9;
  float* v34 = &v8[v33];
  vfloat32m1_t v35 = __riscv_vle32_v_f32m1(v34, v17);
  size_t v36 = v17 * v9;
  float* v37 = &v8[v36];
  vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v17);
  vfloat32m1_t v39;
  v39 = v21;
  vfloat32m1_t v40;
  v40 = v23;
  vfloat32m1_t v41;
  v41 = v26;
  vfloat32m1_t v42;
  v42 = v29;
  vfloat32m1_t v43;
  v43 = v32;
  vfloat32m1_t v44;
  v44 = v35;
  vfloat32m1_t v45;
  v45 = v38;
  for (size_t i46 = v11; i46 < v1; i46 += v12) {
    size_t v47 = i46 * v4;
    size_t v48 = 6;
    float* v49 = &v3[v47];
    vfloat32m1_t v50 = __riscv_vle32_v_f32m1(v49, v48);
    size_t v51 = i46 * v6;
    size_t v52 = v51 + v12;
    size_t v53 = v51 + v13;
    size_t v54 = v51 + v14;
    size_t v55 = v51 + v15;
    size_t v56 = v51 + v16;
    size_t v57 = v51 + v17;
    vfloat32m1_t v58 = v39;
    float v59 = v5[v51];
    vfloat32m1_t v60 = __riscv_vfmacc_vf_f32m1(v58, v59, v50, v48);
    vfloat32m1_t v61 = v40;
    float v62 = v5[v52];
    vfloat32m1_t v63 = __riscv_vfmacc_vf_f32m1(v61, v62, v50, v48);
    vfloat32m1_t v64 = v41;
    float v65 = v5[v53];
    vfloat32m1_t v66 = __riscv_vfmacc_vf_f32m1(v64, v65, v50, v48);
    vfloat32m1_t v67 = v42;
    float v68 = v5[v54];
    vfloat32m1_t v69 = __riscv_vfmacc_vf_f32m1(v67, v68, v50, v48);
    vfloat32m1_t v70 = v43;
    float v71 = v5[v55];
    vfloat32m1_t v72 = __riscv_vfmacc_vf_f32m1(v70, v71, v50, v48);
    vfloat32m1_t v73 = v44;
    float v74 = v5[v56];
    vfloat32m1_t v75 = __riscv_vfmacc_vf_f32m1(v73, v74, v50, v48);
    vfloat32m1_t v76 = v45;
    float v77 = v5[v57];
    vfloat32m1_t v78 = __riscv_vfmacc_vf_f32m1(v76, v77, v50, v48);
    v39 = v60;
    v40 = v63;
    v41 = v66;
    v42 = v69;
    v43 = v72;
    v44 = v75;
    v45 = v78;
  }
  vfloat32m1_t v79 = v39;
  vfloat32m1_t v80 = v40;
  vfloat32m1_t v81 = v41;
  vfloat32m1_t v82 = v42;
  vfloat32m1_t v83 = v43;
  vfloat32m1_t v84 = v44;
  vfloat32m1_t v85 = v45;
  float* v86 = &v8[v11];
  __riscv_vse32_v_f32m1(v86, v79, v17);
  float* v87 = &v8[v9];
  __riscv_vse32_v_f32m1(v87, v80, v17);
  size_t v88 = v13 * v9;
  float* v89 = &v8[v88];
  __riscv_vse32_v_f32m1(v89, v81, v17);
  size_t v90 = v14 * v9;
  float* v91 = &v8[v90];
  __riscv_vse32_v_f32m1(v91, v82, v17);
  size_t v92 = v15 * v9;
  float* v93 = &v8[v92];
  __riscv_vse32_v_f32m1(v93, v83, v17);
  size_t v94 = v16 * v9;
  float* v95 = &v8[v94];
  __riscv_vse32_v_f32m1(v95, v84, v17);
  size_t v96 = v17 * v9;
  float* v97 = &v8[v96];
  __riscv_vse32_v_f32m1(v97, v85, v17);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_6x8_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 6
//   NR         : 8
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_6x8_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v24 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v25 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v26 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v27 = __riscv_vfmv_v_f_f32m1(v10, v17);
  vfloat32m1_t v28;
  v28 = v20;
  vfloat32m1_t v29;
  v29 = v21;
  vfloat32m1_t v30;
  v30 = v22;
  vfloat32m1_t v31;
  v31 = v23;
  vfloat32m1_t v32;
  v32 = v24;
  vfloat32m1_t v33;
  v33 = v25;
  vfloat32m1_t v34;
  v34 = v26;
  vfloat32m1_t v35;
  v35 = v27;
  for (size_t i36 = v11; i36 < v1; i36 += v12) {
    size_t v37 = i36 * v4;
    size_t v38 = 6;
    float* v39 = &v3[v37];
    vfloat32m1_t v40 = __riscv_vle32_v_f32m1(v39, v38);
    size_t v41 = i36 * v6;
    size_t v42 = v41 + v12;
    size_t v43 = v41 + v13;
    size_t v44 = v41 + v14;
    size_t v45 = v41 + v15;
    size_t v46 = v41 + v16;
    size_t v47 = v41 + v17;
    size_t v48 = v41 + v18;
    vfloat32m1_t v49 = v28;
    float v50 = v5[v41];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v40, v38);
    vfloat32m1_t v52 = v29;
    float v53 = v5[v42];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v40, v38);
    vfloat32m1_t v55 = v30;
    float v56 = v5[v43];
    vfloat32m1_t v57 = __riscv_vfmacc_vf_f32m1(v55, v56, v40, v38);
    vfloat32m1_t v58 = v31;
    float v59 = v5[v44];
    vfloat32m1_t v60 = __riscv_vfmacc_vf_f32m1(v58, v59, v40, v38);
    vfloat32m1_t v61 = v32;
    float v62 = v5[v45];
    vfloat32m1_t v63 = __riscv_vfmacc_vf_f32m1(v61, v62, v40, v38);
    vfloat32m1_t v64 = v33;
    float v65 = v5[v46];
    vfloat32m1_t v66 = __riscv_vfmacc_vf_f32m1(v64, v65, v40, v38);
    vfloat32m1_t v67 = v34;
    float v68 = v5[v47];
    vfloat32m1_t v69 = __riscv_vfmacc_vf_f32m1(v67, v68, v40, v38);
    vfloat32m1_t v70 = v35;
    float v71 = v5[v48];
    vfloat32m1_t v72 = __riscv_vfmacc_vf_f32m1(v70, v71, v40, v38);
    v28 = v51;
    v29 = v54;
    v30 = v57;
    v31 = v60;
    v32 = v63;
    v33 = v66;
    v34 = v69;
    v35 = v72;
  }
  vfloat32m1_t v73 = v28;
  vfloat32m1_t v74 = v29;
  vfloat32m1_t v75 = v30;
  vfloat32m1_t v76 = v31;
  vfloat32m1_t v77 = v32;
  vfloat32m1_t v78 = v33;
  vfloat32m1_t v79 = v34;
  vfloat32m1_t v80 = v35;
  float* v81 = &v8[v11];
  __riscv_vse32_v_f32m1(v81, v73, v17);
  float* v82 = &v8[v9];
  __riscv_vse32_v_f32m1(v82, v74, v17);
  size_t v83 = v13 * v9;
  float* v84 = &v8[v83];
  __riscv_vse32_v_f32m1(v84, v75, v17);
  size_t v85 = v14 * v9;
  float* v86 = &v8[v85];
  __riscv_vse32_v_f32m1(v86, v76, v17);
  size_t v87 = v15 * v9;
  float* v88 = &v8[v87];
  __riscv_vse32_v_f32m1(v88, v77, v17);
  size_t v89 = v16 * v9;
  float* v90 = &v8[v89];
  __riscv_vse32_v_f32m1(v90, v78, v17);
  size_t v91 = v17 * v9;
  float* v92 = &v8[v91];
  __riscv_vse32_v_f32m1(v92, v79, v17);
  size_t v93 = v18 * v9;
  float* v94 = &v8[v93];
  __riscv_vse32_v_f32m1(v94, v80, v17);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_6x8_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 6
//   NR         : 8
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_6x8_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v17);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v17);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v17);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v17);
  size_t v30 = v15 * v9;
  float* v31 = &v8[v30];
  vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v17);
  size_t v33 = v16 * v9;
  float* v34 = &v8[v33];
  vfloat32m1_t v35 = __riscv_vle32_v_f32m1(v34, v17);
  size_t v36 = v17 * v9;
  float* v37 = &v8[v36];
  vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v17);
  size_t v39 = v18 * v9;
  float* v40 = &v8[v39];
  vfloat32m1_t v41 = __riscv_vle32_v_f32m1(v40, v17);
  vfloat32m1_t v42;
  v42 = v21;
  vfloat32m1_t v43;
  v43 = v23;
  vfloat32m1_t v44;
  v44 = v26;
  vfloat32m1_t v45;
  v45 = v29;
  vfloat32m1_t v46;
  v46 = v32;
  vfloat32m1_t v47;
  v47 = v35;
  vfloat32m1_t v48;
  v48 = v38;
  vfloat32m1_t v49;
  v49 = v41;
  for (size_t i50 = v11; i50 < v1; i50 += v12) {
    size_t v51 = i50 * v4;
    size_t v52 = 6;
    float* v53 = &v3[v51];
    vfloat32m1_t v54 = __riscv_vle32_v_f32m1(v53, v52);
    size_t v55 = i50 * v6;
    size_t v56 = v55 + v12;
    size_t v57 = v55 + v13;
    size_t v58 = v55 + v14;
    size_t v59 = v55 + v15;
    size_t v60 = v55 + v16;
    size_t v61 = v55 + v17;
    size_t v62 = v55 + v18;
    vfloat32m1_t v63 = v42;
    float v64 = v5[v55];
    vfloat32m1_t v65 = __riscv_vfmacc_vf_f32m1(v63, v64, v54, v52);
    vfloat32m1_t v66 = v43;
    float v67 = v5[v56];
    vfloat32m1_t v68 = __riscv_vfmacc_vf_f32m1(v66, v67, v54, v52);
    vfloat32m1_t v69 = v44;
    float v70 = v5[v57];
    vfloat32m1_t v71 = __riscv_vfmacc_vf_f32m1(v69, v70, v54, v52);
    vfloat32m1_t v72 = v45;
    float v73 = v5[v58];
    vfloat32m1_t v74 = __riscv_vfmacc_vf_f32m1(v72, v73, v54, v52);
    vfloat32m1_t v75 = v46;
    float v76 = v5[v59];
    vfloat32m1_t v77 = __riscv_vfmacc_vf_f32m1(v75, v76, v54, v52);
    vfloat32m1_t v78 = v47;
    float v79 = v5[v60];
    vfloat32m1_t v80 = __riscv_vfmacc_vf_f32m1(v78, v79, v54, v52);
    vfloat32m1_t v81 = v48;
    float v82 = v5[v61];
    vfloat32m1_t v83 = __riscv_vfmacc_vf_f32m1(v81, v82, v54, v52);
    vfloat32m1_t v84 = v49;
    float v85 = v5[v62];
    vfloat32m1_t v86 = __riscv_vfmacc_vf_f32m1(v84, v85, v54, v52);
    v42 = v65;
    v43 = v68;
    v44 = v71;
    v45 = v74;
    v46 = v77;
    v47 = v80;
    v48 = v83;
    v49 = v86;
  }
  vfloat32m1_t v87 = v42;
  vfloat32m1_t v88 = v43;
  vfloat32m1_t v89 = v44;
  vfloat32m1_t v90 = v45;
  vfloat32m1_t v91 = v46;
  vfloat32m1_t v92 = v47;
  vfloat32m1_t v93 = v48;
  vfloat32m1_t v94 = v49;
  float* v95 = &v8[v11];
  __riscv_vse32_v_f32m1(v95, v87, v17);
  float* v96 = &v8[v9];
  __riscv_vse32_v_f32m1(v96, v88, v17);
  size_t v97 = v13 * v9;
  float* v98 = &v8[v97];
  __riscv_vse32_v_f32m1(v98, v89, v17);
  size_t v99 = v14 * v9;
  float* v100 = &v8[v99];
  __riscv_vse32_v_f32m1(v100, v90, v17);
  size_t v101 = v15 * v9;
  float* v102 = &v8[v101];
  __riscv_vse32_v_f32m1(v102, v91, v17);
  size_t v103 = v16 * v9;
  float* v104 = &v8[v103];
  __riscv_vse32_v_f32m1(v104, v92, v17);
  size_t v105 = v17 * v9;
  float* v106 = &v8[v105];
  __riscv_vse32_v_f32m1(v106, v93, v17);
  size_t v107 = v18 * v9;
  float* v108 = &v8[v107];
  __riscv_vse32_v_f32m1(v108, v94, v17);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_7x1_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 7
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_7x1_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v21;
  v21 = v20;
  for (size_t i22 = v11; i22 < v1; i22 += v12) {
    size_t v23 = i22 * v4;
    size_t v24 = 7;
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
  __riscv_vse32_v_f32m1(v32, v31, v18);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_7x1_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 7
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_7x1_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v18);
  vfloat32m1_t v22;
  v22 = v21;
  for (size_t i23 = v11; i23 < v1; i23 += v12) {
    size_t v24 = i23 * v4;
    size_t v25 = 7;
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
  __riscv_vse32_v_f32m1(v33, v32, v18);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_7x2_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 7
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_7x2_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v22;
  v22 = v20;
  vfloat32m1_t v23;
  v23 = v21;
  for (size_t i24 = v11; i24 < v1; i24 += v12) {
    size_t v25 = i24 * v4;
    size_t v26 = 7;
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
  __riscv_vse32_v_f32m1(v39, v37, v18);
  float* v40 = &v8[v9];
  __riscv_vse32_v_f32m1(v40, v38, v18);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_7x2_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 7
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_7x2_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v18);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v18);
  vfloat32m1_t v24;
  v24 = v21;
  vfloat32m1_t v25;
  v25 = v23;
  for (size_t i26 = v11; i26 < v1; i26 += v12) {
    size_t v27 = i26 * v4;
    size_t v28 = 7;
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
  __riscv_vse32_v_f32m1(v41, v39, v18);
  float* v42 = &v8[v9];
  __riscv_vse32_v_f32m1(v42, v40, v18);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_7x3_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 7
//   NR         : 3
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_7x3_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v23;
  v23 = v20;
  vfloat32m1_t v24;
  v24 = v21;
  vfloat32m1_t v25;
  v25 = v22;
  for (size_t i26 = v11; i26 < v1; i26 += v12) {
    size_t v27 = i26 * v4;
    size_t v28 = 7;
    float* v29 = &v3[v27];
    vfloat32m1_t v30 = __riscv_vle32_v_f32m1(v29, v28);
    size_t v31 = i26 * v6;
    size_t v32 = v31 + v12;
    size_t v33 = v31 + v13;
    vfloat32m1_t v34 = v23;
    float v35 = v5[v31];
    vfloat32m1_t v36 = __riscv_vfmacc_vf_f32m1(v34, v35, v30, v28);
    vfloat32m1_t v37 = v24;
    float v38 = v5[v32];
    vfloat32m1_t v39 = __riscv_vfmacc_vf_f32m1(v37, v38, v30, v28);
    vfloat32m1_t v40 = v25;
    float v41 = v5[v33];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v30, v28);
    v23 = v36;
    v24 = v39;
    v25 = v42;
  }
  vfloat32m1_t v43 = v23;
  vfloat32m1_t v44 = v24;
  vfloat32m1_t v45 = v25;
  float* v46 = &v8[v11];
  __riscv_vse32_v_f32m1(v46, v43, v18);
  float* v47 = &v8[v9];
  __riscv_vse32_v_f32m1(v47, v44, v18);
  size_t v48 = v13 * v9;
  float* v49 = &v8[v48];
  __riscv_vse32_v_f32m1(v49, v45, v18);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_7x3_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 7
//   NR         : 3
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_7x3_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v18);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v18);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v18);
  vfloat32m1_t v27;
  v27 = v21;
  vfloat32m1_t v28;
  v28 = v23;
  vfloat32m1_t v29;
  v29 = v26;
  for (size_t i30 = v11; i30 < v1; i30 += v12) {
    size_t v31 = i30 * v4;
    size_t v32 = 7;
    float* v33 = &v3[v31];
    vfloat32m1_t v34 = __riscv_vle32_v_f32m1(v33, v32);
    size_t v35 = i30 * v6;
    size_t v36 = v35 + v12;
    size_t v37 = v35 + v13;
    vfloat32m1_t v38 = v27;
    float v39 = v5[v35];
    vfloat32m1_t v40 = __riscv_vfmacc_vf_f32m1(v38, v39, v34, v32);
    vfloat32m1_t v41 = v28;
    float v42 = v5[v36];
    vfloat32m1_t v43 = __riscv_vfmacc_vf_f32m1(v41, v42, v34, v32);
    vfloat32m1_t v44 = v29;
    float v45 = v5[v37];
    vfloat32m1_t v46 = __riscv_vfmacc_vf_f32m1(v44, v45, v34, v32);
    v27 = v40;
    v28 = v43;
    v29 = v46;
  }
  vfloat32m1_t v47 = v27;
  vfloat32m1_t v48 = v28;
  vfloat32m1_t v49 = v29;
  float* v50 = &v8[v11];
  __riscv_vse32_v_f32m1(v50, v47, v18);
  float* v51 = &v8[v9];
  __riscv_vse32_v_f32m1(v51, v48, v18);
  size_t v52 = v13 * v9;
  float* v53 = &v8[v52];
  __riscv_vse32_v_f32m1(v53, v49, v18);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_7x4_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 7
//   NR         : 4
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_7x4_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v24;
  v24 = v20;
  vfloat32m1_t v25;
  v25 = v21;
  vfloat32m1_t v26;
  v26 = v22;
  vfloat32m1_t v27;
  v27 = v23;
  for (size_t i28 = v11; i28 < v1; i28 += v12) {
    size_t v29 = i28 * v4;
    size_t v30 = 7;
    float* v31 = &v3[v29];
    vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v30);
    size_t v33 = i28 * v6;
    size_t v34 = v33 + v12;
    size_t v35 = v33 + v13;
    size_t v36 = v33 + v14;
    vfloat32m1_t v37 = v24;
    float v38 = v5[v33];
    vfloat32m1_t v39 = __riscv_vfmacc_vf_f32m1(v37, v38, v32, v30);
    vfloat32m1_t v40 = v25;
    float v41 = v5[v34];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v32, v30);
    vfloat32m1_t v43 = v26;
    float v44 = v5[v35];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v32, v30);
    vfloat32m1_t v46 = v27;
    float v47 = v5[v36];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v32, v30);
    v24 = v39;
    v25 = v42;
    v26 = v45;
    v27 = v48;
  }
  vfloat32m1_t v49 = v24;
  vfloat32m1_t v50 = v25;
  vfloat32m1_t v51 = v26;
  vfloat32m1_t v52 = v27;
  float* v53 = &v8[v11];
  __riscv_vse32_v_f32m1(v53, v49, v18);
  float* v54 = &v8[v9];
  __riscv_vse32_v_f32m1(v54, v50, v18);
  size_t v55 = v13 * v9;
  float* v56 = &v8[v55];
  __riscv_vse32_v_f32m1(v56, v51, v18);
  size_t v57 = v14 * v9;
  float* v58 = &v8[v57];
  __riscv_vse32_v_f32m1(v58, v52, v18);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_7x4_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 7
//   NR         : 4
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_7x4_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v18);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v18);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v18);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v18);
  vfloat32m1_t v30;
  v30 = v21;
  vfloat32m1_t v31;
  v31 = v23;
  vfloat32m1_t v32;
  v32 = v26;
  vfloat32m1_t v33;
  v33 = v29;
  for (size_t i34 = v11; i34 < v1; i34 += v12) {
    size_t v35 = i34 * v4;
    size_t v36 = 7;
    float* v37 = &v3[v35];
    vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v36);
    size_t v39 = i34 * v6;
    size_t v40 = v39 + v12;
    size_t v41 = v39 + v13;
    size_t v42 = v39 + v14;
    vfloat32m1_t v43 = v30;
    float v44 = v5[v39];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v38, v36);
    vfloat32m1_t v46 = v31;
    float v47 = v5[v40];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v38, v36);
    vfloat32m1_t v49 = v32;
    float v50 = v5[v41];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v38, v36);
    vfloat32m1_t v52 = v33;
    float v53 = v5[v42];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v38, v36);
    v30 = v45;
    v31 = v48;
    v32 = v51;
    v33 = v54;
  }
  vfloat32m1_t v55 = v30;
  vfloat32m1_t v56 = v31;
  vfloat32m1_t v57 = v32;
  vfloat32m1_t v58 = v33;
  float* v59 = &v8[v11];
  __riscv_vse32_v_f32m1(v59, v55, v18);
  float* v60 = &v8[v9];
  __riscv_vse32_v_f32m1(v60, v56, v18);
  size_t v61 = v13 * v9;
  float* v62 = &v8[v61];
  __riscv_vse32_v_f32m1(v62, v57, v18);
  size_t v63 = v14 * v9;
  float* v64 = &v8[v63];
  __riscv_vse32_v_f32m1(v64, v58, v18);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_7x5_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 7
//   NR         : 5
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_7x5_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v24 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v25;
  v25 = v20;
  vfloat32m1_t v26;
  v26 = v21;
  vfloat32m1_t v27;
  v27 = v22;
  vfloat32m1_t v28;
  v28 = v23;
  vfloat32m1_t v29;
  v29 = v24;
  for (size_t i30 = v11; i30 < v1; i30 += v12) {
    size_t v31 = i30 * v4;
    size_t v32 = 7;
    float* v33 = &v3[v31];
    vfloat32m1_t v34 = __riscv_vle32_v_f32m1(v33, v32);
    size_t v35 = i30 * v6;
    size_t v36 = v35 + v12;
    size_t v37 = v35 + v13;
    size_t v38 = v35 + v14;
    size_t v39 = v35 + v15;
    vfloat32m1_t v40 = v25;
    float v41 = v5[v35];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v34, v32);
    vfloat32m1_t v43 = v26;
    float v44 = v5[v36];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v34, v32);
    vfloat32m1_t v46 = v27;
    float v47 = v5[v37];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v34, v32);
    vfloat32m1_t v49 = v28;
    float v50 = v5[v38];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v34, v32);
    vfloat32m1_t v52 = v29;
    float v53 = v5[v39];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v34, v32);
    v25 = v42;
    v26 = v45;
    v27 = v48;
    v28 = v51;
    v29 = v54;
  }
  vfloat32m1_t v55 = v25;
  vfloat32m1_t v56 = v26;
  vfloat32m1_t v57 = v27;
  vfloat32m1_t v58 = v28;
  vfloat32m1_t v59 = v29;
  float* v60 = &v8[v11];
  __riscv_vse32_v_f32m1(v60, v55, v18);
  float* v61 = &v8[v9];
  __riscv_vse32_v_f32m1(v61, v56, v18);
  size_t v62 = v13 * v9;
  float* v63 = &v8[v62];
  __riscv_vse32_v_f32m1(v63, v57, v18);
  size_t v64 = v14 * v9;
  float* v65 = &v8[v64];
  __riscv_vse32_v_f32m1(v65, v58, v18);
  size_t v66 = v15 * v9;
  float* v67 = &v8[v66];
  __riscv_vse32_v_f32m1(v67, v59, v18);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_7x5_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 7
//   NR         : 5
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_7x5_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v18);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v18);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v18);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v18);
  size_t v30 = v15 * v9;
  float* v31 = &v8[v30];
  vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v18);
  vfloat32m1_t v33;
  v33 = v21;
  vfloat32m1_t v34;
  v34 = v23;
  vfloat32m1_t v35;
  v35 = v26;
  vfloat32m1_t v36;
  v36 = v29;
  vfloat32m1_t v37;
  v37 = v32;
  for (size_t i38 = v11; i38 < v1; i38 += v12) {
    size_t v39 = i38 * v4;
    size_t v40 = 7;
    float* v41 = &v3[v39];
    vfloat32m1_t v42 = __riscv_vle32_v_f32m1(v41, v40);
    size_t v43 = i38 * v6;
    size_t v44 = v43 + v12;
    size_t v45 = v43 + v13;
    size_t v46 = v43 + v14;
    size_t v47 = v43 + v15;
    vfloat32m1_t v48 = v33;
    float v49 = v5[v43];
    vfloat32m1_t v50 = __riscv_vfmacc_vf_f32m1(v48, v49, v42, v40);
    vfloat32m1_t v51 = v34;
    float v52 = v5[v44];
    vfloat32m1_t v53 = __riscv_vfmacc_vf_f32m1(v51, v52, v42, v40);
    vfloat32m1_t v54 = v35;
    float v55 = v5[v45];
    vfloat32m1_t v56 = __riscv_vfmacc_vf_f32m1(v54, v55, v42, v40);
    vfloat32m1_t v57 = v36;
    float v58 = v5[v46];
    vfloat32m1_t v59 = __riscv_vfmacc_vf_f32m1(v57, v58, v42, v40);
    vfloat32m1_t v60 = v37;
    float v61 = v5[v47];
    vfloat32m1_t v62 = __riscv_vfmacc_vf_f32m1(v60, v61, v42, v40);
    v33 = v50;
    v34 = v53;
    v35 = v56;
    v36 = v59;
    v37 = v62;
  }
  vfloat32m1_t v63 = v33;
  vfloat32m1_t v64 = v34;
  vfloat32m1_t v65 = v35;
  vfloat32m1_t v66 = v36;
  vfloat32m1_t v67 = v37;
  float* v68 = &v8[v11];
  __riscv_vse32_v_f32m1(v68, v63, v18);
  float* v69 = &v8[v9];
  __riscv_vse32_v_f32m1(v69, v64, v18);
  size_t v70 = v13 * v9;
  float* v71 = &v8[v70];
  __riscv_vse32_v_f32m1(v71, v65, v18);
  size_t v72 = v14 * v9;
  float* v73 = &v8[v72];
  __riscv_vse32_v_f32m1(v73, v66, v18);
  size_t v74 = v15 * v9;
  float* v75 = &v8[v74];
  __riscv_vse32_v_f32m1(v75, v67, v18);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_7x6_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 7
//   NR         : 6
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_7x6_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v24 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v25 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v26;
  v26 = v20;
  vfloat32m1_t v27;
  v27 = v21;
  vfloat32m1_t v28;
  v28 = v22;
  vfloat32m1_t v29;
  v29 = v23;
  vfloat32m1_t v30;
  v30 = v24;
  vfloat32m1_t v31;
  v31 = v25;
  for (size_t i32 = v11; i32 < v1; i32 += v12) {
    size_t v33 = i32 * v4;
    size_t v34 = 7;
    float* v35 = &v3[v33];
    vfloat32m1_t v36 = __riscv_vle32_v_f32m1(v35, v34);
    size_t v37 = i32 * v6;
    size_t v38 = v37 + v12;
    size_t v39 = v37 + v13;
    size_t v40 = v37 + v14;
    size_t v41 = v37 + v15;
    size_t v42 = v37 + v16;
    vfloat32m1_t v43 = v26;
    float v44 = v5[v37];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v36, v34);
    vfloat32m1_t v46 = v27;
    float v47 = v5[v38];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v36, v34);
    vfloat32m1_t v49 = v28;
    float v50 = v5[v39];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v36, v34);
    vfloat32m1_t v52 = v29;
    float v53 = v5[v40];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v36, v34);
    vfloat32m1_t v55 = v30;
    float v56 = v5[v41];
    vfloat32m1_t v57 = __riscv_vfmacc_vf_f32m1(v55, v56, v36, v34);
    vfloat32m1_t v58 = v31;
    float v59 = v5[v42];
    vfloat32m1_t v60 = __riscv_vfmacc_vf_f32m1(v58, v59, v36, v34);
    v26 = v45;
    v27 = v48;
    v28 = v51;
    v29 = v54;
    v30 = v57;
    v31 = v60;
  }
  vfloat32m1_t v61 = v26;
  vfloat32m1_t v62 = v27;
  vfloat32m1_t v63 = v28;
  vfloat32m1_t v64 = v29;
  vfloat32m1_t v65 = v30;
  vfloat32m1_t v66 = v31;
  float* v67 = &v8[v11];
  __riscv_vse32_v_f32m1(v67, v61, v18);
  float* v68 = &v8[v9];
  __riscv_vse32_v_f32m1(v68, v62, v18);
  size_t v69 = v13 * v9;
  float* v70 = &v8[v69];
  __riscv_vse32_v_f32m1(v70, v63, v18);
  size_t v71 = v14 * v9;
  float* v72 = &v8[v71];
  __riscv_vse32_v_f32m1(v72, v64, v18);
  size_t v73 = v15 * v9;
  float* v74 = &v8[v73];
  __riscv_vse32_v_f32m1(v74, v65, v18);
  size_t v75 = v16 * v9;
  float* v76 = &v8[v75];
  __riscv_vse32_v_f32m1(v76, v66, v18);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_7x6_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 7
//   NR         : 6
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_7x6_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v18);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v18);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v18);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v18);
  size_t v30 = v15 * v9;
  float* v31 = &v8[v30];
  vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v18);
  size_t v33 = v16 * v9;
  float* v34 = &v8[v33];
  vfloat32m1_t v35 = __riscv_vle32_v_f32m1(v34, v18);
  vfloat32m1_t v36;
  v36 = v21;
  vfloat32m1_t v37;
  v37 = v23;
  vfloat32m1_t v38;
  v38 = v26;
  vfloat32m1_t v39;
  v39 = v29;
  vfloat32m1_t v40;
  v40 = v32;
  vfloat32m1_t v41;
  v41 = v35;
  for (size_t i42 = v11; i42 < v1; i42 += v12) {
    size_t v43 = i42 * v4;
    size_t v44 = 7;
    float* v45 = &v3[v43];
    vfloat32m1_t v46 = __riscv_vle32_v_f32m1(v45, v44);
    size_t v47 = i42 * v6;
    size_t v48 = v47 + v12;
    size_t v49 = v47 + v13;
    size_t v50 = v47 + v14;
    size_t v51 = v47 + v15;
    size_t v52 = v47 + v16;
    vfloat32m1_t v53 = v36;
    float v54 = v5[v47];
    vfloat32m1_t v55 = __riscv_vfmacc_vf_f32m1(v53, v54, v46, v44);
    vfloat32m1_t v56 = v37;
    float v57 = v5[v48];
    vfloat32m1_t v58 = __riscv_vfmacc_vf_f32m1(v56, v57, v46, v44);
    vfloat32m1_t v59 = v38;
    float v60 = v5[v49];
    vfloat32m1_t v61 = __riscv_vfmacc_vf_f32m1(v59, v60, v46, v44);
    vfloat32m1_t v62 = v39;
    float v63 = v5[v50];
    vfloat32m1_t v64 = __riscv_vfmacc_vf_f32m1(v62, v63, v46, v44);
    vfloat32m1_t v65 = v40;
    float v66 = v5[v51];
    vfloat32m1_t v67 = __riscv_vfmacc_vf_f32m1(v65, v66, v46, v44);
    vfloat32m1_t v68 = v41;
    float v69 = v5[v52];
    vfloat32m1_t v70 = __riscv_vfmacc_vf_f32m1(v68, v69, v46, v44);
    v36 = v55;
    v37 = v58;
    v38 = v61;
    v39 = v64;
    v40 = v67;
    v41 = v70;
  }
  vfloat32m1_t v71 = v36;
  vfloat32m1_t v72 = v37;
  vfloat32m1_t v73 = v38;
  vfloat32m1_t v74 = v39;
  vfloat32m1_t v75 = v40;
  vfloat32m1_t v76 = v41;
  float* v77 = &v8[v11];
  __riscv_vse32_v_f32m1(v77, v71, v18);
  float* v78 = &v8[v9];
  __riscv_vse32_v_f32m1(v78, v72, v18);
  size_t v79 = v13 * v9;
  float* v80 = &v8[v79];
  __riscv_vse32_v_f32m1(v80, v73, v18);
  size_t v81 = v14 * v9;
  float* v82 = &v8[v81];
  __riscv_vse32_v_f32m1(v82, v74, v18);
  size_t v83 = v15 * v9;
  float* v84 = &v8[v83];
  __riscv_vse32_v_f32m1(v84, v75, v18);
  size_t v85 = v16 * v9;
  float* v86 = &v8[v85];
  __riscv_vse32_v_f32m1(v86, v76, v18);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_7x7_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 7
//   NR         : 7
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_7x7_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v24 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v25 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v26 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v27;
  v27 = v20;
  vfloat32m1_t v28;
  v28 = v21;
  vfloat32m1_t v29;
  v29 = v22;
  vfloat32m1_t v30;
  v30 = v23;
  vfloat32m1_t v31;
  v31 = v24;
  vfloat32m1_t v32;
  v32 = v25;
  vfloat32m1_t v33;
  v33 = v26;
  for (size_t i34 = v11; i34 < v1; i34 += v12) {
    size_t v35 = i34 * v4;
    size_t v36 = 7;
    float* v37 = &v3[v35];
    vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v36);
    size_t v39 = i34 * v6;
    size_t v40 = v39 + v12;
    size_t v41 = v39 + v13;
    size_t v42 = v39 + v14;
    size_t v43 = v39 + v15;
    size_t v44 = v39 + v16;
    size_t v45 = v39 + v17;
    vfloat32m1_t v46 = v27;
    float v47 = v5[v39];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v38, v36);
    vfloat32m1_t v49 = v28;
    float v50 = v5[v40];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v38, v36);
    vfloat32m1_t v52 = v29;
    float v53 = v5[v41];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v38, v36);
    vfloat32m1_t v55 = v30;
    float v56 = v5[v42];
    vfloat32m1_t v57 = __riscv_vfmacc_vf_f32m1(v55, v56, v38, v36);
    vfloat32m1_t v58 = v31;
    float v59 = v5[v43];
    vfloat32m1_t v60 = __riscv_vfmacc_vf_f32m1(v58, v59, v38, v36);
    vfloat32m1_t v61 = v32;
    float v62 = v5[v44];
    vfloat32m1_t v63 = __riscv_vfmacc_vf_f32m1(v61, v62, v38, v36);
    vfloat32m1_t v64 = v33;
    float v65 = v5[v45];
    vfloat32m1_t v66 = __riscv_vfmacc_vf_f32m1(v64, v65, v38, v36);
    v27 = v48;
    v28 = v51;
    v29 = v54;
    v30 = v57;
    v31 = v60;
    v32 = v63;
    v33 = v66;
  }
  vfloat32m1_t v67 = v27;
  vfloat32m1_t v68 = v28;
  vfloat32m1_t v69 = v29;
  vfloat32m1_t v70 = v30;
  vfloat32m1_t v71 = v31;
  vfloat32m1_t v72 = v32;
  vfloat32m1_t v73 = v33;
  float* v74 = &v8[v11];
  __riscv_vse32_v_f32m1(v74, v67, v18);
  float* v75 = &v8[v9];
  __riscv_vse32_v_f32m1(v75, v68, v18);
  size_t v76 = v13 * v9;
  float* v77 = &v8[v76];
  __riscv_vse32_v_f32m1(v77, v69, v18);
  size_t v78 = v14 * v9;
  float* v79 = &v8[v78];
  __riscv_vse32_v_f32m1(v79, v70, v18);
  size_t v80 = v15 * v9;
  float* v81 = &v8[v80];
  __riscv_vse32_v_f32m1(v81, v71, v18);
  size_t v82 = v16 * v9;
  float* v83 = &v8[v82];
  __riscv_vse32_v_f32m1(v83, v72, v18);
  size_t v84 = v17 * v9;
  float* v85 = &v8[v84];
  __riscv_vse32_v_f32m1(v85, v73, v18);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_7x7_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 7
//   NR         : 7
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_7x7_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v18);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v18);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v18);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v18);
  size_t v30 = v15 * v9;
  float* v31 = &v8[v30];
  vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v18);
  size_t v33 = v16 * v9;
  float* v34 = &v8[v33];
  vfloat32m1_t v35 = __riscv_vle32_v_f32m1(v34, v18);
  size_t v36 = v17 * v9;
  float* v37 = &v8[v36];
  vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v18);
  vfloat32m1_t v39;
  v39 = v21;
  vfloat32m1_t v40;
  v40 = v23;
  vfloat32m1_t v41;
  v41 = v26;
  vfloat32m1_t v42;
  v42 = v29;
  vfloat32m1_t v43;
  v43 = v32;
  vfloat32m1_t v44;
  v44 = v35;
  vfloat32m1_t v45;
  v45 = v38;
  for (size_t i46 = v11; i46 < v1; i46 += v12) {
    size_t v47 = i46 * v4;
    size_t v48 = 7;
    float* v49 = &v3[v47];
    vfloat32m1_t v50 = __riscv_vle32_v_f32m1(v49, v48);
    size_t v51 = i46 * v6;
    size_t v52 = v51 + v12;
    size_t v53 = v51 + v13;
    size_t v54 = v51 + v14;
    size_t v55 = v51 + v15;
    size_t v56 = v51 + v16;
    size_t v57 = v51 + v17;
    vfloat32m1_t v58 = v39;
    float v59 = v5[v51];
    vfloat32m1_t v60 = __riscv_vfmacc_vf_f32m1(v58, v59, v50, v48);
    vfloat32m1_t v61 = v40;
    float v62 = v5[v52];
    vfloat32m1_t v63 = __riscv_vfmacc_vf_f32m1(v61, v62, v50, v48);
    vfloat32m1_t v64 = v41;
    float v65 = v5[v53];
    vfloat32m1_t v66 = __riscv_vfmacc_vf_f32m1(v64, v65, v50, v48);
    vfloat32m1_t v67 = v42;
    float v68 = v5[v54];
    vfloat32m1_t v69 = __riscv_vfmacc_vf_f32m1(v67, v68, v50, v48);
    vfloat32m1_t v70 = v43;
    float v71 = v5[v55];
    vfloat32m1_t v72 = __riscv_vfmacc_vf_f32m1(v70, v71, v50, v48);
    vfloat32m1_t v73 = v44;
    float v74 = v5[v56];
    vfloat32m1_t v75 = __riscv_vfmacc_vf_f32m1(v73, v74, v50, v48);
    vfloat32m1_t v76 = v45;
    float v77 = v5[v57];
    vfloat32m1_t v78 = __riscv_vfmacc_vf_f32m1(v76, v77, v50, v48);
    v39 = v60;
    v40 = v63;
    v41 = v66;
    v42 = v69;
    v43 = v72;
    v44 = v75;
    v45 = v78;
  }
  vfloat32m1_t v79 = v39;
  vfloat32m1_t v80 = v40;
  vfloat32m1_t v81 = v41;
  vfloat32m1_t v82 = v42;
  vfloat32m1_t v83 = v43;
  vfloat32m1_t v84 = v44;
  vfloat32m1_t v85 = v45;
  float* v86 = &v8[v11];
  __riscv_vse32_v_f32m1(v86, v79, v18);
  float* v87 = &v8[v9];
  __riscv_vse32_v_f32m1(v87, v80, v18);
  size_t v88 = v13 * v9;
  float* v89 = &v8[v88];
  __riscv_vse32_v_f32m1(v89, v81, v18);
  size_t v90 = v14 * v9;
  float* v91 = &v8[v90];
  __riscv_vse32_v_f32m1(v91, v82, v18);
  size_t v92 = v15 * v9;
  float* v93 = &v8[v92];
  __riscv_vse32_v_f32m1(v93, v83, v18);
  size_t v94 = v16 * v9;
  float* v95 = &v8[v94];
  __riscv_vse32_v_f32m1(v95, v84, v18);
  size_t v96 = v17 * v9;
  float* v97 = &v8[v96];
  __riscv_vse32_v_f32m1(v97, v85, v18);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_7x8_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 7
//   NR         : 8
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_7x8_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v24 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v25 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v26 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v27 = __riscv_vfmv_v_f_f32m1(v10, v18);
  vfloat32m1_t v28;
  v28 = v20;
  vfloat32m1_t v29;
  v29 = v21;
  vfloat32m1_t v30;
  v30 = v22;
  vfloat32m1_t v31;
  v31 = v23;
  vfloat32m1_t v32;
  v32 = v24;
  vfloat32m1_t v33;
  v33 = v25;
  vfloat32m1_t v34;
  v34 = v26;
  vfloat32m1_t v35;
  v35 = v27;
  for (size_t i36 = v11; i36 < v1; i36 += v12) {
    size_t v37 = i36 * v4;
    size_t v38 = 7;
    float* v39 = &v3[v37];
    vfloat32m1_t v40 = __riscv_vle32_v_f32m1(v39, v38);
    size_t v41 = i36 * v6;
    size_t v42 = v41 + v12;
    size_t v43 = v41 + v13;
    size_t v44 = v41 + v14;
    size_t v45 = v41 + v15;
    size_t v46 = v41 + v16;
    size_t v47 = v41 + v17;
    size_t v48 = v41 + v18;
    vfloat32m1_t v49 = v28;
    float v50 = v5[v41];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v40, v38);
    vfloat32m1_t v52 = v29;
    float v53 = v5[v42];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v40, v38);
    vfloat32m1_t v55 = v30;
    float v56 = v5[v43];
    vfloat32m1_t v57 = __riscv_vfmacc_vf_f32m1(v55, v56, v40, v38);
    vfloat32m1_t v58 = v31;
    float v59 = v5[v44];
    vfloat32m1_t v60 = __riscv_vfmacc_vf_f32m1(v58, v59, v40, v38);
    vfloat32m1_t v61 = v32;
    float v62 = v5[v45];
    vfloat32m1_t v63 = __riscv_vfmacc_vf_f32m1(v61, v62, v40, v38);
    vfloat32m1_t v64 = v33;
    float v65 = v5[v46];
    vfloat32m1_t v66 = __riscv_vfmacc_vf_f32m1(v64, v65, v40, v38);
    vfloat32m1_t v67 = v34;
    float v68 = v5[v47];
    vfloat32m1_t v69 = __riscv_vfmacc_vf_f32m1(v67, v68, v40, v38);
    vfloat32m1_t v70 = v35;
    float v71 = v5[v48];
    vfloat32m1_t v72 = __riscv_vfmacc_vf_f32m1(v70, v71, v40, v38);
    v28 = v51;
    v29 = v54;
    v30 = v57;
    v31 = v60;
    v32 = v63;
    v33 = v66;
    v34 = v69;
    v35 = v72;
  }
  vfloat32m1_t v73 = v28;
  vfloat32m1_t v74 = v29;
  vfloat32m1_t v75 = v30;
  vfloat32m1_t v76 = v31;
  vfloat32m1_t v77 = v32;
  vfloat32m1_t v78 = v33;
  vfloat32m1_t v79 = v34;
  vfloat32m1_t v80 = v35;
  float* v81 = &v8[v11];
  __riscv_vse32_v_f32m1(v81, v73, v18);
  float* v82 = &v8[v9];
  __riscv_vse32_v_f32m1(v82, v74, v18);
  size_t v83 = v13 * v9;
  float* v84 = &v8[v83];
  __riscv_vse32_v_f32m1(v84, v75, v18);
  size_t v85 = v14 * v9;
  float* v86 = &v8[v85];
  __riscv_vse32_v_f32m1(v86, v76, v18);
  size_t v87 = v15 * v9;
  float* v88 = &v8[v87];
  __riscv_vse32_v_f32m1(v88, v77, v18);
  size_t v89 = v16 * v9;
  float* v90 = &v8[v89];
  __riscv_vse32_v_f32m1(v90, v78, v18);
  size_t v91 = v17 * v9;
  float* v92 = &v8[v91];
  __riscv_vse32_v_f32m1(v92, v79, v18);
  size_t v93 = v18 * v9;
  float* v94 = &v8[v93];
  __riscv_vse32_v_f32m1(v94, v80, v18);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_7x8_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 7
//   NR         : 8
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_7x8_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v18);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v18);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v18);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v18);
  size_t v30 = v15 * v9;
  float* v31 = &v8[v30];
  vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v18);
  size_t v33 = v16 * v9;
  float* v34 = &v8[v33];
  vfloat32m1_t v35 = __riscv_vle32_v_f32m1(v34, v18);
  size_t v36 = v17 * v9;
  float* v37 = &v8[v36];
  vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v18);
  size_t v39 = v18 * v9;
  float* v40 = &v8[v39];
  vfloat32m1_t v41 = __riscv_vle32_v_f32m1(v40, v18);
  vfloat32m1_t v42;
  v42 = v21;
  vfloat32m1_t v43;
  v43 = v23;
  vfloat32m1_t v44;
  v44 = v26;
  vfloat32m1_t v45;
  v45 = v29;
  vfloat32m1_t v46;
  v46 = v32;
  vfloat32m1_t v47;
  v47 = v35;
  vfloat32m1_t v48;
  v48 = v38;
  vfloat32m1_t v49;
  v49 = v41;
  for (size_t i50 = v11; i50 < v1; i50 += v12) {
    size_t v51 = i50 * v4;
    size_t v52 = 7;
    float* v53 = &v3[v51];
    vfloat32m1_t v54 = __riscv_vle32_v_f32m1(v53, v52);
    size_t v55 = i50 * v6;
    size_t v56 = v55 + v12;
    size_t v57 = v55 + v13;
    size_t v58 = v55 + v14;
    size_t v59 = v55 + v15;
    size_t v60 = v55 + v16;
    size_t v61 = v55 + v17;
    size_t v62 = v55 + v18;
    vfloat32m1_t v63 = v42;
    float v64 = v5[v55];
    vfloat32m1_t v65 = __riscv_vfmacc_vf_f32m1(v63, v64, v54, v52);
    vfloat32m1_t v66 = v43;
    float v67 = v5[v56];
    vfloat32m1_t v68 = __riscv_vfmacc_vf_f32m1(v66, v67, v54, v52);
    vfloat32m1_t v69 = v44;
    float v70 = v5[v57];
    vfloat32m1_t v71 = __riscv_vfmacc_vf_f32m1(v69, v70, v54, v52);
    vfloat32m1_t v72 = v45;
    float v73 = v5[v58];
    vfloat32m1_t v74 = __riscv_vfmacc_vf_f32m1(v72, v73, v54, v52);
    vfloat32m1_t v75 = v46;
    float v76 = v5[v59];
    vfloat32m1_t v77 = __riscv_vfmacc_vf_f32m1(v75, v76, v54, v52);
    vfloat32m1_t v78 = v47;
    float v79 = v5[v60];
    vfloat32m1_t v80 = __riscv_vfmacc_vf_f32m1(v78, v79, v54, v52);
    vfloat32m1_t v81 = v48;
    float v82 = v5[v61];
    vfloat32m1_t v83 = __riscv_vfmacc_vf_f32m1(v81, v82, v54, v52);
    vfloat32m1_t v84 = v49;
    float v85 = v5[v62];
    vfloat32m1_t v86 = __riscv_vfmacc_vf_f32m1(v84, v85, v54, v52);
    v42 = v65;
    v43 = v68;
    v44 = v71;
    v45 = v74;
    v46 = v77;
    v47 = v80;
    v48 = v83;
    v49 = v86;
  }
  vfloat32m1_t v87 = v42;
  vfloat32m1_t v88 = v43;
  vfloat32m1_t v89 = v44;
  vfloat32m1_t v90 = v45;
  vfloat32m1_t v91 = v46;
  vfloat32m1_t v92 = v47;
  vfloat32m1_t v93 = v48;
  vfloat32m1_t v94 = v49;
  float* v95 = &v8[v11];
  __riscv_vse32_v_f32m1(v95, v87, v18);
  float* v96 = &v8[v9];
  __riscv_vse32_v_f32m1(v96, v88, v18);
  size_t v97 = v13 * v9;
  float* v98 = &v8[v97];
  __riscv_vse32_v_f32m1(v98, v89, v18);
  size_t v99 = v14 * v9;
  float* v100 = &v8[v99];
  __riscv_vse32_v_f32m1(v100, v90, v18);
  size_t v101 = v15 * v9;
  float* v102 = &v8[v101];
  __riscv_vse32_v_f32m1(v102, v91, v18);
  size_t v103 = v16 * v9;
  float* v104 = &v8[v103];
  __riscv_vse32_v_f32m1(v104, v92, v18);
  size_t v105 = v17 * v9;
  float* v106 = &v8[v105];
  __riscv_vse32_v_f32m1(v106, v93, v18);
  size_t v107 = v18 * v9;
  float* v108 = &v8[v107];
  __riscv_vse32_v_f32m1(v108, v94, v18);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_8x1_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 8
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_8x1_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v21;
  v21 = v20;
  for (size_t i22 = v11; i22 < v1; i22 += v12) {
    size_t v23 = i22 * v4;
    size_t v24 = 8;
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
  __riscv_vse32_v_f32m1(v32, v31, v19);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_8x1_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 8
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_8x1_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v19);
  vfloat32m1_t v22;
  v22 = v21;
  for (size_t i23 = v11; i23 < v1; i23 += v12) {
    size_t v24 = i23 * v4;
    size_t v25 = 8;
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
  __riscv_vse32_v_f32m1(v33, v32, v19);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_8x2_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 8
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_8x2_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v22;
  v22 = v20;
  vfloat32m1_t v23;
  v23 = v21;
  for (size_t i24 = v11; i24 < v1; i24 += v12) {
    size_t v25 = i24 * v4;
    size_t v26 = 8;
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
  __riscv_vse32_v_f32m1(v39, v37, v19);
  float* v40 = &v8[v9];
  __riscv_vse32_v_f32m1(v40, v38, v19);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_8x2_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 8
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_8x2_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v19);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v19);
  vfloat32m1_t v24;
  v24 = v21;
  vfloat32m1_t v25;
  v25 = v23;
  for (size_t i26 = v11; i26 < v1; i26 += v12) {
    size_t v27 = i26 * v4;
    size_t v28 = 8;
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
  __riscv_vse32_v_f32m1(v41, v39, v19);
  float* v42 = &v8[v9];
  __riscv_vse32_v_f32m1(v42, v40, v19);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_8x3_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 8
//   NR         : 3
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_8x3_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v23;
  v23 = v20;
  vfloat32m1_t v24;
  v24 = v21;
  vfloat32m1_t v25;
  v25 = v22;
  for (size_t i26 = v11; i26 < v1; i26 += v12) {
    size_t v27 = i26 * v4;
    size_t v28 = 8;
    float* v29 = &v3[v27];
    vfloat32m1_t v30 = __riscv_vle32_v_f32m1(v29, v28);
    size_t v31 = i26 * v6;
    size_t v32 = v31 + v12;
    size_t v33 = v31 + v13;
    vfloat32m1_t v34 = v23;
    float v35 = v5[v31];
    vfloat32m1_t v36 = __riscv_vfmacc_vf_f32m1(v34, v35, v30, v28);
    vfloat32m1_t v37 = v24;
    float v38 = v5[v32];
    vfloat32m1_t v39 = __riscv_vfmacc_vf_f32m1(v37, v38, v30, v28);
    vfloat32m1_t v40 = v25;
    float v41 = v5[v33];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v30, v28);
    v23 = v36;
    v24 = v39;
    v25 = v42;
  }
  vfloat32m1_t v43 = v23;
  vfloat32m1_t v44 = v24;
  vfloat32m1_t v45 = v25;
  float* v46 = &v8[v11];
  __riscv_vse32_v_f32m1(v46, v43, v19);
  float* v47 = &v8[v9];
  __riscv_vse32_v_f32m1(v47, v44, v19);
  size_t v48 = v13 * v9;
  float* v49 = &v8[v48];
  __riscv_vse32_v_f32m1(v49, v45, v19);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_8x3_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 8
//   NR         : 3
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_8x3_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v19);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v19);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v19);
  vfloat32m1_t v27;
  v27 = v21;
  vfloat32m1_t v28;
  v28 = v23;
  vfloat32m1_t v29;
  v29 = v26;
  for (size_t i30 = v11; i30 < v1; i30 += v12) {
    size_t v31 = i30 * v4;
    size_t v32 = 8;
    float* v33 = &v3[v31];
    vfloat32m1_t v34 = __riscv_vle32_v_f32m1(v33, v32);
    size_t v35 = i30 * v6;
    size_t v36 = v35 + v12;
    size_t v37 = v35 + v13;
    vfloat32m1_t v38 = v27;
    float v39 = v5[v35];
    vfloat32m1_t v40 = __riscv_vfmacc_vf_f32m1(v38, v39, v34, v32);
    vfloat32m1_t v41 = v28;
    float v42 = v5[v36];
    vfloat32m1_t v43 = __riscv_vfmacc_vf_f32m1(v41, v42, v34, v32);
    vfloat32m1_t v44 = v29;
    float v45 = v5[v37];
    vfloat32m1_t v46 = __riscv_vfmacc_vf_f32m1(v44, v45, v34, v32);
    v27 = v40;
    v28 = v43;
    v29 = v46;
  }
  vfloat32m1_t v47 = v27;
  vfloat32m1_t v48 = v28;
  vfloat32m1_t v49 = v29;
  float* v50 = &v8[v11];
  __riscv_vse32_v_f32m1(v50, v47, v19);
  float* v51 = &v8[v9];
  __riscv_vse32_v_f32m1(v51, v48, v19);
  size_t v52 = v13 * v9;
  float* v53 = &v8[v52];
  __riscv_vse32_v_f32m1(v53, v49, v19);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_8x4_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 8
//   NR         : 4
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_8x4_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v24;
  v24 = v20;
  vfloat32m1_t v25;
  v25 = v21;
  vfloat32m1_t v26;
  v26 = v22;
  vfloat32m1_t v27;
  v27 = v23;
  for (size_t i28 = v11; i28 < v1; i28 += v12) {
    size_t v29 = i28 * v4;
    size_t v30 = 8;
    float* v31 = &v3[v29];
    vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v30);
    size_t v33 = i28 * v6;
    size_t v34 = v33 + v12;
    size_t v35 = v33 + v13;
    size_t v36 = v33 + v14;
    vfloat32m1_t v37 = v24;
    float v38 = v5[v33];
    vfloat32m1_t v39 = __riscv_vfmacc_vf_f32m1(v37, v38, v32, v30);
    vfloat32m1_t v40 = v25;
    float v41 = v5[v34];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v32, v30);
    vfloat32m1_t v43 = v26;
    float v44 = v5[v35];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v32, v30);
    vfloat32m1_t v46 = v27;
    float v47 = v5[v36];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v32, v30);
    v24 = v39;
    v25 = v42;
    v26 = v45;
    v27 = v48;
  }
  vfloat32m1_t v49 = v24;
  vfloat32m1_t v50 = v25;
  vfloat32m1_t v51 = v26;
  vfloat32m1_t v52 = v27;
  float* v53 = &v8[v11];
  __riscv_vse32_v_f32m1(v53, v49, v19);
  float* v54 = &v8[v9];
  __riscv_vse32_v_f32m1(v54, v50, v19);
  size_t v55 = v13 * v9;
  float* v56 = &v8[v55];
  __riscv_vse32_v_f32m1(v56, v51, v19);
  size_t v57 = v14 * v9;
  float* v58 = &v8[v57];
  __riscv_vse32_v_f32m1(v58, v52, v19);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_8x4_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 8
//   NR         : 4
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_8x4_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v19);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v19);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v19);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v19);
  vfloat32m1_t v30;
  v30 = v21;
  vfloat32m1_t v31;
  v31 = v23;
  vfloat32m1_t v32;
  v32 = v26;
  vfloat32m1_t v33;
  v33 = v29;
  for (size_t i34 = v11; i34 < v1; i34 += v12) {
    size_t v35 = i34 * v4;
    size_t v36 = 8;
    float* v37 = &v3[v35];
    vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v36);
    size_t v39 = i34 * v6;
    size_t v40 = v39 + v12;
    size_t v41 = v39 + v13;
    size_t v42 = v39 + v14;
    vfloat32m1_t v43 = v30;
    float v44 = v5[v39];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v38, v36);
    vfloat32m1_t v46 = v31;
    float v47 = v5[v40];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v38, v36);
    vfloat32m1_t v49 = v32;
    float v50 = v5[v41];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v38, v36);
    vfloat32m1_t v52 = v33;
    float v53 = v5[v42];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v38, v36);
    v30 = v45;
    v31 = v48;
    v32 = v51;
    v33 = v54;
  }
  vfloat32m1_t v55 = v30;
  vfloat32m1_t v56 = v31;
  vfloat32m1_t v57 = v32;
  vfloat32m1_t v58 = v33;
  float* v59 = &v8[v11];
  __riscv_vse32_v_f32m1(v59, v55, v19);
  float* v60 = &v8[v9];
  __riscv_vse32_v_f32m1(v60, v56, v19);
  size_t v61 = v13 * v9;
  float* v62 = &v8[v61];
  __riscv_vse32_v_f32m1(v62, v57, v19);
  size_t v63 = v14 * v9;
  float* v64 = &v8[v63];
  __riscv_vse32_v_f32m1(v64, v58, v19);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_8x5_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 8
//   NR         : 5
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_8x5_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v24 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v25;
  v25 = v20;
  vfloat32m1_t v26;
  v26 = v21;
  vfloat32m1_t v27;
  v27 = v22;
  vfloat32m1_t v28;
  v28 = v23;
  vfloat32m1_t v29;
  v29 = v24;
  for (size_t i30 = v11; i30 < v1; i30 += v12) {
    size_t v31 = i30 * v4;
    size_t v32 = 8;
    float* v33 = &v3[v31];
    vfloat32m1_t v34 = __riscv_vle32_v_f32m1(v33, v32);
    size_t v35 = i30 * v6;
    size_t v36 = v35 + v12;
    size_t v37 = v35 + v13;
    size_t v38 = v35 + v14;
    size_t v39 = v35 + v15;
    vfloat32m1_t v40 = v25;
    float v41 = v5[v35];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v34, v32);
    vfloat32m1_t v43 = v26;
    float v44 = v5[v36];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v34, v32);
    vfloat32m1_t v46 = v27;
    float v47 = v5[v37];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v34, v32);
    vfloat32m1_t v49 = v28;
    float v50 = v5[v38];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v34, v32);
    vfloat32m1_t v52 = v29;
    float v53 = v5[v39];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v34, v32);
    v25 = v42;
    v26 = v45;
    v27 = v48;
    v28 = v51;
    v29 = v54;
  }
  vfloat32m1_t v55 = v25;
  vfloat32m1_t v56 = v26;
  vfloat32m1_t v57 = v27;
  vfloat32m1_t v58 = v28;
  vfloat32m1_t v59 = v29;
  float* v60 = &v8[v11];
  __riscv_vse32_v_f32m1(v60, v55, v19);
  float* v61 = &v8[v9];
  __riscv_vse32_v_f32m1(v61, v56, v19);
  size_t v62 = v13 * v9;
  float* v63 = &v8[v62];
  __riscv_vse32_v_f32m1(v63, v57, v19);
  size_t v64 = v14 * v9;
  float* v65 = &v8[v64];
  __riscv_vse32_v_f32m1(v65, v58, v19);
  size_t v66 = v15 * v9;
  float* v67 = &v8[v66];
  __riscv_vse32_v_f32m1(v67, v59, v19);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_8x5_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 8
//   NR         : 5
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_8x5_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v19);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v19);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v19);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v19);
  size_t v30 = v15 * v9;
  float* v31 = &v8[v30];
  vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v19);
  vfloat32m1_t v33;
  v33 = v21;
  vfloat32m1_t v34;
  v34 = v23;
  vfloat32m1_t v35;
  v35 = v26;
  vfloat32m1_t v36;
  v36 = v29;
  vfloat32m1_t v37;
  v37 = v32;
  for (size_t i38 = v11; i38 < v1; i38 += v12) {
    size_t v39 = i38 * v4;
    size_t v40 = 8;
    float* v41 = &v3[v39];
    vfloat32m1_t v42 = __riscv_vle32_v_f32m1(v41, v40);
    size_t v43 = i38 * v6;
    size_t v44 = v43 + v12;
    size_t v45 = v43 + v13;
    size_t v46 = v43 + v14;
    size_t v47 = v43 + v15;
    vfloat32m1_t v48 = v33;
    float v49 = v5[v43];
    vfloat32m1_t v50 = __riscv_vfmacc_vf_f32m1(v48, v49, v42, v40);
    vfloat32m1_t v51 = v34;
    float v52 = v5[v44];
    vfloat32m1_t v53 = __riscv_vfmacc_vf_f32m1(v51, v52, v42, v40);
    vfloat32m1_t v54 = v35;
    float v55 = v5[v45];
    vfloat32m1_t v56 = __riscv_vfmacc_vf_f32m1(v54, v55, v42, v40);
    vfloat32m1_t v57 = v36;
    float v58 = v5[v46];
    vfloat32m1_t v59 = __riscv_vfmacc_vf_f32m1(v57, v58, v42, v40);
    vfloat32m1_t v60 = v37;
    float v61 = v5[v47];
    vfloat32m1_t v62 = __riscv_vfmacc_vf_f32m1(v60, v61, v42, v40);
    v33 = v50;
    v34 = v53;
    v35 = v56;
    v36 = v59;
    v37 = v62;
  }
  vfloat32m1_t v63 = v33;
  vfloat32m1_t v64 = v34;
  vfloat32m1_t v65 = v35;
  vfloat32m1_t v66 = v36;
  vfloat32m1_t v67 = v37;
  float* v68 = &v8[v11];
  __riscv_vse32_v_f32m1(v68, v63, v19);
  float* v69 = &v8[v9];
  __riscv_vse32_v_f32m1(v69, v64, v19);
  size_t v70 = v13 * v9;
  float* v71 = &v8[v70];
  __riscv_vse32_v_f32m1(v71, v65, v19);
  size_t v72 = v14 * v9;
  float* v73 = &v8[v72];
  __riscv_vse32_v_f32m1(v73, v66, v19);
  size_t v74 = v15 * v9;
  float* v75 = &v8[v74];
  __riscv_vse32_v_f32m1(v75, v67, v19);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_8x6_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 8
//   NR         : 6
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_8x6_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v24 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v25 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v26;
  v26 = v20;
  vfloat32m1_t v27;
  v27 = v21;
  vfloat32m1_t v28;
  v28 = v22;
  vfloat32m1_t v29;
  v29 = v23;
  vfloat32m1_t v30;
  v30 = v24;
  vfloat32m1_t v31;
  v31 = v25;
  for (size_t i32 = v11; i32 < v1; i32 += v12) {
    size_t v33 = i32 * v4;
    size_t v34 = 8;
    float* v35 = &v3[v33];
    vfloat32m1_t v36 = __riscv_vle32_v_f32m1(v35, v34);
    size_t v37 = i32 * v6;
    size_t v38 = v37 + v12;
    size_t v39 = v37 + v13;
    size_t v40 = v37 + v14;
    size_t v41 = v37 + v15;
    size_t v42 = v37 + v16;
    vfloat32m1_t v43 = v26;
    float v44 = v5[v37];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v36, v34);
    vfloat32m1_t v46 = v27;
    float v47 = v5[v38];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v36, v34);
    vfloat32m1_t v49 = v28;
    float v50 = v5[v39];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v36, v34);
    vfloat32m1_t v52 = v29;
    float v53 = v5[v40];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v36, v34);
    vfloat32m1_t v55 = v30;
    float v56 = v5[v41];
    vfloat32m1_t v57 = __riscv_vfmacc_vf_f32m1(v55, v56, v36, v34);
    vfloat32m1_t v58 = v31;
    float v59 = v5[v42];
    vfloat32m1_t v60 = __riscv_vfmacc_vf_f32m1(v58, v59, v36, v34);
    v26 = v45;
    v27 = v48;
    v28 = v51;
    v29 = v54;
    v30 = v57;
    v31 = v60;
  }
  vfloat32m1_t v61 = v26;
  vfloat32m1_t v62 = v27;
  vfloat32m1_t v63 = v28;
  vfloat32m1_t v64 = v29;
  vfloat32m1_t v65 = v30;
  vfloat32m1_t v66 = v31;
  float* v67 = &v8[v11];
  __riscv_vse32_v_f32m1(v67, v61, v19);
  float* v68 = &v8[v9];
  __riscv_vse32_v_f32m1(v68, v62, v19);
  size_t v69 = v13 * v9;
  float* v70 = &v8[v69];
  __riscv_vse32_v_f32m1(v70, v63, v19);
  size_t v71 = v14 * v9;
  float* v72 = &v8[v71];
  __riscv_vse32_v_f32m1(v72, v64, v19);
  size_t v73 = v15 * v9;
  float* v74 = &v8[v73];
  __riscv_vse32_v_f32m1(v74, v65, v19);
  size_t v75 = v16 * v9;
  float* v76 = &v8[v75];
  __riscv_vse32_v_f32m1(v76, v66, v19);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_8x6_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 8
//   NR         : 6
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_8x6_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v19);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v19);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v19);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v19);
  size_t v30 = v15 * v9;
  float* v31 = &v8[v30];
  vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v19);
  size_t v33 = v16 * v9;
  float* v34 = &v8[v33];
  vfloat32m1_t v35 = __riscv_vle32_v_f32m1(v34, v19);
  vfloat32m1_t v36;
  v36 = v21;
  vfloat32m1_t v37;
  v37 = v23;
  vfloat32m1_t v38;
  v38 = v26;
  vfloat32m1_t v39;
  v39 = v29;
  vfloat32m1_t v40;
  v40 = v32;
  vfloat32m1_t v41;
  v41 = v35;
  for (size_t i42 = v11; i42 < v1; i42 += v12) {
    size_t v43 = i42 * v4;
    size_t v44 = 8;
    float* v45 = &v3[v43];
    vfloat32m1_t v46 = __riscv_vle32_v_f32m1(v45, v44);
    size_t v47 = i42 * v6;
    size_t v48 = v47 + v12;
    size_t v49 = v47 + v13;
    size_t v50 = v47 + v14;
    size_t v51 = v47 + v15;
    size_t v52 = v47 + v16;
    vfloat32m1_t v53 = v36;
    float v54 = v5[v47];
    vfloat32m1_t v55 = __riscv_vfmacc_vf_f32m1(v53, v54, v46, v44);
    vfloat32m1_t v56 = v37;
    float v57 = v5[v48];
    vfloat32m1_t v58 = __riscv_vfmacc_vf_f32m1(v56, v57, v46, v44);
    vfloat32m1_t v59 = v38;
    float v60 = v5[v49];
    vfloat32m1_t v61 = __riscv_vfmacc_vf_f32m1(v59, v60, v46, v44);
    vfloat32m1_t v62 = v39;
    float v63 = v5[v50];
    vfloat32m1_t v64 = __riscv_vfmacc_vf_f32m1(v62, v63, v46, v44);
    vfloat32m1_t v65 = v40;
    float v66 = v5[v51];
    vfloat32m1_t v67 = __riscv_vfmacc_vf_f32m1(v65, v66, v46, v44);
    vfloat32m1_t v68 = v41;
    float v69 = v5[v52];
    vfloat32m1_t v70 = __riscv_vfmacc_vf_f32m1(v68, v69, v46, v44);
    v36 = v55;
    v37 = v58;
    v38 = v61;
    v39 = v64;
    v40 = v67;
    v41 = v70;
  }
  vfloat32m1_t v71 = v36;
  vfloat32m1_t v72 = v37;
  vfloat32m1_t v73 = v38;
  vfloat32m1_t v74 = v39;
  vfloat32m1_t v75 = v40;
  vfloat32m1_t v76 = v41;
  float* v77 = &v8[v11];
  __riscv_vse32_v_f32m1(v77, v71, v19);
  float* v78 = &v8[v9];
  __riscv_vse32_v_f32m1(v78, v72, v19);
  size_t v79 = v13 * v9;
  float* v80 = &v8[v79];
  __riscv_vse32_v_f32m1(v80, v73, v19);
  size_t v81 = v14 * v9;
  float* v82 = &v8[v81];
  __riscv_vse32_v_f32m1(v82, v74, v19);
  size_t v83 = v15 * v9;
  float* v84 = &v8[v83];
  __riscv_vse32_v_f32m1(v84, v75, v19);
  size_t v85 = v16 * v9;
  float* v86 = &v8[v85];
  __riscv_vse32_v_f32m1(v86, v76, v19);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_8x7_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 8
//   NR         : 7
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_8x7_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v24 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v25 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v26 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v27;
  v27 = v20;
  vfloat32m1_t v28;
  v28 = v21;
  vfloat32m1_t v29;
  v29 = v22;
  vfloat32m1_t v30;
  v30 = v23;
  vfloat32m1_t v31;
  v31 = v24;
  vfloat32m1_t v32;
  v32 = v25;
  vfloat32m1_t v33;
  v33 = v26;
  for (size_t i34 = v11; i34 < v1; i34 += v12) {
    size_t v35 = i34 * v4;
    size_t v36 = 8;
    float* v37 = &v3[v35];
    vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v36);
    size_t v39 = i34 * v6;
    size_t v40 = v39 + v12;
    size_t v41 = v39 + v13;
    size_t v42 = v39 + v14;
    size_t v43 = v39 + v15;
    size_t v44 = v39 + v16;
    size_t v45 = v39 + v17;
    vfloat32m1_t v46 = v27;
    float v47 = v5[v39];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v38, v36);
    vfloat32m1_t v49 = v28;
    float v50 = v5[v40];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v38, v36);
    vfloat32m1_t v52 = v29;
    float v53 = v5[v41];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v38, v36);
    vfloat32m1_t v55 = v30;
    float v56 = v5[v42];
    vfloat32m1_t v57 = __riscv_vfmacc_vf_f32m1(v55, v56, v38, v36);
    vfloat32m1_t v58 = v31;
    float v59 = v5[v43];
    vfloat32m1_t v60 = __riscv_vfmacc_vf_f32m1(v58, v59, v38, v36);
    vfloat32m1_t v61 = v32;
    float v62 = v5[v44];
    vfloat32m1_t v63 = __riscv_vfmacc_vf_f32m1(v61, v62, v38, v36);
    vfloat32m1_t v64 = v33;
    float v65 = v5[v45];
    vfloat32m1_t v66 = __riscv_vfmacc_vf_f32m1(v64, v65, v38, v36);
    v27 = v48;
    v28 = v51;
    v29 = v54;
    v30 = v57;
    v31 = v60;
    v32 = v63;
    v33 = v66;
  }
  vfloat32m1_t v67 = v27;
  vfloat32m1_t v68 = v28;
  vfloat32m1_t v69 = v29;
  vfloat32m1_t v70 = v30;
  vfloat32m1_t v71 = v31;
  vfloat32m1_t v72 = v32;
  vfloat32m1_t v73 = v33;
  float* v74 = &v8[v11];
  __riscv_vse32_v_f32m1(v74, v67, v19);
  float* v75 = &v8[v9];
  __riscv_vse32_v_f32m1(v75, v68, v19);
  size_t v76 = v13 * v9;
  float* v77 = &v8[v76];
  __riscv_vse32_v_f32m1(v77, v69, v19);
  size_t v78 = v14 * v9;
  float* v79 = &v8[v78];
  __riscv_vse32_v_f32m1(v79, v70, v19);
  size_t v80 = v15 * v9;
  float* v81 = &v8[v80];
  __riscv_vse32_v_f32m1(v81, v71, v19);
  size_t v82 = v16 * v9;
  float* v83 = &v8[v82];
  __riscv_vse32_v_f32m1(v83, v72, v19);
  size_t v84 = v17 * v9;
  float* v85 = &v8[v84];
  __riscv_vse32_v_f32m1(v85, v73, v19);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_8x7_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 8
//   NR         : 7
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_8x7_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v19);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v19);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v19);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v19);
  size_t v30 = v15 * v9;
  float* v31 = &v8[v30];
  vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v19);
  size_t v33 = v16 * v9;
  float* v34 = &v8[v33];
  vfloat32m1_t v35 = __riscv_vle32_v_f32m1(v34, v19);
  size_t v36 = v17 * v9;
  float* v37 = &v8[v36];
  vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v19);
  vfloat32m1_t v39;
  v39 = v21;
  vfloat32m1_t v40;
  v40 = v23;
  vfloat32m1_t v41;
  v41 = v26;
  vfloat32m1_t v42;
  v42 = v29;
  vfloat32m1_t v43;
  v43 = v32;
  vfloat32m1_t v44;
  v44 = v35;
  vfloat32m1_t v45;
  v45 = v38;
  for (size_t i46 = v11; i46 < v1; i46 += v12) {
    size_t v47 = i46 * v4;
    size_t v48 = 8;
    float* v49 = &v3[v47];
    vfloat32m1_t v50 = __riscv_vle32_v_f32m1(v49, v48);
    size_t v51 = i46 * v6;
    size_t v52 = v51 + v12;
    size_t v53 = v51 + v13;
    size_t v54 = v51 + v14;
    size_t v55 = v51 + v15;
    size_t v56 = v51 + v16;
    size_t v57 = v51 + v17;
    vfloat32m1_t v58 = v39;
    float v59 = v5[v51];
    vfloat32m1_t v60 = __riscv_vfmacc_vf_f32m1(v58, v59, v50, v48);
    vfloat32m1_t v61 = v40;
    float v62 = v5[v52];
    vfloat32m1_t v63 = __riscv_vfmacc_vf_f32m1(v61, v62, v50, v48);
    vfloat32m1_t v64 = v41;
    float v65 = v5[v53];
    vfloat32m1_t v66 = __riscv_vfmacc_vf_f32m1(v64, v65, v50, v48);
    vfloat32m1_t v67 = v42;
    float v68 = v5[v54];
    vfloat32m1_t v69 = __riscv_vfmacc_vf_f32m1(v67, v68, v50, v48);
    vfloat32m1_t v70 = v43;
    float v71 = v5[v55];
    vfloat32m1_t v72 = __riscv_vfmacc_vf_f32m1(v70, v71, v50, v48);
    vfloat32m1_t v73 = v44;
    float v74 = v5[v56];
    vfloat32m1_t v75 = __riscv_vfmacc_vf_f32m1(v73, v74, v50, v48);
    vfloat32m1_t v76 = v45;
    float v77 = v5[v57];
    vfloat32m1_t v78 = __riscv_vfmacc_vf_f32m1(v76, v77, v50, v48);
    v39 = v60;
    v40 = v63;
    v41 = v66;
    v42 = v69;
    v43 = v72;
    v44 = v75;
    v45 = v78;
  }
  vfloat32m1_t v79 = v39;
  vfloat32m1_t v80 = v40;
  vfloat32m1_t v81 = v41;
  vfloat32m1_t v82 = v42;
  vfloat32m1_t v83 = v43;
  vfloat32m1_t v84 = v44;
  vfloat32m1_t v85 = v45;
  float* v86 = &v8[v11];
  __riscv_vse32_v_f32m1(v86, v79, v19);
  float* v87 = &v8[v9];
  __riscv_vse32_v_f32m1(v87, v80, v19);
  size_t v88 = v13 * v9;
  float* v89 = &v8[v88];
  __riscv_vse32_v_f32m1(v89, v81, v19);
  size_t v90 = v14 * v9;
  float* v91 = &v8[v90];
  __riscv_vse32_v_f32m1(v91, v82, v19);
  size_t v92 = v15 * v9;
  float* v93 = &v8[v92];
  __riscv_vse32_v_f32m1(v93, v83, v19);
  size_t v94 = v16 * v9;
  float* v95 = &v8[v94];
  __riscv_vse32_v_f32m1(v95, v84, v19);
  size_t v96 = v17 * v9;
  float* v97 = &v8[v96];
  __riscv_vse32_v_f32m1(v97, v85, v19);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_8x8_b0_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 8
//   NR         : 8
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_8x8_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v24 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v25 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v26 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v27 = __riscv_vfmv_v_f_f32m1(v10, v19);
  vfloat32m1_t v28;
  v28 = v20;
  vfloat32m1_t v29;
  v29 = v21;
  vfloat32m1_t v30;
  v30 = v22;
  vfloat32m1_t v31;
  v31 = v23;
  vfloat32m1_t v32;
  v32 = v24;
  vfloat32m1_t v33;
  v33 = v25;
  vfloat32m1_t v34;
  v34 = v26;
  vfloat32m1_t v35;
  v35 = v27;
  for (size_t i36 = v11; i36 < v1; i36 += v12) {
    size_t v37 = i36 * v4;
    size_t v38 = 8;
    float* v39 = &v3[v37];
    vfloat32m1_t v40 = __riscv_vle32_v_f32m1(v39, v38);
    size_t v41 = i36 * v6;
    size_t v42 = v41 + v12;
    size_t v43 = v41 + v13;
    size_t v44 = v41 + v14;
    size_t v45 = v41 + v15;
    size_t v46 = v41 + v16;
    size_t v47 = v41 + v17;
    size_t v48 = v41 + v18;
    vfloat32m1_t v49 = v28;
    float v50 = v5[v41];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v40, v38);
    vfloat32m1_t v52 = v29;
    float v53 = v5[v42];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v40, v38);
    vfloat32m1_t v55 = v30;
    float v56 = v5[v43];
    vfloat32m1_t v57 = __riscv_vfmacc_vf_f32m1(v55, v56, v40, v38);
    vfloat32m1_t v58 = v31;
    float v59 = v5[v44];
    vfloat32m1_t v60 = __riscv_vfmacc_vf_f32m1(v58, v59, v40, v38);
    vfloat32m1_t v61 = v32;
    float v62 = v5[v45];
    vfloat32m1_t v63 = __riscv_vfmacc_vf_f32m1(v61, v62, v40, v38);
    vfloat32m1_t v64 = v33;
    float v65 = v5[v46];
    vfloat32m1_t v66 = __riscv_vfmacc_vf_f32m1(v64, v65, v40, v38);
    vfloat32m1_t v67 = v34;
    float v68 = v5[v47];
    vfloat32m1_t v69 = __riscv_vfmacc_vf_f32m1(v67, v68, v40, v38);
    vfloat32m1_t v70 = v35;
    float v71 = v5[v48];
    vfloat32m1_t v72 = __riscv_vfmacc_vf_f32m1(v70, v71, v40, v38);
    v28 = v51;
    v29 = v54;
    v30 = v57;
    v31 = v60;
    v32 = v63;
    v33 = v66;
    v34 = v69;
    v35 = v72;
  }
  vfloat32m1_t v73 = v28;
  vfloat32m1_t v74 = v29;
  vfloat32m1_t v75 = v30;
  vfloat32m1_t v76 = v31;
  vfloat32m1_t v77 = v32;
  vfloat32m1_t v78 = v33;
  vfloat32m1_t v79 = v34;
  vfloat32m1_t v80 = v35;
  float* v81 = &v8[v11];
  __riscv_vse32_v_f32m1(v81, v73, v19);
  float* v82 = &v8[v9];
  __riscv_vse32_v_f32m1(v82, v74, v19);
  size_t v83 = v13 * v9;
  float* v84 = &v8[v83];
  __riscv_vse32_v_f32m1(v84, v75, v19);
  size_t v85 = v14 * v9;
  float* v86 = &v8[v85];
  __riscv_vse32_v_f32m1(v86, v76, v19);
  size_t v87 = v15 * v9;
  float* v88 = &v8[v87];
  __riscv_vse32_v_f32m1(v88, v77, v19);
  size_t v89 = v16 * v9;
  float* v90 = &v8[v89];
  __riscv_vse32_v_f32m1(v90, v78, v19);
  size_t v91 = v17 * v9;
  float* v92 = &v8[v91];
  __riscv_vse32_v_f32m1(v92, v79, v19);
  size_t v93 = v18 * v9;
  float* v94 = &v8[v93];
  __riscv_vse32_v_f32m1(v94, v80, v19);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_8x8_b1_fp32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-04-17 10:24:59
//   MR         : 8
//   NR         : 8
//   VLEN       : 8  (fp32 elements per vector register)
//   Precision  : fp32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_8x8_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
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
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v19);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v19);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v19);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v19);
  size_t v30 = v15 * v9;
  float* v31 = &v8[v30];
  vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v19);
  size_t v33 = v16 * v9;
  float* v34 = &v8[v33];
  vfloat32m1_t v35 = __riscv_vle32_v_f32m1(v34, v19);
  size_t v36 = v17 * v9;
  float* v37 = &v8[v36];
  vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v19);
  size_t v39 = v18 * v9;
  float* v40 = &v8[v39];
  vfloat32m1_t v41 = __riscv_vle32_v_f32m1(v40, v19);
  vfloat32m1_t v42;
  v42 = v21;
  vfloat32m1_t v43;
  v43 = v23;
  vfloat32m1_t v44;
  v44 = v26;
  vfloat32m1_t v45;
  v45 = v29;
  vfloat32m1_t v46;
  v46 = v32;
  vfloat32m1_t v47;
  v47 = v35;
  vfloat32m1_t v48;
  v48 = v38;
  vfloat32m1_t v49;
  v49 = v41;
  for (size_t i50 = v11; i50 < v1; i50 += v12) {
    size_t v51 = i50 * v4;
    size_t v52 = 8;
    float* v53 = &v3[v51];
    vfloat32m1_t v54 = __riscv_vle32_v_f32m1(v53, v52);
    size_t v55 = i50 * v6;
    size_t v56 = v55 + v12;
    size_t v57 = v55 + v13;
    size_t v58 = v55 + v14;
    size_t v59 = v55 + v15;
    size_t v60 = v55 + v16;
    size_t v61 = v55 + v17;
    size_t v62 = v55 + v18;
    vfloat32m1_t v63 = v42;
    float v64 = v5[v55];
    vfloat32m1_t v65 = __riscv_vfmacc_vf_f32m1(v63, v64, v54, v52);
    vfloat32m1_t v66 = v43;
    float v67 = v5[v56];
    vfloat32m1_t v68 = __riscv_vfmacc_vf_f32m1(v66, v67, v54, v52);
    vfloat32m1_t v69 = v44;
    float v70 = v5[v57];
    vfloat32m1_t v71 = __riscv_vfmacc_vf_f32m1(v69, v70, v54, v52);
    vfloat32m1_t v72 = v45;
    float v73 = v5[v58];
    vfloat32m1_t v74 = __riscv_vfmacc_vf_f32m1(v72, v73, v54, v52);
    vfloat32m1_t v75 = v46;
    float v76 = v5[v59];
    vfloat32m1_t v77 = __riscv_vfmacc_vf_f32m1(v75, v76, v54, v52);
    vfloat32m1_t v78 = v47;
    float v79 = v5[v60];
    vfloat32m1_t v80 = __riscv_vfmacc_vf_f32m1(v78, v79, v54, v52);
    vfloat32m1_t v81 = v48;
    float v82 = v5[v61];
    vfloat32m1_t v83 = __riscv_vfmacc_vf_f32m1(v81, v82, v54, v52);
    vfloat32m1_t v84 = v49;
    float v85 = v5[v62];
    vfloat32m1_t v86 = __riscv_vfmacc_vf_f32m1(v84, v85, v54, v52);
    v42 = v65;
    v43 = v68;
    v44 = v71;
    v45 = v74;
    v46 = v77;
    v47 = v80;
    v48 = v83;
    v49 = v86;
  }
  vfloat32m1_t v87 = v42;
  vfloat32m1_t v88 = v43;
  vfloat32m1_t v89 = v44;
  vfloat32m1_t v90 = v45;
  vfloat32m1_t v91 = v46;
  vfloat32m1_t v92 = v47;
  vfloat32m1_t v93 = v48;
  vfloat32m1_t v94 = v49;
  float* v95 = &v8[v11];
  __riscv_vse32_v_f32m1(v95, v87, v19);
  float* v96 = &v8[v9];
  __riscv_vse32_v_f32m1(v96, v88, v19);
  size_t v97 = v13 * v9;
  float* v98 = &v8[v97];
  __riscv_vse32_v_f32m1(v98, v89, v19);
  size_t v99 = v14 * v9;
  float* v100 = &v8[v99];
  __riscv_vse32_v_f32m1(v100, v90, v19);
  size_t v101 = v15 * v9;
  float* v102 = &v8[v101];
  __riscv_vse32_v_f32m1(v102, v91, v19);
  size_t v103 = v16 * v9;
  float* v104 = &v8[v103];
  __riscv_vse32_v_f32m1(v104, v92, v19);
  size_t v105 = v17 * v9;
  float* v106 = &v8[v105];
  __riscv_vse32_v_f32m1(v106, v93, v19);
  size_t v107 = v18 * v9;
  float* v108 = &v8[v107];
  __riscv_vse32_v_f32m1(v108, v94, v19);
  return;
}



