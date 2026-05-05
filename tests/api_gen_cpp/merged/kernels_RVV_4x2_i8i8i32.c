// ============================================================
// xDSL Generated Merged Kernels
// ------------------------------------------------------------
// Precision : i8i8i32
// Types     : A=int8_t, B=int8_t, C=int32_t
// Generation: 2026-05-05 19:51:58
// ============================================================
#include "kernels_RVV_4x2_i8i8i32.h"
#include <riscv_vector.h>

// ------------------------------------------------------------
// Merged from xdsl_api_gemm_1x1_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-05 19:51:58
//   MR         : 1
//   NR         : 1
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_1x1_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
  size_t v10 = 0;
  size_t v11 = 1;
  size_t v12 = 2;
  size_t v13 = 3;
  size_t v14 = 4;
  size_t v15 = 5;
  size_t v16 = 6;
  size_t v17 = 7;
  size_t v18 = 8;
  int16_t v19 = 0;
  int32_t v20 = 0;
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v11);
  vint32m1_t v22;
  v22 = v21;
  for (size_t i23 = v10; i23 < v1; i23 += v11) {
    size_t v24 = i23 * v4;
    size_t v25 = 1;
    int8_t* v26 = &v3[v24];
    vint8mf4_t v27 = __riscv_vle8_v_i8mf4(v26, v25);
    vint16mf2_t v28 = __riscv_vsext_vf2_i16mf2(v27, v25);
    size_t v29 = i23 * v6;
    int8_t v30 = v5[v29];
    int16_t v31 = (int16_t) v30;
    vint32m1_t v32 = v22;
    vint32m1_t v33 = __riscv_vwmacc_vx_i32m1(v32, v31, v28, v25);
    v22 = v33;
  }
  vint32m1_t v34 = v22;
  int32_t* v35 = &v8[v10];
  __riscv_vse32_v_i32m1(v35, v34, v11);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_1x1_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-05 19:51:58
//   MR         : 1
//   NR         : 1
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_1x1_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
  size_t v10 = 0;
  size_t v11 = 1;
  size_t v12 = 2;
  size_t v13 = 3;
  size_t v14 = 4;
  size_t v15 = 5;
  size_t v16 = 6;
  size_t v17 = 7;
  size_t v18 = 8;
  int16_t v19 = 0;
  int32_t v20 = 0;
  size_t v21 = v10 * v9;
  size_t v22 = v10 + v21;
  int32_t* v23 = &v8[v22];
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v11);
  vint32m1_t v25;
  v25 = v24;
  for (size_t i26 = v10; i26 < v1; i26 += v11) {
    size_t v27 = i26 * v4;
    size_t v28 = 1;
    int8_t* v29 = &v3[v27];
    vint8mf4_t v30 = __riscv_vle8_v_i8mf4(v29, v28);
    vint16mf2_t v31 = __riscv_vsext_vf2_i16mf2(v30, v28);
    size_t v32 = i26 * v6;
    int8_t v33 = v5[v32];
    int16_t v34 = (int16_t) v33;
    vint32m1_t v35 = v25;
    vint32m1_t v36 = __riscv_vwmacc_vx_i32m1(v35, v34, v31, v28);
    v25 = v36;
  }
  vint32m1_t v37 = v25;
  int32_t* v38 = &v8[v10];
  __riscv_vse32_v_i32m1(v38, v37, v11);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_1x2_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-05 19:51:58
//   MR         : 1
//   NR         : 2
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_1x2_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
  size_t v10 = 0;
  size_t v11 = 1;
  size_t v12 = 2;
  size_t v13 = 3;
  size_t v14 = 4;
  size_t v15 = 5;
  size_t v16 = 6;
  size_t v17 = 7;
  size_t v18 = 8;
  int16_t v19 = 0;
  int32_t v20 = 0;
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v11);
  vint32m1_t v22 = __riscv_vmv_v_x_i32m1(v20, v11);
  vint32m1_t v23;
  v23 = v21;
  vint32m1_t v24;
  v24 = v22;
  for (size_t i25 = v10; i25 < v1; i25 += v11) {
    size_t v26 = i25 * v4;
    size_t v27 = 1;
    int8_t* v28 = &v3[v26];
    vint8mf4_t v29 = __riscv_vle8_v_i8mf4(v28, v27);
    vint16mf2_t v30 = __riscv_vsext_vf2_i16mf2(v29, v27);
    size_t v31 = i25 * v6;
    size_t v32 = v31 + v11;
    int8_t v33 = v5[v31];
    int16_t v34 = (int16_t) v33;
    vint32m1_t v35 = v23;
    vint32m1_t v36 = __riscv_vwmacc_vx_i32m1(v35, v34, v30, v27);
    int8_t v37 = v5[v32];
    int16_t v38 = (int16_t) v37;
    vint32m1_t v39 = v24;
    vint32m1_t v40 = __riscv_vwmacc_vx_i32m1(v39, v38, v30, v27);
    v23 = v36;
    v24 = v40;
  }
  vint32m1_t v41 = v23;
  vint32m1_t v42 = v24;
  int32_t* v43 = &v8[v10];
  __riscv_vse32_v_i32m1(v43, v41, v11);
  int32_t* v44 = &v8[v9];
  __riscv_vse32_v_i32m1(v44, v42, v11);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_1x2_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-05 19:51:58
//   MR         : 1
//   NR         : 2
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_1x2_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
  size_t v10 = 0;
  size_t v11 = 1;
  size_t v12 = 2;
  size_t v13 = 3;
  size_t v14 = 4;
  size_t v15 = 5;
  size_t v16 = 6;
  size_t v17 = 7;
  size_t v18 = 8;
  int16_t v19 = 0;
  int32_t v20 = 0;
  size_t v21 = v10 * v9;
  size_t v22 = v10 + v21;
  int32_t* v23 = &v8[v22];
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v11);
  size_t v25 = v11 * v9;
  size_t v26 = v10 + v25;
  int32_t* v27 = &v8[v26];
  vint32m1_t v28 = __riscv_vle32_v_i32m1(v27, v11);
  vint32m1_t v29;
  v29 = v24;
  vint32m1_t v30;
  v30 = v28;
  for (size_t i31 = v10; i31 < v1; i31 += v11) {
    size_t v32 = i31 * v4;
    size_t v33 = 1;
    int8_t* v34 = &v3[v32];
    vint8mf4_t v35 = __riscv_vle8_v_i8mf4(v34, v33);
    vint16mf2_t v36 = __riscv_vsext_vf2_i16mf2(v35, v33);
    size_t v37 = i31 * v6;
    size_t v38 = v37 + v11;
    int8_t v39 = v5[v37];
    int16_t v40 = (int16_t) v39;
    vint32m1_t v41 = v29;
    vint32m1_t v42 = __riscv_vwmacc_vx_i32m1(v41, v40, v36, v33);
    int8_t v43 = v5[v38];
    int16_t v44 = (int16_t) v43;
    vint32m1_t v45 = v30;
    vint32m1_t v46 = __riscv_vwmacc_vx_i32m1(v45, v44, v36, v33);
    v29 = v42;
    v30 = v46;
  }
  vint32m1_t v47 = v29;
  vint32m1_t v48 = v30;
  int32_t* v49 = &v8[v10];
  __riscv_vse32_v_i32m1(v49, v47, v11);
  int32_t* v50 = &v8[v9];
  __riscv_vse32_v_i32m1(v50, v48, v11);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_2x1_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-05 19:51:58
//   MR         : 2
//   NR         : 1
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_2x1_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
  size_t v10 = 0;
  size_t v11 = 1;
  size_t v12 = 2;
  size_t v13 = 3;
  size_t v14 = 4;
  size_t v15 = 5;
  size_t v16 = 6;
  size_t v17 = 7;
  size_t v18 = 8;
  int16_t v19 = 0;
  int32_t v20 = 0;
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v12);
  vint32m1_t v22;
  v22 = v21;
  for (size_t i23 = v10; i23 < v1; i23 += v11) {
    size_t v24 = i23 * v4;
    size_t v25 = 2;
    int8_t* v26 = &v3[v24];
    vint8mf4_t v27 = __riscv_vle8_v_i8mf4(v26, v25);
    vint16mf2_t v28 = __riscv_vsext_vf2_i16mf2(v27, v25);
    size_t v29 = i23 * v6;
    int8_t v30 = v5[v29];
    int16_t v31 = (int16_t) v30;
    vint32m1_t v32 = v22;
    vint32m1_t v33 = __riscv_vwmacc_vx_i32m1(v32, v31, v28, v25);
    v22 = v33;
  }
  vint32m1_t v34 = v22;
  int32_t* v35 = &v8[v10];
  __riscv_vse32_v_i32m1(v35, v34, v12);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_2x1_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-05 19:51:58
//   MR         : 2
//   NR         : 1
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_2x1_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
  size_t v10 = 0;
  size_t v11 = 1;
  size_t v12 = 2;
  size_t v13 = 3;
  size_t v14 = 4;
  size_t v15 = 5;
  size_t v16 = 6;
  size_t v17 = 7;
  size_t v18 = 8;
  int16_t v19 = 0;
  int32_t v20 = 0;
  size_t v21 = v10 * v9;
  size_t v22 = v10 + v21;
  int32_t* v23 = &v8[v22];
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v12);
  vint32m1_t v25;
  v25 = v24;
  for (size_t i26 = v10; i26 < v1; i26 += v11) {
    size_t v27 = i26 * v4;
    size_t v28 = 2;
    int8_t* v29 = &v3[v27];
    vint8mf4_t v30 = __riscv_vle8_v_i8mf4(v29, v28);
    vint16mf2_t v31 = __riscv_vsext_vf2_i16mf2(v30, v28);
    size_t v32 = i26 * v6;
    int8_t v33 = v5[v32];
    int16_t v34 = (int16_t) v33;
    vint32m1_t v35 = v25;
    vint32m1_t v36 = __riscv_vwmacc_vx_i32m1(v35, v34, v31, v28);
    v25 = v36;
  }
  vint32m1_t v37 = v25;
  int32_t* v38 = &v8[v10];
  __riscv_vse32_v_i32m1(v38, v37, v12);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_2x2_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-05 19:51:58
//   MR         : 2
//   NR         : 2
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_2x2_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
  size_t v10 = 0;
  size_t v11 = 1;
  size_t v12 = 2;
  size_t v13 = 3;
  size_t v14 = 4;
  size_t v15 = 5;
  size_t v16 = 6;
  size_t v17 = 7;
  size_t v18 = 8;
  int16_t v19 = 0;
  int32_t v20 = 0;
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v12);
  vint32m1_t v22 = __riscv_vmv_v_x_i32m1(v20, v12);
  vint32m1_t v23;
  v23 = v21;
  vint32m1_t v24;
  v24 = v22;
  for (size_t i25 = v10; i25 < v1; i25 += v11) {
    size_t v26 = i25 * v4;
    size_t v27 = 2;
    int8_t* v28 = &v3[v26];
    vint8mf4_t v29 = __riscv_vle8_v_i8mf4(v28, v27);
    vint16mf2_t v30 = __riscv_vsext_vf2_i16mf2(v29, v27);
    size_t v31 = i25 * v6;
    size_t v32 = v31 + v11;
    int8_t v33 = v5[v31];
    int16_t v34 = (int16_t) v33;
    vint32m1_t v35 = v23;
    vint32m1_t v36 = __riscv_vwmacc_vx_i32m1(v35, v34, v30, v27);
    int8_t v37 = v5[v32];
    int16_t v38 = (int16_t) v37;
    vint32m1_t v39 = v24;
    vint32m1_t v40 = __riscv_vwmacc_vx_i32m1(v39, v38, v30, v27);
    v23 = v36;
    v24 = v40;
  }
  vint32m1_t v41 = v23;
  vint32m1_t v42 = v24;
  int32_t* v43 = &v8[v10];
  __riscv_vse32_v_i32m1(v43, v41, v12);
  int32_t* v44 = &v8[v9];
  __riscv_vse32_v_i32m1(v44, v42, v12);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_2x2_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-05 19:51:58
//   MR         : 2
//   NR         : 2
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_2x2_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
  size_t v10 = 0;
  size_t v11 = 1;
  size_t v12 = 2;
  size_t v13 = 3;
  size_t v14 = 4;
  size_t v15 = 5;
  size_t v16 = 6;
  size_t v17 = 7;
  size_t v18 = 8;
  int16_t v19 = 0;
  int32_t v20 = 0;
  size_t v21 = v10 * v9;
  size_t v22 = v10 + v21;
  int32_t* v23 = &v8[v22];
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v12);
  size_t v25 = v11 * v9;
  size_t v26 = v10 + v25;
  int32_t* v27 = &v8[v26];
  vint32m1_t v28 = __riscv_vle32_v_i32m1(v27, v12);
  vint32m1_t v29;
  v29 = v24;
  vint32m1_t v30;
  v30 = v28;
  for (size_t i31 = v10; i31 < v1; i31 += v11) {
    size_t v32 = i31 * v4;
    size_t v33 = 2;
    int8_t* v34 = &v3[v32];
    vint8mf4_t v35 = __riscv_vle8_v_i8mf4(v34, v33);
    vint16mf2_t v36 = __riscv_vsext_vf2_i16mf2(v35, v33);
    size_t v37 = i31 * v6;
    size_t v38 = v37 + v11;
    int8_t v39 = v5[v37];
    int16_t v40 = (int16_t) v39;
    vint32m1_t v41 = v29;
    vint32m1_t v42 = __riscv_vwmacc_vx_i32m1(v41, v40, v36, v33);
    int8_t v43 = v5[v38];
    int16_t v44 = (int16_t) v43;
    vint32m1_t v45 = v30;
    vint32m1_t v46 = __riscv_vwmacc_vx_i32m1(v45, v44, v36, v33);
    v29 = v42;
    v30 = v46;
  }
  vint32m1_t v47 = v29;
  vint32m1_t v48 = v30;
  int32_t* v49 = &v8[v10];
  __riscv_vse32_v_i32m1(v49, v47, v12);
  int32_t* v50 = &v8[v9];
  __riscv_vse32_v_i32m1(v50, v48, v12);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_3x1_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-05 19:51:58
//   MR         : 3
//   NR         : 1
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_3x1_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
  size_t v10 = 0;
  size_t v11 = 1;
  size_t v12 = 2;
  size_t v13 = 3;
  size_t v14 = 4;
  size_t v15 = 5;
  size_t v16 = 6;
  size_t v17 = 7;
  size_t v18 = 8;
  int16_t v19 = 0;
  int32_t v20 = 0;
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v13);
  vint32m1_t v22;
  v22 = v21;
  for (size_t i23 = v10; i23 < v1; i23 += v11) {
    size_t v24 = i23 * v4;
    size_t v25 = 3;
    int8_t* v26 = &v3[v24];
    vint8mf4_t v27 = __riscv_vle8_v_i8mf4(v26, v25);
    vint16mf2_t v28 = __riscv_vsext_vf2_i16mf2(v27, v25);
    size_t v29 = i23 * v6;
    int8_t v30 = v5[v29];
    int16_t v31 = (int16_t) v30;
    vint32m1_t v32 = v22;
    vint32m1_t v33 = __riscv_vwmacc_vx_i32m1(v32, v31, v28, v25);
    v22 = v33;
  }
  vint32m1_t v34 = v22;
  int32_t* v35 = &v8[v10];
  __riscv_vse32_v_i32m1(v35, v34, v13);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_3x1_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-05 19:51:58
//   MR         : 3
//   NR         : 1
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_3x1_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
  size_t v10 = 0;
  size_t v11 = 1;
  size_t v12 = 2;
  size_t v13 = 3;
  size_t v14 = 4;
  size_t v15 = 5;
  size_t v16 = 6;
  size_t v17 = 7;
  size_t v18 = 8;
  int16_t v19 = 0;
  int32_t v20 = 0;
  size_t v21 = v10 * v9;
  size_t v22 = v10 + v21;
  int32_t* v23 = &v8[v22];
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v13);
  vint32m1_t v25;
  v25 = v24;
  for (size_t i26 = v10; i26 < v1; i26 += v11) {
    size_t v27 = i26 * v4;
    size_t v28 = 3;
    int8_t* v29 = &v3[v27];
    vint8mf4_t v30 = __riscv_vle8_v_i8mf4(v29, v28);
    vint16mf2_t v31 = __riscv_vsext_vf2_i16mf2(v30, v28);
    size_t v32 = i26 * v6;
    int8_t v33 = v5[v32];
    int16_t v34 = (int16_t) v33;
    vint32m1_t v35 = v25;
    vint32m1_t v36 = __riscv_vwmacc_vx_i32m1(v35, v34, v31, v28);
    v25 = v36;
  }
  vint32m1_t v37 = v25;
  int32_t* v38 = &v8[v10];
  __riscv_vse32_v_i32m1(v38, v37, v13);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_3x2_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-05 19:51:58
//   MR         : 3
//   NR         : 2
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_3x2_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
  size_t v10 = 0;
  size_t v11 = 1;
  size_t v12 = 2;
  size_t v13 = 3;
  size_t v14 = 4;
  size_t v15 = 5;
  size_t v16 = 6;
  size_t v17 = 7;
  size_t v18 = 8;
  int16_t v19 = 0;
  int32_t v20 = 0;
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v13);
  vint32m1_t v22 = __riscv_vmv_v_x_i32m1(v20, v13);
  vint32m1_t v23;
  v23 = v21;
  vint32m1_t v24;
  v24 = v22;
  for (size_t i25 = v10; i25 < v1; i25 += v11) {
    size_t v26 = i25 * v4;
    size_t v27 = 3;
    int8_t* v28 = &v3[v26];
    vint8mf4_t v29 = __riscv_vle8_v_i8mf4(v28, v27);
    vint16mf2_t v30 = __riscv_vsext_vf2_i16mf2(v29, v27);
    size_t v31 = i25 * v6;
    size_t v32 = v31 + v11;
    int8_t v33 = v5[v31];
    int16_t v34 = (int16_t) v33;
    vint32m1_t v35 = v23;
    vint32m1_t v36 = __riscv_vwmacc_vx_i32m1(v35, v34, v30, v27);
    int8_t v37 = v5[v32];
    int16_t v38 = (int16_t) v37;
    vint32m1_t v39 = v24;
    vint32m1_t v40 = __riscv_vwmacc_vx_i32m1(v39, v38, v30, v27);
    v23 = v36;
    v24 = v40;
  }
  vint32m1_t v41 = v23;
  vint32m1_t v42 = v24;
  int32_t* v43 = &v8[v10];
  __riscv_vse32_v_i32m1(v43, v41, v13);
  int32_t* v44 = &v8[v9];
  __riscv_vse32_v_i32m1(v44, v42, v13);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_3x2_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-05 19:51:58
//   MR         : 3
//   NR         : 2
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_3x2_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
  size_t v10 = 0;
  size_t v11 = 1;
  size_t v12 = 2;
  size_t v13 = 3;
  size_t v14 = 4;
  size_t v15 = 5;
  size_t v16 = 6;
  size_t v17 = 7;
  size_t v18 = 8;
  int16_t v19 = 0;
  int32_t v20 = 0;
  size_t v21 = v10 * v9;
  size_t v22 = v10 + v21;
  int32_t* v23 = &v8[v22];
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v13);
  size_t v25 = v11 * v9;
  size_t v26 = v10 + v25;
  int32_t* v27 = &v8[v26];
  vint32m1_t v28 = __riscv_vle32_v_i32m1(v27, v13);
  vint32m1_t v29;
  v29 = v24;
  vint32m1_t v30;
  v30 = v28;
  for (size_t i31 = v10; i31 < v1; i31 += v11) {
    size_t v32 = i31 * v4;
    size_t v33 = 3;
    int8_t* v34 = &v3[v32];
    vint8mf4_t v35 = __riscv_vle8_v_i8mf4(v34, v33);
    vint16mf2_t v36 = __riscv_vsext_vf2_i16mf2(v35, v33);
    size_t v37 = i31 * v6;
    size_t v38 = v37 + v11;
    int8_t v39 = v5[v37];
    int16_t v40 = (int16_t) v39;
    vint32m1_t v41 = v29;
    vint32m1_t v42 = __riscv_vwmacc_vx_i32m1(v41, v40, v36, v33);
    int8_t v43 = v5[v38];
    int16_t v44 = (int16_t) v43;
    vint32m1_t v45 = v30;
    vint32m1_t v46 = __riscv_vwmacc_vx_i32m1(v45, v44, v36, v33);
    v29 = v42;
    v30 = v46;
  }
  vint32m1_t v47 = v29;
  vint32m1_t v48 = v30;
  int32_t* v49 = &v8[v10];
  __riscv_vse32_v_i32m1(v49, v47, v13);
  int32_t* v50 = &v8[v9];
  __riscv_vse32_v_i32m1(v50, v48, v13);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_4x1_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-05 19:51:58
//   MR         : 4
//   NR         : 1
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_4x1_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
  size_t v10 = 0;
  size_t v11 = 1;
  size_t v12 = 2;
  size_t v13 = 3;
  size_t v14 = 4;
  size_t v15 = 5;
  size_t v16 = 6;
  size_t v17 = 7;
  size_t v18 = 8;
  int16_t v19 = 0;
  int32_t v20 = 0;
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v14);
  vint32m1_t v22;
  v22 = v21;
  for (size_t i23 = v10; i23 < v1; i23 += v11) {
    size_t v24 = i23 * v4;
    size_t v25 = 4;
    int8_t* v26 = &v3[v24];
    vint8mf4_t v27 = __riscv_vle8_v_i8mf4(v26, v25);
    vint16mf2_t v28 = __riscv_vsext_vf2_i16mf2(v27, v25);
    size_t v29 = i23 * v6;
    int8_t v30 = v5[v29];
    int16_t v31 = (int16_t) v30;
    vint32m1_t v32 = v22;
    vint32m1_t v33 = __riscv_vwmacc_vx_i32m1(v32, v31, v28, v25);
    v22 = v33;
  }
  vint32m1_t v34 = v22;
  int32_t* v35 = &v8[v10];
  __riscv_vse32_v_i32m1(v35, v34, v14);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_4x1_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-05 19:51:58
//   MR         : 4
//   NR         : 1
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_4x1_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
  size_t v10 = 0;
  size_t v11 = 1;
  size_t v12 = 2;
  size_t v13 = 3;
  size_t v14 = 4;
  size_t v15 = 5;
  size_t v16 = 6;
  size_t v17 = 7;
  size_t v18 = 8;
  int16_t v19 = 0;
  int32_t v20 = 0;
  size_t v21 = v10 * v9;
  size_t v22 = v10 + v21;
  int32_t* v23 = &v8[v22];
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v14);
  vint32m1_t v25;
  v25 = v24;
  for (size_t i26 = v10; i26 < v1; i26 += v11) {
    size_t v27 = i26 * v4;
    size_t v28 = 4;
    int8_t* v29 = &v3[v27];
    vint8mf4_t v30 = __riscv_vle8_v_i8mf4(v29, v28);
    vint16mf2_t v31 = __riscv_vsext_vf2_i16mf2(v30, v28);
    size_t v32 = i26 * v6;
    int8_t v33 = v5[v32];
    int16_t v34 = (int16_t) v33;
    vint32m1_t v35 = v25;
    vint32m1_t v36 = __riscv_vwmacc_vx_i32m1(v35, v34, v31, v28);
    v25 = v36;
  }
  vint32m1_t v37 = v25;
  int32_t* v38 = &v8[v10];
  __riscv_vse32_v_i32m1(v38, v37, v14);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_4x2_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-05 19:51:58
//   MR         : 4
//   NR         : 2
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_4x2_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
  size_t v10 = 0;
  size_t v11 = 1;
  size_t v12 = 2;
  size_t v13 = 3;
  size_t v14 = 4;
  size_t v15 = 5;
  size_t v16 = 6;
  size_t v17 = 7;
  size_t v18 = 8;
  int16_t v19 = 0;
  int32_t v20 = 0;
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v14);
  vint32m1_t v22 = __riscv_vmv_v_x_i32m1(v20, v14);
  vint32m1_t v23;
  v23 = v21;
  vint32m1_t v24;
  v24 = v22;
  for (size_t i25 = v10; i25 < v1; i25 += v11) {
    size_t v26 = i25 * v4;
    size_t v27 = 4;
    int8_t* v28 = &v3[v26];
    vint8mf4_t v29 = __riscv_vle8_v_i8mf4(v28, v27);
    vint16mf2_t v30 = __riscv_vsext_vf2_i16mf2(v29, v27);
    size_t v31 = i25 * v6;
    size_t v32 = v31 + v11;
    int8_t v33 = v5[v31];
    int16_t v34 = (int16_t) v33;
    vint32m1_t v35 = v23;
    vint32m1_t v36 = __riscv_vwmacc_vx_i32m1(v35, v34, v30, v27);
    int8_t v37 = v5[v32];
    int16_t v38 = (int16_t) v37;
    vint32m1_t v39 = v24;
    vint32m1_t v40 = __riscv_vwmacc_vx_i32m1(v39, v38, v30, v27);
    v23 = v36;
    v24 = v40;
  }
  vint32m1_t v41 = v23;
  vint32m1_t v42 = v24;
  int32_t* v43 = &v8[v10];
  __riscv_vse32_v_i32m1(v43, v41, v14);
  int32_t* v44 = &v8[v9];
  __riscv_vse32_v_i32m1(v44, v42, v14);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_4x2_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-05 19:51:58
//   MR         : 4
//   NR         : 2
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_4x2_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
  size_t v10 = 0;
  size_t v11 = 1;
  size_t v12 = 2;
  size_t v13 = 3;
  size_t v14 = 4;
  size_t v15 = 5;
  size_t v16 = 6;
  size_t v17 = 7;
  size_t v18 = 8;
  int16_t v19 = 0;
  int32_t v20 = 0;
  size_t v21 = v10 * v9;
  size_t v22 = v10 + v21;
  int32_t* v23 = &v8[v22];
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v14);
  size_t v25 = v11 * v9;
  size_t v26 = v10 + v25;
  int32_t* v27 = &v8[v26];
  vint32m1_t v28 = __riscv_vle32_v_i32m1(v27, v14);
  vint32m1_t v29;
  v29 = v24;
  vint32m1_t v30;
  v30 = v28;
  for (size_t i31 = v10; i31 < v1; i31 += v11) {
    size_t v32 = i31 * v4;
    size_t v33 = 4;
    int8_t* v34 = &v3[v32];
    vint8mf4_t v35 = __riscv_vle8_v_i8mf4(v34, v33);
    vint16mf2_t v36 = __riscv_vsext_vf2_i16mf2(v35, v33);
    size_t v37 = i31 * v6;
    size_t v38 = v37 + v11;
    int8_t v39 = v5[v37];
    int16_t v40 = (int16_t) v39;
    vint32m1_t v41 = v29;
    vint32m1_t v42 = __riscv_vwmacc_vx_i32m1(v41, v40, v36, v33);
    int8_t v43 = v5[v38];
    int16_t v44 = (int16_t) v43;
    vint32m1_t v45 = v30;
    vint32m1_t v46 = __riscv_vwmacc_vx_i32m1(v45, v44, v36, v33);
    v29 = v42;
    v30 = v46;
  }
  vint32m1_t v47 = v29;
  vint32m1_t v48 = v30;
  int32_t* v49 = &v8[v10];
  __riscv_vse32_v_i32m1(v49, v47, v14);
  int32_t* v50 = &v8[v9];
  __riscv_vse32_v_i32m1(v50, v48, v14);
  return;
}



