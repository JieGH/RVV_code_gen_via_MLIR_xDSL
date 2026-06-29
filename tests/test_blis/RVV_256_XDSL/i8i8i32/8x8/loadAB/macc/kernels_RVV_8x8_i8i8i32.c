// ============================================================
// xDSL Generated Merged Kernels
// ------------------------------------------------------------
// Precision : i8i8i32
// Types     : A=int8_t, B=int8_t, C=int32_t
// Generation: 2026-06-29 11:41:11
// ============================================================
#include "kernels_RVV_8x8_i8i8i32.h"
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
//   Generated   : 2026-06-29 11:41:10
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
//   Generated   : 2026-06-29 11:41:10
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
//   Generated   : 2026-06-29 11:41:10
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
//   Generated   : 2026-06-29 11:41:10
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
// Merged from xdsl_api_gemm_1x3_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 1
//   NR         : 3
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_1x3_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v11);
  vint32m1_t v24;
  v24 = v21;
  vint32m1_t v25;
  v25 = v22;
  vint32m1_t v26;
  v26 = v23;
  for (size_t i27 = v10; i27 < v1; i27 += v11) {
    size_t v28 = i27 * v4;
    size_t v29 = 1;
    int8_t* v30 = &v3[v28];
    vint8mf4_t v31 = __riscv_vle8_v_i8mf4(v30, v29);
    vint16mf2_t v32 = __riscv_vsext_vf2_i16mf2(v31, v29);
    size_t v33 = i27 * v6;
    size_t v34 = v33 + v11;
    size_t v35 = v33 + v12;
    int8_t v36 = v5[v33];
    int16_t v37 = (int16_t) v36;
    vint32m1_t v38 = v24;
    vint32m1_t v39 = __riscv_vwmacc_vx_i32m1(v38, v37, v32, v29);
    int8_t v40 = v5[v34];
    int16_t v41 = (int16_t) v40;
    vint32m1_t v42 = v25;
    vint32m1_t v43 = __riscv_vwmacc_vx_i32m1(v42, v41, v32, v29);
    int8_t v44 = v5[v35];
    int16_t v45 = (int16_t) v44;
    vint32m1_t v46 = v26;
    vint32m1_t v47 = __riscv_vwmacc_vx_i32m1(v46, v45, v32, v29);
    v24 = v39;
    v25 = v43;
    v26 = v47;
  }
  vint32m1_t v48 = v24;
  vint32m1_t v49 = v25;
  vint32m1_t v50 = v26;
  int32_t* v51 = &v8[v10];
  __riscv_vse32_v_i32m1(v51, v48, v11);
  int32_t* v52 = &v8[v9];
  __riscv_vse32_v_i32m1(v52, v49, v11);
  size_t v53 = v12 * v9;
  int32_t* v54 = &v8[v53];
  __riscv_vse32_v_i32m1(v54, v50, v11);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_1x3_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 1
//   NR         : 3
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_1x3_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v11);
  vint32m1_t v33;
  v33 = v24;
  vint32m1_t v34;
  v34 = v28;
  vint32m1_t v35;
  v35 = v32;
  for (size_t i36 = v10; i36 < v1; i36 += v11) {
    size_t v37 = i36 * v4;
    size_t v38 = 1;
    int8_t* v39 = &v3[v37];
    vint8mf4_t v40 = __riscv_vle8_v_i8mf4(v39, v38);
    vint16mf2_t v41 = __riscv_vsext_vf2_i16mf2(v40, v38);
    size_t v42 = i36 * v6;
    size_t v43 = v42 + v11;
    size_t v44 = v42 + v12;
    int8_t v45 = v5[v42];
    int16_t v46 = (int16_t) v45;
    vint32m1_t v47 = v33;
    vint32m1_t v48 = __riscv_vwmacc_vx_i32m1(v47, v46, v41, v38);
    int8_t v49 = v5[v43];
    int16_t v50 = (int16_t) v49;
    vint32m1_t v51 = v34;
    vint32m1_t v52 = __riscv_vwmacc_vx_i32m1(v51, v50, v41, v38);
    int8_t v53 = v5[v44];
    int16_t v54 = (int16_t) v53;
    vint32m1_t v55 = v35;
    vint32m1_t v56 = __riscv_vwmacc_vx_i32m1(v55, v54, v41, v38);
    v33 = v48;
    v34 = v52;
    v35 = v56;
  }
  vint32m1_t v57 = v33;
  vint32m1_t v58 = v34;
  vint32m1_t v59 = v35;
  int32_t* v60 = &v8[v10];
  __riscv_vse32_v_i32m1(v60, v57, v11);
  int32_t* v61 = &v8[v9];
  __riscv_vse32_v_i32m1(v61, v58, v11);
  size_t v62 = v12 * v9;
  int32_t* v63 = &v8[v62];
  __riscv_vse32_v_i32m1(v63, v59, v11);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_1x4_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 1
//   NR         : 4
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_1x4_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v11);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v11);
  vint32m1_t v25;
  v25 = v21;
  vint32m1_t v26;
  v26 = v22;
  vint32m1_t v27;
  v27 = v23;
  vint32m1_t v28;
  v28 = v24;
  for (size_t i29 = v10; i29 < v1; i29 += v11) {
    size_t v30 = i29 * v4;
    size_t v31 = 1;
    int8_t* v32 = &v3[v30];
    vint8mf4_t v33 = __riscv_vle8_v_i8mf4(v32, v31);
    vint16mf2_t v34 = __riscv_vsext_vf2_i16mf2(v33, v31);
    size_t v35 = i29 * v6;
    size_t v36 = v35 + v11;
    size_t v37 = v35 + v12;
    size_t v38 = v35 + v13;
    int8_t v39 = v5[v35];
    int16_t v40 = (int16_t) v39;
    vint32m1_t v41 = v25;
    vint32m1_t v42 = __riscv_vwmacc_vx_i32m1(v41, v40, v34, v31);
    int8_t v43 = v5[v36];
    int16_t v44 = (int16_t) v43;
    vint32m1_t v45 = v26;
    vint32m1_t v46 = __riscv_vwmacc_vx_i32m1(v45, v44, v34, v31);
    int8_t v47 = v5[v37];
    int16_t v48 = (int16_t) v47;
    vint32m1_t v49 = v27;
    vint32m1_t v50 = __riscv_vwmacc_vx_i32m1(v49, v48, v34, v31);
    int8_t v51 = v5[v38];
    int16_t v52 = (int16_t) v51;
    vint32m1_t v53 = v28;
    vint32m1_t v54 = __riscv_vwmacc_vx_i32m1(v53, v52, v34, v31);
    v25 = v42;
    v26 = v46;
    v27 = v50;
    v28 = v54;
  }
  vint32m1_t v55 = v25;
  vint32m1_t v56 = v26;
  vint32m1_t v57 = v27;
  vint32m1_t v58 = v28;
  int32_t* v59 = &v8[v10];
  __riscv_vse32_v_i32m1(v59, v55, v11);
  int32_t* v60 = &v8[v9];
  __riscv_vse32_v_i32m1(v60, v56, v11);
  size_t v61 = v12 * v9;
  int32_t* v62 = &v8[v61];
  __riscv_vse32_v_i32m1(v62, v57, v11);
  size_t v63 = v13 * v9;
  int32_t* v64 = &v8[v63];
  __riscv_vse32_v_i32m1(v64, v58, v11);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_1x4_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 1
//   NR         : 4
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_1x4_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v11);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v11);
  vint32m1_t v37;
  v37 = v24;
  vint32m1_t v38;
  v38 = v28;
  vint32m1_t v39;
  v39 = v32;
  vint32m1_t v40;
  v40 = v36;
  for (size_t i41 = v10; i41 < v1; i41 += v11) {
    size_t v42 = i41 * v4;
    size_t v43 = 1;
    int8_t* v44 = &v3[v42];
    vint8mf4_t v45 = __riscv_vle8_v_i8mf4(v44, v43);
    vint16mf2_t v46 = __riscv_vsext_vf2_i16mf2(v45, v43);
    size_t v47 = i41 * v6;
    size_t v48 = v47 + v11;
    size_t v49 = v47 + v12;
    size_t v50 = v47 + v13;
    int8_t v51 = v5[v47];
    int16_t v52 = (int16_t) v51;
    vint32m1_t v53 = v37;
    vint32m1_t v54 = __riscv_vwmacc_vx_i32m1(v53, v52, v46, v43);
    int8_t v55 = v5[v48];
    int16_t v56 = (int16_t) v55;
    vint32m1_t v57 = v38;
    vint32m1_t v58 = __riscv_vwmacc_vx_i32m1(v57, v56, v46, v43);
    int8_t v59 = v5[v49];
    int16_t v60 = (int16_t) v59;
    vint32m1_t v61 = v39;
    vint32m1_t v62 = __riscv_vwmacc_vx_i32m1(v61, v60, v46, v43);
    int8_t v63 = v5[v50];
    int16_t v64 = (int16_t) v63;
    vint32m1_t v65 = v40;
    vint32m1_t v66 = __riscv_vwmacc_vx_i32m1(v65, v64, v46, v43);
    v37 = v54;
    v38 = v58;
    v39 = v62;
    v40 = v66;
  }
  vint32m1_t v67 = v37;
  vint32m1_t v68 = v38;
  vint32m1_t v69 = v39;
  vint32m1_t v70 = v40;
  int32_t* v71 = &v8[v10];
  __riscv_vse32_v_i32m1(v71, v67, v11);
  int32_t* v72 = &v8[v9];
  __riscv_vse32_v_i32m1(v72, v68, v11);
  size_t v73 = v12 * v9;
  int32_t* v74 = &v8[v73];
  __riscv_vse32_v_i32m1(v74, v69, v11);
  size_t v75 = v13 * v9;
  int32_t* v76 = &v8[v75];
  __riscv_vse32_v_i32m1(v76, v70, v11);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_1x5_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 1
//   NR         : 5
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_1x5_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v11);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v11);
  vint32m1_t v25 = __riscv_vmv_v_x_i32m1(v20, v11);
  vint32m1_t v26;
  v26 = v21;
  vint32m1_t v27;
  v27 = v22;
  vint32m1_t v28;
  v28 = v23;
  vint32m1_t v29;
  v29 = v24;
  vint32m1_t v30;
  v30 = v25;
  for (size_t i31 = v10; i31 < v1; i31 += v11) {
    size_t v32 = i31 * v4;
    size_t v33 = 1;
    int8_t* v34 = &v3[v32];
    vint8mf4_t v35 = __riscv_vle8_v_i8mf4(v34, v33);
    vint16mf2_t v36 = __riscv_vsext_vf2_i16mf2(v35, v33);
    size_t v37 = i31 * v6;
    size_t v38 = v37 + v11;
    size_t v39 = v37 + v12;
    size_t v40 = v37 + v13;
    size_t v41 = v37 + v14;
    int8_t v42 = v5[v37];
    int16_t v43 = (int16_t) v42;
    vint32m1_t v44 = v26;
    vint32m1_t v45 = __riscv_vwmacc_vx_i32m1(v44, v43, v36, v33);
    int8_t v46 = v5[v38];
    int16_t v47 = (int16_t) v46;
    vint32m1_t v48 = v27;
    vint32m1_t v49 = __riscv_vwmacc_vx_i32m1(v48, v47, v36, v33);
    int8_t v50 = v5[v39];
    int16_t v51 = (int16_t) v50;
    vint32m1_t v52 = v28;
    vint32m1_t v53 = __riscv_vwmacc_vx_i32m1(v52, v51, v36, v33);
    int8_t v54 = v5[v40];
    int16_t v55 = (int16_t) v54;
    vint32m1_t v56 = v29;
    vint32m1_t v57 = __riscv_vwmacc_vx_i32m1(v56, v55, v36, v33);
    int8_t v58 = v5[v41];
    int16_t v59 = (int16_t) v58;
    vint32m1_t v60 = v30;
    vint32m1_t v61 = __riscv_vwmacc_vx_i32m1(v60, v59, v36, v33);
    v26 = v45;
    v27 = v49;
    v28 = v53;
    v29 = v57;
    v30 = v61;
  }
  vint32m1_t v62 = v26;
  vint32m1_t v63 = v27;
  vint32m1_t v64 = v28;
  vint32m1_t v65 = v29;
  vint32m1_t v66 = v30;
  int32_t* v67 = &v8[v10];
  __riscv_vse32_v_i32m1(v67, v62, v11);
  int32_t* v68 = &v8[v9];
  __riscv_vse32_v_i32m1(v68, v63, v11);
  size_t v69 = v12 * v9;
  int32_t* v70 = &v8[v69];
  __riscv_vse32_v_i32m1(v70, v64, v11);
  size_t v71 = v13 * v9;
  int32_t* v72 = &v8[v71];
  __riscv_vse32_v_i32m1(v72, v65, v11);
  size_t v73 = v14 * v9;
  int32_t* v74 = &v8[v73];
  __riscv_vse32_v_i32m1(v74, v66, v11);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_1x5_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 1
//   NR         : 5
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_1x5_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v11);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v11);
  size_t v37 = v14 * v9;
  size_t v38 = v10 + v37;
  int32_t* v39 = &v8[v38];
  vint32m1_t v40 = __riscv_vle32_v_i32m1(v39, v11);
  vint32m1_t v41;
  v41 = v24;
  vint32m1_t v42;
  v42 = v28;
  vint32m1_t v43;
  v43 = v32;
  vint32m1_t v44;
  v44 = v36;
  vint32m1_t v45;
  v45 = v40;
  for (size_t i46 = v10; i46 < v1; i46 += v11) {
    size_t v47 = i46 * v4;
    size_t v48 = 1;
    int8_t* v49 = &v3[v47];
    vint8mf4_t v50 = __riscv_vle8_v_i8mf4(v49, v48);
    vint16mf2_t v51 = __riscv_vsext_vf2_i16mf2(v50, v48);
    size_t v52 = i46 * v6;
    size_t v53 = v52 + v11;
    size_t v54 = v52 + v12;
    size_t v55 = v52 + v13;
    size_t v56 = v52 + v14;
    int8_t v57 = v5[v52];
    int16_t v58 = (int16_t) v57;
    vint32m1_t v59 = v41;
    vint32m1_t v60 = __riscv_vwmacc_vx_i32m1(v59, v58, v51, v48);
    int8_t v61 = v5[v53];
    int16_t v62 = (int16_t) v61;
    vint32m1_t v63 = v42;
    vint32m1_t v64 = __riscv_vwmacc_vx_i32m1(v63, v62, v51, v48);
    int8_t v65 = v5[v54];
    int16_t v66 = (int16_t) v65;
    vint32m1_t v67 = v43;
    vint32m1_t v68 = __riscv_vwmacc_vx_i32m1(v67, v66, v51, v48);
    int8_t v69 = v5[v55];
    int16_t v70 = (int16_t) v69;
    vint32m1_t v71 = v44;
    vint32m1_t v72 = __riscv_vwmacc_vx_i32m1(v71, v70, v51, v48);
    int8_t v73 = v5[v56];
    int16_t v74 = (int16_t) v73;
    vint32m1_t v75 = v45;
    vint32m1_t v76 = __riscv_vwmacc_vx_i32m1(v75, v74, v51, v48);
    v41 = v60;
    v42 = v64;
    v43 = v68;
    v44 = v72;
    v45 = v76;
  }
  vint32m1_t v77 = v41;
  vint32m1_t v78 = v42;
  vint32m1_t v79 = v43;
  vint32m1_t v80 = v44;
  vint32m1_t v81 = v45;
  int32_t* v82 = &v8[v10];
  __riscv_vse32_v_i32m1(v82, v77, v11);
  int32_t* v83 = &v8[v9];
  __riscv_vse32_v_i32m1(v83, v78, v11);
  size_t v84 = v12 * v9;
  int32_t* v85 = &v8[v84];
  __riscv_vse32_v_i32m1(v85, v79, v11);
  size_t v86 = v13 * v9;
  int32_t* v87 = &v8[v86];
  __riscv_vse32_v_i32m1(v87, v80, v11);
  size_t v88 = v14 * v9;
  int32_t* v89 = &v8[v88];
  __riscv_vse32_v_i32m1(v89, v81, v11);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_1x6_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 1
//   NR         : 6
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_1x6_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v11);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v11);
  vint32m1_t v25 = __riscv_vmv_v_x_i32m1(v20, v11);
  vint32m1_t v26 = __riscv_vmv_v_x_i32m1(v20, v11);
  vint32m1_t v27;
  v27 = v21;
  vint32m1_t v28;
  v28 = v22;
  vint32m1_t v29;
  v29 = v23;
  vint32m1_t v30;
  v30 = v24;
  vint32m1_t v31;
  v31 = v25;
  vint32m1_t v32;
  v32 = v26;
  for (size_t i33 = v10; i33 < v1; i33 += v11) {
    size_t v34 = i33 * v4;
    size_t v35 = 1;
    int8_t* v36 = &v3[v34];
    vint8mf4_t v37 = __riscv_vle8_v_i8mf4(v36, v35);
    vint16mf2_t v38 = __riscv_vsext_vf2_i16mf2(v37, v35);
    size_t v39 = i33 * v6;
    size_t v40 = v39 + v11;
    size_t v41 = v39 + v12;
    size_t v42 = v39 + v13;
    size_t v43 = v39 + v14;
    size_t v44 = v39 + v15;
    int8_t v45 = v5[v39];
    int16_t v46 = (int16_t) v45;
    vint32m1_t v47 = v27;
    vint32m1_t v48 = __riscv_vwmacc_vx_i32m1(v47, v46, v38, v35);
    int8_t v49 = v5[v40];
    int16_t v50 = (int16_t) v49;
    vint32m1_t v51 = v28;
    vint32m1_t v52 = __riscv_vwmacc_vx_i32m1(v51, v50, v38, v35);
    int8_t v53 = v5[v41];
    int16_t v54 = (int16_t) v53;
    vint32m1_t v55 = v29;
    vint32m1_t v56 = __riscv_vwmacc_vx_i32m1(v55, v54, v38, v35);
    int8_t v57 = v5[v42];
    int16_t v58 = (int16_t) v57;
    vint32m1_t v59 = v30;
    vint32m1_t v60 = __riscv_vwmacc_vx_i32m1(v59, v58, v38, v35);
    int8_t v61 = v5[v43];
    int16_t v62 = (int16_t) v61;
    vint32m1_t v63 = v31;
    vint32m1_t v64 = __riscv_vwmacc_vx_i32m1(v63, v62, v38, v35);
    int8_t v65 = v5[v44];
    int16_t v66 = (int16_t) v65;
    vint32m1_t v67 = v32;
    vint32m1_t v68 = __riscv_vwmacc_vx_i32m1(v67, v66, v38, v35);
    v27 = v48;
    v28 = v52;
    v29 = v56;
    v30 = v60;
    v31 = v64;
    v32 = v68;
  }
  vint32m1_t v69 = v27;
  vint32m1_t v70 = v28;
  vint32m1_t v71 = v29;
  vint32m1_t v72 = v30;
  vint32m1_t v73 = v31;
  vint32m1_t v74 = v32;
  int32_t* v75 = &v8[v10];
  __riscv_vse32_v_i32m1(v75, v69, v11);
  int32_t* v76 = &v8[v9];
  __riscv_vse32_v_i32m1(v76, v70, v11);
  size_t v77 = v12 * v9;
  int32_t* v78 = &v8[v77];
  __riscv_vse32_v_i32m1(v78, v71, v11);
  size_t v79 = v13 * v9;
  int32_t* v80 = &v8[v79];
  __riscv_vse32_v_i32m1(v80, v72, v11);
  size_t v81 = v14 * v9;
  int32_t* v82 = &v8[v81];
  __riscv_vse32_v_i32m1(v82, v73, v11);
  size_t v83 = v15 * v9;
  int32_t* v84 = &v8[v83];
  __riscv_vse32_v_i32m1(v84, v74, v11);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_1x6_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 1
//   NR         : 6
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_1x6_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v11);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v11);
  size_t v37 = v14 * v9;
  size_t v38 = v10 + v37;
  int32_t* v39 = &v8[v38];
  vint32m1_t v40 = __riscv_vle32_v_i32m1(v39, v11);
  size_t v41 = v15 * v9;
  size_t v42 = v10 + v41;
  int32_t* v43 = &v8[v42];
  vint32m1_t v44 = __riscv_vle32_v_i32m1(v43, v11);
  vint32m1_t v45;
  v45 = v24;
  vint32m1_t v46;
  v46 = v28;
  vint32m1_t v47;
  v47 = v32;
  vint32m1_t v48;
  v48 = v36;
  vint32m1_t v49;
  v49 = v40;
  vint32m1_t v50;
  v50 = v44;
  for (size_t i51 = v10; i51 < v1; i51 += v11) {
    size_t v52 = i51 * v4;
    size_t v53 = 1;
    int8_t* v54 = &v3[v52];
    vint8mf4_t v55 = __riscv_vle8_v_i8mf4(v54, v53);
    vint16mf2_t v56 = __riscv_vsext_vf2_i16mf2(v55, v53);
    size_t v57 = i51 * v6;
    size_t v58 = v57 + v11;
    size_t v59 = v57 + v12;
    size_t v60 = v57 + v13;
    size_t v61 = v57 + v14;
    size_t v62 = v57 + v15;
    int8_t v63 = v5[v57];
    int16_t v64 = (int16_t) v63;
    vint32m1_t v65 = v45;
    vint32m1_t v66 = __riscv_vwmacc_vx_i32m1(v65, v64, v56, v53);
    int8_t v67 = v5[v58];
    int16_t v68 = (int16_t) v67;
    vint32m1_t v69 = v46;
    vint32m1_t v70 = __riscv_vwmacc_vx_i32m1(v69, v68, v56, v53);
    int8_t v71 = v5[v59];
    int16_t v72 = (int16_t) v71;
    vint32m1_t v73 = v47;
    vint32m1_t v74 = __riscv_vwmacc_vx_i32m1(v73, v72, v56, v53);
    int8_t v75 = v5[v60];
    int16_t v76 = (int16_t) v75;
    vint32m1_t v77 = v48;
    vint32m1_t v78 = __riscv_vwmacc_vx_i32m1(v77, v76, v56, v53);
    int8_t v79 = v5[v61];
    int16_t v80 = (int16_t) v79;
    vint32m1_t v81 = v49;
    vint32m1_t v82 = __riscv_vwmacc_vx_i32m1(v81, v80, v56, v53);
    int8_t v83 = v5[v62];
    int16_t v84 = (int16_t) v83;
    vint32m1_t v85 = v50;
    vint32m1_t v86 = __riscv_vwmacc_vx_i32m1(v85, v84, v56, v53);
    v45 = v66;
    v46 = v70;
    v47 = v74;
    v48 = v78;
    v49 = v82;
    v50 = v86;
  }
  vint32m1_t v87 = v45;
  vint32m1_t v88 = v46;
  vint32m1_t v89 = v47;
  vint32m1_t v90 = v48;
  vint32m1_t v91 = v49;
  vint32m1_t v92 = v50;
  int32_t* v93 = &v8[v10];
  __riscv_vse32_v_i32m1(v93, v87, v11);
  int32_t* v94 = &v8[v9];
  __riscv_vse32_v_i32m1(v94, v88, v11);
  size_t v95 = v12 * v9;
  int32_t* v96 = &v8[v95];
  __riscv_vse32_v_i32m1(v96, v89, v11);
  size_t v97 = v13 * v9;
  int32_t* v98 = &v8[v97];
  __riscv_vse32_v_i32m1(v98, v90, v11);
  size_t v99 = v14 * v9;
  int32_t* v100 = &v8[v99];
  __riscv_vse32_v_i32m1(v100, v91, v11);
  size_t v101 = v15 * v9;
  int32_t* v102 = &v8[v101];
  __riscv_vse32_v_i32m1(v102, v92, v11);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_1x7_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 1
//   NR         : 7
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_1x7_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v11);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v11);
  vint32m1_t v25 = __riscv_vmv_v_x_i32m1(v20, v11);
  vint32m1_t v26 = __riscv_vmv_v_x_i32m1(v20, v11);
  vint32m1_t v27 = __riscv_vmv_v_x_i32m1(v20, v11);
  vint32m1_t v28;
  v28 = v21;
  vint32m1_t v29;
  v29 = v22;
  vint32m1_t v30;
  v30 = v23;
  vint32m1_t v31;
  v31 = v24;
  vint32m1_t v32;
  v32 = v25;
  vint32m1_t v33;
  v33 = v26;
  vint32m1_t v34;
  v34 = v27;
  for (size_t i35 = v10; i35 < v1; i35 += v11) {
    size_t v36 = i35 * v4;
    size_t v37 = 1;
    int8_t* v38 = &v3[v36];
    vint8mf4_t v39 = __riscv_vle8_v_i8mf4(v38, v37);
    vint16mf2_t v40 = __riscv_vsext_vf2_i16mf2(v39, v37);
    size_t v41 = i35 * v6;
    size_t v42 = v41 + v11;
    size_t v43 = v41 + v12;
    size_t v44 = v41 + v13;
    size_t v45 = v41 + v14;
    size_t v46 = v41 + v15;
    size_t v47 = v41 + v16;
    int8_t v48 = v5[v41];
    int16_t v49 = (int16_t) v48;
    vint32m1_t v50 = v28;
    vint32m1_t v51 = __riscv_vwmacc_vx_i32m1(v50, v49, v40, v37);
    int8_t v52 = v5[v42];
    int16_t v53 = (int16_t) v52;
    vint32m1_t v54 = v29;
    vint32m1_t v55 = __riscv_vwmacc_vx_i32m1(v54, v53, v40, v37);
    int8_t v56 = v5[v43];
    int16_t v57 = (int16_t) v56;
    vint32m1_t v58 = v30;
    vint32m1_t v59 = __riscv_vwmacc_vx_i32m1(v58, v57, v40, v37);
    int8_t v60 = v5[v44];
    int16_t v61 = (int16_t) v60;
    vint32m1_t v62 = v31;
    vint32m1_t v63 = __riscv_vwmacc_vx_i32m1(v62, v61, v40, v37);
    int8_t v64 = v5[v45];
    int16_t v65 = (int16_t) v64;
    vint32m1_t v66 = v32;
    vint32m1_t v67 = __riscv_vwmacc_vx_i32m1(v66, v65, v40, v37);
    int8_t v68 = v5[v46];
    int16_t v69 = (int16_t) v68;
    vint32m1_t v70 = v33;
    vint32m1_t v71 = __riscv_vwmacc_vx_i32m1(v70, v69, v40, v37);
    int8_t v72 = v5[v47];
    int16_t v73 = (int16_t) v72;
    vint32m1_t v74 = v34;
    vint32m1_t v75 = __riscv_vwmacc_vx_i32m1(v74, v73, v40, v37);
    v28 = v51;
    v29 = v55;
    v30 = v59;
    v31 = v63;
    v32 = v67;
    v33 = v71;
    v34 = v75;
  }
  vint32m1_t v76 = v28;
  vint32m1_t v77 = v29;
  vint32m1_t v78 = v30;
  vint32m1_t v79 = v31;
  vint32m1_t v80 = v32;
  vint32m1_t v81 = v33;
  vint32m1_t v82 = v34;
  int32_t* v83 = &v8[v10];
  __riscv_vse32_v_i32m1(v83, v76, v11);
  int32_t* v84 = &v8[v9];
  __riscv_vse32_v_i32m1(v84, v77, v11);
  size_t v85 = v12 * v9;
  int32_t* v86 = &v8[v85];
  __riscv_vse32_v_i32m1(v86, v78, v11);
  size_t v87 = v13 * v9;
  int32_t* v88 = &v8[v87];
  __riscv_vse32_v_i32m1(v88, v79, v11);
  size_t v89 = v14 * v9;
  int32_t* v90 = &v8[v89];
  __riscv_vse32_v_i32m1(v90, v80, v11);
  size_t v91 = v15 * v9;
  int32_t* v92 = &v8[v91];
  __riscv_vse32_v_i32m1(v92, v81, v11);
  size_t v93 = v16 * v9;
  int32_t* v94 = &v8[v93];
  __riscv_vse32_v_i32m1(v94, v82, v11);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_1x7_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 1
//   NR         : 7
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_1x7_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v11);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v11);
  size_t v37 = v14 * v9;
  size_t v38 = v10 + v37;
  int32_t* v39 = &v8[v38];
  vint32m1_t v40 = __riscv_vle32_v_i32m1(v39, v11);
  size_t v41 = v15 * v9;
  size_t v42 = v10 + v41;
  int32_t* v43 = &v8[v42];
  vint32m1_t v44 = __riscv_vle32_v_i32m1(v43, v11);
  size_t v45 = v16 * v9;
  size_t v46 = v10 + v45;
  int32_t* v47 = &v8[v46];
  vint32m1_t v48 = __riscv_vle32_v_i32m1(v47, v11);
  vint32m1_t v49;
  v49 = v24;
  vint32m1_t v50;
  v50 = v28;
  vint32m1_t v51;
  v51 = v32;
  vint32m1_t v52;
  v52 = v36;
  vint32m1_t v53;
  v53 = v40;
  vint32m1_t v54;
  v54 = v44;
  vint32m1_t v55;
  v55 = v48;
  for (size_t i56 = v10; i56 < v1; i56 += v11) {
    size_t v57 = i56 * v4;
    size_t v58 = 1;
    int8_t* v59 = &v3[v57];
    vint8mf4_t v60 = __riscv_vle8_v_i8mf4(v59, v58);
    vint16mf2_t v61 = __riscv_vsext_vf2_i16mf2(v60, v58);
    size_t v62 = i56 * v6;
    size_t v63 = v62 + v11;
    size_t v64 = v62 + v12;
    size_t v65 = v62 + v13;
    size_t v66 = v62 + v14;
    size_t v67 = v62 + v15;
    size_t v68 = v62 + v16;
    int8_t v69 = v5[v62];
    int16_t v70 = (int16_t) v69;
    vint32m1_t v71 = v49;
    vint32m1_t v72 = __riscv_vwmacc_vx_i32m1(v71, v70, v61, v58);
    int8_t v73 = v5[v63];
    int16_t v74 = (int16_t) v73;
    vint32m1_t v75 = v50;
    vint32m1_t v76 = __riscv_vwmacc_vx_i32m1(v75, v74, v61, v58);
    int8_t v77 = v5[v64];
    int16_t v78 = (int16_t) v77;
    vint32m1_t v79 = v51;
    vint32m1_t v80 = __riscv_vwmacc_vx_i32m1(v79, v78, v61, v58);
    int8_t v81 = v5[v65];
    int16_t v82 = (int16_t) v81;
    vint32m1_t v83 = v52;
    vint32m1_t v84 = __riscv_vwmacc_vx_i32m1(v83, v82, v61, v58);
    int8_t v85 = v5[v66];
    int16_t v86 = (int16_t) v85;
    vint32m1_t v87 = v53;
    vint32m1_t v88 = __riscv_vwmacc_vx_i32m1(v87, v86, v61, v58);
    int8_t v89 = v5[v67];
    int16_t v90 = (int16_t) v89;
    vint32m1_t v91 = v54;
    vint32m1_t v92 = __riscv_vwmacc_vx_i32m1(v91, v90, v61, v58);
    int8_t v93 = v5[v68];
    int16_t v94 = (int16_t) v93;
    vint32m1_t v95 = v55;
    vint32m1_t v96 = __riscv_vwmacc_vx_i32m1(v95, v94, v61, v58);
    v49 = v72;
    v50 = v76;
    v51 = v80;
    v52 = v84;
    v53 = v88;
    v54 = v92;
    v55 = v96;
  }
  vint32m1_t v97 = v49;
  vint32m1_t v98 = v50;
  vint32m1_t v99 = v51;
  vint32m1_t v100 = v52;
  vint32m1_t v101 = v53;
  vint32m1_t v102 = v54;
  vint32m1_t v103 = v55;
  int32_t* v104 = &v8[v10];
  __riscv_vse32_v_i32m1(v104, v97, v11);
  int32_t* v105 = &v8[v9];
  __riscv_vse32_v_i32m1(v105, v98, v11);
  size_t v106 = v12 * v9;
  int32_t* v107 = &v8[v106];
  __riscv_vse32_v_i32m1(v107, v99, v11);
  size_t v108 = v13 * v9;
  int32_t* v109 = &v8[v108];
  __riscv_vse32_v_i32m1(v109, v100, v11);
  size_t v110 = v14 * v9;
  int32_t* v111 = &v8[v110];
  __riscv_vse32_v_i32m1(v111, v101, v11);
  size_t v112 = v15 * v9;
  int32_t* v113 = &v8[v112];
  __riscv_vse32_v_i32m1(v113, v102, v11);
  size_t v114 = v16 * v9;
  int32_t* v115 = &v8[v114];
  __riscv_vse32_v_i32m1(v115, v103, v11);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_1x8_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 1
//   NR         : 8
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_1x8_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v11);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v11);
  vint32m1_t v25 = __riscv_vmv_v_x_i32m1(v20, v11);
  vint32m1_t v26 = __riscv_vmv_v_x_i32m1(v20, v11);
  vint32m1_t v27 = __riscv_vmv_v_x_i32m1(v20, v11);
  vint32m1_t v28 = __riscv_vmv_v_x_i32m1(v20, v11);
  vint32m1_t v29;
  v29 = v21;
  vint32m1_t v30;
  v30 = v22;
  vint32m1_t v31;
  v31 = v23;
  vint32m1_t v32;
  v32 = v24;
  vint32m1_t v33;
  v33 = v25;
  vint32m1_t v34;
  v34 = v26;
  vint32m1_t v35;
  v35 = v27;
  vint32m1_t v36;
  v36 = v28;
  for (size_t i37 = v10; i37 < v1; i37 += v11) {
    size_t v38 = i37 * v4;
    size_t v39 = 1;
    int8_t* v40 = &v3[v38];
    vint8mf4_t v41 = __riscv_vle8_v_i8mf4(v40, v39);
    vint16mf2_t v42 = __riscv_vsext_vf2_i16mf2(v41, v39);
    size_t v43 = i37 * v6;
    size_t v44 = v43 + v11;
    size_t v45 = v43 + v12;
    size_t v46 = v43 + v13;
    size_t v47 = v43 + v14;
    size_t v48 = v43 + v15;
    size_t v49 = v43 + v16;
    size_t v50 = v43 + v17;
    int8_t v51 = v5[v43];
    int16_t v52 = (int16_t) v51;
    vint32m1_t v53 = v29;
    vint32m1_t v54 = __riscv_vwmacc_vx_i32m1(v53, v52, v42, v39);
    int8_t v55 = v5[v44];
    int16_t v56 = (int16_t) v55;
    vint32m1_t v57 = v30;
    vint32m1_t v58 = __riscv_vwmacc_vx_i32m1(v57, v56, v42, v39);
    int8_t v59 = v5[v45];
    int16_t v60 = (int16_t) v59;
    vint32m1_t v61 = v31;
    vint32m1_t v62 = __riscv_vwmacc_vx_i32m1(v61, v60, v42, v39);
    int8_t v63 = v5[v46];
    int16_t v64 = (int16_t) v63;
    vint32m1_t v65 = v32;
    vint32m1_t v66 = __riscv_vwmacc_vx_i32m1(v65, v64, v42, v39);
    int8_t v67 = v5[v47];
    int16_t v68 = (int16_t) v67;
    vint32m1_t v69 = v33;
    vint32m1_t v70 = __riscv_vwmacc_vx_i32m1(v69, v68, v42, v39);
    int8_t v71 = v5[v48];
    int16_t v72 = (int16_t) v71;
    vint32m1_t v73 = v34;
    vint32m1_t v74 = __riscv_vwmacc_vx_i32m1(v73, v72, v42, v39);
    int8_t v75 = v5[v49];
    int16_t v76 = (int16_t) v75;
    vint32m1_t v77 = v35;
    vint32m1_t v78 = __riscv_vwmacc_vx_i32m1(v77, v76, v42, v39);
    int8_t v79 = v5[v50];
    int16_t v80 = (int16_t) v79;
    vint32m1_t v81 = v36;
    vint32m1_t v82 = __riscv_vwmacc_vx_i32m1(v81, v80, v42, v39);
    v29 = v54;
    v30 = v58;
    v31 = v62;
    v32 = v66;
    v33 = v70;
    v34 = v74;
    v35 = v78;
    v36 = v82;
  }
  vint32m1_t v83 = v29;
  vint32m1_t v84 = v30;
  vint32m1_t v85 = v31;
  vint32m1_t v86 = v32;
  vint32m1_t v87 = v33;
  vint32m1_t v88 = v34;
  vint32m1_t v89 = v35;
  vint32m1_t v90 = v36;
  int32_t* v91 = &v8[v10];
  __riscv_vse32_v_i32m1(v91, v83, v11);
  int32_t* v92 = &v8[v9];
  __riscv_vse32_v_i32m1(v92, v84, v11);
  size_t v93 = v12 * v9;
  int32_t* v94 = &v8[v93];
  __riscv_vse32_v_i32m1(v94, v85, v11);
  size_t v95 = v13 * v9;
  int32_t* v96 = &v8[v95];
  __riscv_vse32_v_i32m1(v96, v86, v11);
  size_t v97 = v14 * v9;
  int32_t* v98 = &v8[v97];
  __riscv_vse32_v_i32m1(v98, v87, v11);
  size_t v99 = v15 * v9;
  int32_t* v100 = &v8[v99];
  __riscv_vse32_v_i32m1(v100, v88, v11);
  size_t v101 = v16 * v9;
  int32_t* v102 = &v8[v101];
  __riscv_vse32_v_i32m1(v102, v89, v11);
  size_t v103 = v17 * v9;
  int32_t* v104 = &v8[v103];
  __riscv_vse32_v_i32m1(v104, v90, v11);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_1x8_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 1
//   NR         : 8
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_1x8_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v11);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v11);
  size_t v37 = v14 * v9;
  size_t v38 = v10 + v37;
  int32_t* v39 = &v8[v38];
  vint32m1_t v40 = __riscv_vle32_v_i32m1(v39, v11);
  size_t v41 = v15 * v9;
  size_t v42 = v10 + v41;
  int32_t* v43 = &v8[v42];
  vint32m1_t v44 = __riscv_vle32_v_i32m1(v43, v11);
  size_t v45 = v16 * v9;
  size_t v46 = v10 + v45;
  int32_t* v47 = &v8[v46];
  vint32m1_t v48 = __riscv_vle32_v_i32m1(v47, v11);
  size_t v49 = v17 * v9;
  size_t v50 = v10 + v49;
  int32_t* v51 = &v8[v50];
  vint32m1_t v52 = __riscv_vle32_v_i32m1(v51, v11);
  vint32m1_t v53;
  v53 = v24;
  vint32m1_t v54;
  v54 = v28;
  vint32m1_t v55;
  v55 = v32;
  vint32m1_t v56;
  v56 = v36;
  vint32m1_t v57;
  v57 = v40;
  vint32m1_t v58;
  v58 = v44;
  vint32m1_t v59;
  v59 = v48;
  vint32m1_t v60;
  v60 = v52;
  for (size_t i61 = v10; i61 < v1; i61 += v11) {
    size_t v62 = i61 * v4;
    size_t v63 = 1;
    int8_t* v64 = &v3[v62];
    vint8mf4_t v65 = __riscv_vle8_v_i8mf4(v64, v63);
    vint16mf2_t v66 = __riscv_vsext_vf2_i16mf2(v65, v63);
    size_t v67 = i61 * v6;
    size_t v68 = v67 + v11;
    size_t v69 = v67 + v12;
    size_t v70 = v67 + v13;
    size_t v71 = v67 + v14;
    size_t v72 = v67 + v15;
    size_t v73 = v67 + v16;
    size_t v74 = v67 + v17;
    int8_t v75 = v5[v67];
    int16_t v76 = (int16_t) v75;
    vint32m1_t v77 = v53;
    vint32m1_t v78 = __riscv_vwmacc_vx_i32m1(v77, v76, v66, v63);
    int8_t v79 = v5[v68];
    int16_t v80 = (int16_t) v79;
    vint32m1_t v81 = v54;
    vint32m1_t v82 = __riscv_vwmacc_vx_i32m1(v81, v80, v66, v63);
    int8_t v83 = v5[v69];
    int16_t v84 = (int16_t) v83;
    vint32m1_t v85 = v55;
    vint32m1_t v86 = __riscv_vwmacc_vx_i32m1(v85, v84, v66, v63);
    int8_t v87 = v5[v70];
    int16_t v88 = (int16_t) v87;
    vint32m1_t v89 = v56;
    vint32m1_t v90 = __riscv_vwmacc_vx_i32m1(v89, v88, v66, v63);
    int8_t v91 = v5[v71];
    int16_t v92 = (int16_t) v91;
    vint32m1_t v93 = v57;
    vint32m1_t v94 = __riscv_vwmacc_vx_i32m1(v93, v92, v66, v63);
    int8_t v95 = v5[v72];
    int16_t v96 = (int16_t) v95;
    vint32m1_t v97 = v58;
    vint32m1_t v98 = __riscv_vwmacc_vx_i32m1(v97, v96, v66, v63);
    int8_t v99 = v5[v73];
    int16_t v100 = (int16_t) v99;
    vint32m1_t v101 = v59;
    vint32m1_t v102 = __riscv_vwmacc_vx_i32m1(v101, v100, v66, v63);
    int8_t v103 = v5[v74];
    int16_t v104 = (int16_t) v103;
    vint32m1_t v105 = v60;
    vint32m1_t v106 = __riscv_vwmacc_vx_i32m1(v105, v104, v66, v63);
    v53 = v78;
    v54 = v82;
    v55 = v86;
    v56 = v90;
    v57 = v94;
    v58 = v98;
    v59 = v102;
    v60 = v106;
  }
  vint32m1_t v107 = v53;
  vint32m1_t v108 = v54;
  vint32m1_t v109 = v55;
  vint32m1_t v110 = v56;
  vint32m1_t v111 = v57;
  vint32m1_t v112 = v58;
  vint32m1_t v113 = v59;
  vint32m1_t v114 = v60;
  int32_t* v115 = &v8[v10];
  __riscv_vse32_v_i32m1(v115, v107, v11);
  int32_t* v116 = &v8[v9];
  __riscv_vse32_v_i32m1(v116, v108, v11);
  size_t v117 = v12 * v9;
  int32_t* v118 = &v8[v117];
  __riscv_vse32_v_i32m1(v118, v109, v11);
  size_t v119 = v13 * v9;
  int32_t* v120 = &v8[v119];
  __riscv_vse32_v_i32m1(v120, v110, v11);
  size_t v121 = v14 * v9;
  int32_t* v122 = &v8[v121];
  __riscv_vse32_v_i32m1(v122, v111, v11);
  size_t v123 = v15 * v9;
  int32_t* v124 = &v8[v123];
  __riscv_vse32_v_i32m1(v124, v112, v11);
  size_t v125 = v16 * v9;
  int32_t* v126 = &v8[v125];
  __riscv_vse32_v_i32m1(v126, v113, v11);
  size_t v127 = v17 * v9;
  int32_t* v128 = &v8[v127];
  __riscv_vse32_v_i32m1(v128, v114, v11);
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
//   Generated   : 2026-06-29 11:41:10
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
//   Generated   : 2026-06-29 11:41:10
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
//   Generated   : 2026-06-29 11:41:10
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
//   Generated   : 2026-06-29 11:41:10
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
// Merged from xdsl_api_gemm_2x3_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 2
//   NR         : 3
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_2x3_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v12);
  vint32m1_t v24;
  v24 = v21;
  vint32m1_t v25;
  v25 = v22;
  vint32m1_t v26;
  v26 = v23;
  for (size_t i27 = v10; i27 < v1; i27 += v11) {
    size_t v28 = i27 * v4;
    size_t v29 = 2;
    int8_t* v30 = &v3[v28];
    vint8mf4_t v31 = __riscv_vle8_v_i8mf4(v30, v29);
    vint16mf2_t v32 = __riscv_vsext_vf2_i16mf2(v31, v29);
    size_t v33 = i27 * v6;
    size_t v34 = v33 + v11;
    size_t v35 = v33 + v12;
    int8_t v36 = v5[v33];
    int16_t v37 = (int16_t) v36;
    vint32m1_t v38 = v24;
    vint32m1_t v39 = __riscv_vwmacc_vx_i32m1(v38, v37, v32, v29);
    int8_t v40 = v5[v34];
    int16_t v41 = (int16_t) v40;
    vint32m1_t v42 = v25;
    vint32m1_t v43 = __riscv_vwmacc_vx_i32m1(v42, v41, v32, v29);
    int8_t v44 = v5[v35];
    int16_t v45 = (int16_t) v44;
    vint32m1_t v46 = v26;
    vint32m1_t v47 = __riscv_vwmacc_vx_i32m1(v46, v45, v32, v29);
    v24 = v39;
    v25 = v43;
    v26 = v47;
  }
  vint32m1_t v48 = v24;
  vint32m1_t v49 = v25;
  vint32m1_t v50 = v26;
  int32_t* v51 = &v8[v10];
  __riscv_vse32_v_i32m1(v51, v48, v12);
  int32_t* v52 = &v8[v9];
  __riscv_vse32_v_i32m1(v52, v49, v12);
  size_t v53 = v12 * v9;
  int32_t* v54 = &v8[v53];
  __riscv_vse32_v_i32m1(v54, v50, v12);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_2x3_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 2
//   NR         : 3
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_2x3_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v12);
  vint32m1_t v33;
  v33 = v24;
  vint32m1_t v34;
  v34 = v28;
  vint32m1_t v35;
  v35 = v32;
  for (size_t i36 = v10; i36 < v1; i36 += v11) {
    size_t v37 = i36 * v4;
    size_t v38 = 2;
    int8_t* v39 = &v3[v37];
    vint8mf4_t v40 = __riscv_vle8_v_i8mf4(v39, v38);
    vint16mf2_t v41 = __riscv_vsext_vf2_i16mf2(v40, v38);
    size_t v42 = i36 * v6;
    size_t v43 = v42 + v11;
    size_t v44 = v42 + v12;
    int8_t v45 = v5[v42];
    int16_t v46 = (int16_t) v45;
    vint32m1_t v47 = v33;
    vint32m1_t v48 = __riscv_vwmacc_vx_i32m1(v47, v46, v41, v38);
    int8_t v49 = v5[v43];
    int16_t v50 = (int16_t) v49;
    vint32m1_t v51 = v34;
    vint32m1_t v52 = __riscv_vwmacc_vx_i32m1(v51, v50, v41, v38);
    int8_t v53 = v5[v44];
    int16_t v54 = (int16_t) v53;
    vint32m1_t v55 = v35;
    vint32m1_t v56 = __riscv_vwmacc_vx_i32m1(v55, v54, v41, v38);
    v33 = v48;
    v34 = v52;
    v35 = v56;
  }
  vint32m1_t v57 = v33;
  vint32m1_t v58 = v34;
  vint32m1_t v59 = v35;
  int32_t* v60 = &v8[v10];
  __riscv_vse32_v_i32m1(v60, v57, v12);
  int32_t* v61 = &v8[v9];
  __riscv_vse32_v_i32m1(v61, v58, v12);
  size_t v62 = v12 * v9;
  int32_t* v63 = &v8[v62];
  __riscv_vse32_v_i32m1(v63, v59, v12);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_2x4_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 2
//   NR         : 4
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_2x4_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v12);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v12);
  vint32m1_t v25;
  v25 = v21;
  vint32m1_t v26;
  v26 = v22;
  vint32m1_t v27;
  v27 = v23;
  vint32m1_t v28;
  v28 = v24;
  for (size_t i29 = v10; i29 < v1; i29 += v11) {
    size_t v30 = i29 * v4;
    size_t v31 = 2;
    int8_t* v32 = &v3[v30];
    vint8mf4_t v33 = __riscv_vle8_v_i8mf4(v32, v31);
    vint16mf2_t v34 = __riscv_vsext_vf2_i16mf2(v33, v31);
    size_t v35 = i29 * v6;
    size_t v36 = v35 + v11;
    size_t v37 = v35 + v12;
    size_t v38 = v35 + v13;
    int8_t v39 = v5[v35];
    int16_t v40 = (int16_t) v39;
    vint32m1_t v41 = v25;
    vint32m1_t v42 = __riscv_vwmacc_vx_i32m1(v41, v40, v34, v31);
    int8_t v43 = v5[v36];
    int16_t v44 = (int16_t) v43;
    vint32m1_t v45 = v26;
    vint32m1_t v46 = __riscv_vwmacc_vx_i32m1(v45, v44, v34, v31);
    int8_t v47 = v5[v37];
    int16_t v48 = (int16_t) v47;
    vint32m1_t v49 = v27;
    vint32m1_t v50 = __riscv_vwmacc_vx_i32m1(v49, v48, v34, v31);
    int8_t v51 = v5[v38];
    int16_t v52 = (int16_t) v51;
    vint32m1_t v53 = v28;
    vint32m1_t v54 = __riscv_vwmacc_vx_i32m1(v53, v52, v34, v31);
    v25 = v42;
    v26 = v46;
    v27 = v50;
    v28 = v54;
  }
  vint32m1_t v55 = v25;
  vint32m1_t v56 = v26;
  vint32m1_t v57 = v27;
  vint32m1_t v58 = v28;
  int32_t* v59 = &v8[v10];
  __riscv_vse32_v_i32m1(v59, v55, v12);
  int32_t* v60 = &v8[v9];
  __riscv_vse32_v_i32m1(v60, v56, v12);
  size_t v61 = v12 * v9;
  int32_t* v62 = &v8[v61];
  __riscv_vse32_v_i32m1(v62, v57, v12);
  size_t v63 = v13 * v9;
  int32_t* v64 = &v8[v63];
  __riscv_vse32_v_i32m1(v64, v58, v12);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_2x4_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 2
//   NR         : 4
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_2x4_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v12);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v12);
  vint32m1_t v37;
  v37 = v24;
  vint32m1_t v38;
  v38 = v28;
  vint32m1_t v39;
  v39 = v32;
  vint32m1_t v40;
  v40 = v36;
  for (size_t i41 = v10; i41 < v1; i41 += v11) {
    size_t v42 = i41 * v4;
    size_t v43 = 2;
    int8_t* v44 = &v3[v42];
    vint8mf4_t v45 = __riscv_vle8_v_i8mf4(v44, v43);
    vint16mf2_t v46 = __riscv_vsext_vf2_i16mf2(v45, v43);
    size_t v47 = i41 * v6;
    size_t v48 = v47 + v11;
    size_t v49 = v47 + v12;
    size_t v50 = v47 + v13;
    int8_t v51 = v5[v47];
    int16_t v52 = (int16_t) v51;
    vint32m1_t v53 = v37;
    vint32m1_t v54 = __riscv_vwmacc_vx_i32m1(v53, v52, v46, v43);
    int8_t v55 = v5[v48];
    int16_t v56 = (int16_t) v55;
    vint32m1_t v57 = v38;
    vint32m1_t v58 = __riscv_vwmacc_vx_i32m1(v57, v56, v46, v43);
    int8_t v59 = v5[v49];
    int16_t v60 = (int16_t) v59;
    vint32m1_t v61 = v39;
    vint32m1_t v62 = __riscv_vwmacc_vx_i32m1(v61, v60, v46, v43);
    int8_t v63 = v5[v50];
    int16_t v64 = (int16_t) v63;
    vint32m1_t v65 = v40;
    vint32m1_t v66 = __riscv_vwmacc_vx_i32m1(v65, v64, v46, v43);
    v37 = v54;
    v38 = v58;
    v39 = v62;
    v40 = v66;
  }
  vint32m1_t v67 = v37;
  vint32m1_t v68 = v38;
  vint32m1_t v69 = v39;
  vint32m1_t v70 = v40;
  int32_t* v71 = &v8[v10];
  __riscv_vse32_v_i32m1(v71, v67, v12);
  int32_t* v72 = &v8[v9];
  __riscv_vse32_v_i32m1(v72, v68, v12);
  size_t v73 = v12 * v9;
  int32_t* v74 = &v8[v73];
  __riscv_vse32_v_i32m1(v74, v69, v12);
  size_t v75 = v13 * v9;
  int32_t* v76 = &v8[v75];
  __riscv_vse32_v_i32m1(v76, v70, v12);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_2x5_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 2
//   NR         : 5
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_2x5_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v12);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v12);
  vint32m1_t v25 = __riscv_vmv_v_x_i32m1(v20, v12);
  vint32m1_t v26;
  v26 = v21;
  vint32m1_t v27;
  v27 = v22;
  vint32m1_t v28;
  v28 = v23;
  vint32m1_t v29;
  v29 = v24;
  vint32m1_t v30;
  v30 = v25;
  for (size_t i31 = v10; i31 < v1; i31 += v11) {
    size_t v32 = i31 * v4;
    size_t v33 = 2;
    int8_t* v34 = &v3[v32];
    vint8mf4_t v35 = __riscv_vle8_v_i8mf4(v34, v33);
    vint16mf2_t v36 = __riscv_vsext_vf2_i16mf2(v35, v33);
    size_t v37 = i31 * v6;
    size_t v38 = v37 + v11;
    size_t v39 = v37 + v12;
    size_t v40 = v37 + v13;
    size_t v41 = v37 + v14;
    int8_t v42 = v5[v37];
    int16_t v43 = (int16_t) v42;
    vint32m1_t v44 = v26;
    vint32m1_t v45 = __riscv_vwmacc_vx_i32m1(v44, v43, v36, v33);
    int8_t v46 = v5[v38];
    int16_t v47 = (int16_t) v46;
    vint32m1_t v48 = v27;
    vint32m1_t v49 = __riscv_vwmacc_vx_i32m1(v48, v47, v36, v33);
    int8_t v50 = v5[v39];
    int16_t v51 = (int16_t) v50;
    vint32m1_t v52 = v28;
    vint32m1_t v53 = __riscv_vwmacc_vx_i32m1(v52, v51, v36, v33);
    int8_t v54 = v5[v40];
    int16_t v55 = (int16_t) v54;
    vint32m1_t v56 = v29;
    vint32m1_t v57 = __riscv_vwmacc_vx_i32m1(v56, v55, v36, v33);
    int8_t v58 = v5[v41];
    int16_t v59 = (int16_t) v58;
    vint32m1_t v60 = v30;
    vint32m1_t v61 = __riscv_vwmacc_vx_i32m1(v60, v59, v36, v33);
    v26 = v45;
    v27 = v49;
    v28 = v53;
    v29 = v57;
    v30 = v61;
  }
  vint32m1_t v62 = v26;
  vint32m1_t v63 = v27;
  vint32m1_t v64 = v28;
  vint32m1_t v65 = v29;
  vint32m1_t v66 = v30;
  int32_t* v67 = &v8[v10];
  __riscv_vse32_v_i32m1(v67, v62, v12);
  int32_t* v68 = &v8[v9];
  __riscv_vse32_v_i32m1(v68, v63, v12);
  size_t v69 = v12 * v9;
  int32_t* v70 = &v8[v69];
  __riscv_vse32_v_i32m1(v70, v64, v12);
  size_t v71 = v13 * v9;
  int32_t* v72 = &v8[v71];
  __riscv_vse32_v_i32m1(v72, v65, v12);
  size_t v73 = v14 * v9;
  int32_t* v74 = &v8[v73];
  __riscv_vse32_v_i32m1(v74, v66, v12);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_2x5_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 2
//   NR         : 5
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_2x5_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v12);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v12);
  size_t v37 = v14 * v9;
  size_t v38 = v10 + v37;
  int32_t* v39 = &v8[v38];
  vint32m1_t v40 = __riscv_vle32_v_i32m1(v39, v12);
  vint32m1_t v41;
  v41 = v24;
  vint32m1_t v42;
  v42 = v28;
  vint32m1_t v43;
  v43 = v32;
  vint32m1_t v44;
  v44 = v36;
  vint32m1_t v45;
  v45 = v40;
  for (size_t i46 = v10; i46 < v1; i46 += v11) {
    size_t v47 = i46 * v4;
    size_t v48 = 2;
    int8_t* v49 = &v3[v47];
    vint8mf4_t v50 = __riscv_vle8_v_i8mf4(v49, v48);
    vint16mf2_t v51 = __riscv_vsext_vf2_i16mf2(v50, v48);
    size_t v52 = i46 * v6;
    size_t v53 = v52 + v11;
    size_t v54 = v52 + v12;
    size_t v55 = v52 + v13;
    size_t v56 = v52 + v14;
    int8_t v57 = v5[v52];
    int16_t v58 = (int16_t) v57;
    vint32m1_t v59 = v41;
    vint32m1_t v60 = __riscv_vwmacc_vx_i32m1(v59, v58, v51, v48);
    int8_t v61 = v5[v53];
    int16_t v62 = (int16_t) v61;
    vint32m1_t v63 = v42;
    vint32m1_t v64 = __riscv_vwmacc_vx_i32m1(v63, v62, v51, v48);
    int8_t v65 = v5[v54];
    int16_t v66 = (int16_t) v65;
    vint32m1_t v67 = v43;
    vint32m1_t v68 = __riscv_vwmacc_vx_i32m1(v67, v66, v51, v48);
    int8_t v69 = v5[v55];
    int16_t v70 = (int16_t) v69;
    vint32m1_t v71 = v44;
    vint32m1_t v72 = __riscv_vwmacc_vx_i32m1(v71, v70, v51, v48);
    int8_t v73 = v5[v56];
    int16_t v74 = (int16_t) v73;
    vint32m1_t v75 = v45;
    vint32m1_t v76 = __riscv_vwmacc_vx_i32m1(v75, v74, v51, v48);
    v41 = v60;
    v42 = v64;
    v43 = v68;
    v44 = v72;
    v45 = v76;
  }
  vint32m1_t v77 = v41;
  vint32m1_t v78 = v42;
  vint32m1_t v79 = v43;
  vint32m1_t v80 = v44;
  vint32m1_t v81 = v45;
  int32_t* v82 = &v8[v10];
  __riscv_vse32_v_i32m1(v82, v77, v12);
  int32_t* v83 = &v8[v9];
  __riscv_vse32_v_i32m1(v83, v78, v12);
  size_t v84 = v12 * v9;
  int32_t* v85 = &v8[v84];
  __riscv_vse32_v_i32m1(v85, v79, v12);
  size_t v86 = v13 * v9;
  int32_t* v87 = &v8[v86];
  __riscv_vse32_v_i32m1(v87, v80, v12);
  size_t v88 = v14 * v9;
  int32_t* v89 = &v8[v88];
  __riscv_vse32_v_i32m1(v89, v81, v12);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_2x6_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 2
//   NR         : 6
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_2x6_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v12);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v12);
  vint32m1_t v25 = __riscv_vmv_v_x_i32m1(v20, v12);
  vint32m1_t v26 = __riscv_vmv_v_x_i32m1(v20, v12);
  vint32m1_t v27;
  v27 = v21;
  vint32m1_t v28;
  v28 = v22;
  vint32m1_t v29;
  v29 = v23;
  vint32m1_t v30;
  v30 = v24;
  vint32m1_t v31;
  v31 = v25;
  vint32m1_t v32;
  v32 = v26;
  for (size_t i33 = v10; i33 < v1; i33 += v11) {
    size_t v34 = i33 * v4;
    size_t v35 = 2;
    int8_t* v36 = &v3[v34];
    vint8mf4_t v37 = __riscv_vle8_v_i8mf4(v36, v35);
    vint16mf2_t v38 = __riscv_vsext_vf2_i16mf2(v37, v35);
    size_t v39 = i33 * v6;
    size_t v40 = v39 + v11;
    size_t v41 = v39 + v12;
    size_t v42 = v39 + v13;
    size_t v43 = v39 + v14;
    size_t v44 = v39 + v15;
    int8_t v45 = v5[v39];
    int16_t v46 = (int16_t) v45;
    vint32m1_t v47 = v27;
    vint32m1_t v48 = __riscv_vwmacc_vx_i32m1(v47, v46, v38, v35);
    int8_t v49 = v5[v40];
    int16_t v50 = (int16_t) v49;
    vint32m1_t v51 = v28;
    vint32m1_t v52 = __riscv_vwmacc_vx_i32m1(v51, v50, v38, v35);
    int8_t v53 = v5[v41];
    int16_t v54 = (int16_t) v53;
    vint32m1_t v55 = v29;
    vint32m1_t v56 = __riscv_vwmacc_vx_i32m1(v55, v54, v38, v35);
    int8_t v57 = v5[v42];
    int16_t v58 = (int16_t) v57;
    vint32m1_t v59 = v30;
    vint32m1_t v60 = __riscv_vwmacc_vx_i32m1(v59, v58, v38, v35);
    int8_t v61 = v5[v43];
    int16_t v62 = (int16_t) v61;
    vint32m1_t v63 = v31;
    vint32m1_t v64 = __riscv_vwmacc_vx_i32m1(v63, v62, v38, v35);
    int8_t v65 = v5[v44];
    int16_t v66 = (int16_t) v65;
    vint32m1_t v67 = v32;
    vint32m1_t v68 = __riscv_vwmacc_vx_i32m1(v67, v66, v38, v35);
    v27 = v48;
    v28 = v52;
    v29 = v56;
    v30 = v60;
    v31 = v64;
    v32 = v68;
  }
  vint32m1_t v69 = v27;
  vint32m1_t v70 = v28;
  vint32m1_t v71 = v29;
  vint32m1_t v72 = v30;
  vint32m1_t v73 = v31;
  vint32m1_t v74 = v32;
  int32_t* v75 = &v8[v10];
  __riscv_vse32_v_i32m1(v75, v69, v12);
  int32_t* v76 = &v8[v9];
  __riscv_vse32_v_i32m1(v76, v70, v12);
  size_t v77 = v12 * v9;
  int32_t* v78 = &v8[v77];
  __riscv_vse32_v_i32m1(v78, v71, v12);
  size_t v79 = v13 * v9;
  int32_t* v80 = &v8[v79];
  __riscv_vse32_v_i32m1(v80, v72, v12);
  size_t v81 = v14 * v9;
  int32_t* v82 = &v8[v81];
  __riscv_vse32_v_i32m1(v82, v73, v12);
  size_t v83 = v15 * v9;
  int32_t* v84 = &v8[v83];
  __riscv_vse32_v_i32m1(v84, v74, v12);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_2x6_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 2
//   NR         : 6
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_2x6_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v12);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v12);
  size_t v37 = v14 * v9;
  size_t v38 = v10 + v37;
  int32_t* v39 = &v8[v38];
  vint32m1_t v40 = __riscv_vle32_v_i32m1(v39, v12);
  size_t v41 = v15 * v9;
  size_t v42 = v10 + v41;
  int32_t* v43 = &v8[v42];
  vint32m1_t v44 = __riscv_vle32_v_i32m1(v43, v12);
  vint32m1_t v45;
  v45 = v24;
  vint32m1_t v46;
  v46 = v28;
  vint32m1_t v47;
  v47 = v32;
  vint32m1_t v48;
  v48 = v36;
  vint32m1_t v49;
  v49 = v40;
  vint32m1_t v50;
  v50 = v44;
  for (size_t i51 = v10; i51 < v1; i51 += v11) {
    size_t v52 = i51 * v4;
    size_t v53 = 2;
    int8_t* v54 = &v3[v52];
    vint8mf4_t v55 = __riscv_vle8_v_i8mf4(v54, v53);
    vint16mf2_t v56 = __riscv_vsext_vf2_i16mf2(v55, v53);
    size_t v57 = i51 * v6;
    size_t v58 = v57 + v11;
    size_t v59 = v57 + v12;
    size_t v60 = v57 + v13;
    size_t v61 = v57 + v14;
    size_t v62 = v57 + v15;
    int8_t v63 = v5[v57];
    int16_t v64 = (int16_t) v63;
    vint32m1_t v65 = v45;
    vint32m1_t v66 = __riscv_vwmacc_vx_i32m1(v65, v64, v56, v53);
    int8_t v67 = v5[v58];
    int16_t v68 = (int16_t) v67;
    vint32m1_t v69 = v46;
    vint32m1_t v70 = __riscv_vwmacc_vx_i32m1(v69, v68, v56, v53);
    int8_t v71 = v5[v59];
    int16_t v72 = (int16_t) v71;
    vint32m1_t v73 = v47;
    vint32m1_t v74 = __riscv_vwmacc_vx_i32m1(v73, v72, v56, v53);
    int8_t v75 = v5[v60];
    int16_t v76 = (int16_t) v75;
    vint32m1_t v77 = v48;
    vint32m1_t v78 = __riscv_vwmacc_vx_i32m1(v77, v76, v56, v53);
    int8_t v79 = v5[v61];
    int16_t v80 = (int16_t) v79;
    vint32m1_t v81 = v49;
    vint32m1_t v82 = __riscv_vwmacc_vx_i32m1(v81, v80, v56, v53);
    int8_t v83 = v5[v62];
    int16_t v84 = (int16_t) v83;
    vint32m1_t v85 = v50;
    vint32m1_t v86 = __riscv_vwmacc_vx_i32m1(v85, v84, v56, v53);
    v45 = v66;
    v46 = v70;
    v47 = v74;
    v48 = v78;
    v49 = v82;
    v50 = v86;
  }
  vint32m1_t v87 = v45;
  vint32m1_t v88 = v46;
  vint32m1_t v89 = v47;
  vint32m1_t v90 = v48;
  vint32m1_t v91 = v49;
  vint32m1_t v92 = v50;
  int32_t* v93 = &v8[v10];
  __riscv_vse32_v_i32m1(v93, v87, v12);
  int32_t* v94 = &v8[v9];
  __riscv_vse32_v_i32m1(v94, v88, v12);
  size_t v95 = v12 * v9;
  int32_t* v96 = &v8[v95];
  __riscv_vse32_v_i32m1(v96, v89, v12);
  size_t v97 = v13 * v9;
  int32_t* v98 = &v8[v97];
  __riscv_vse32_v_i32m1(v98, v90, v12);
  size_t v99 = v14 * v9;
  int32_t* v100 = &v8[v99];
  __riscv_vse32_v_i32m1(v100, v91, v12);
  size_t v101 = v15 * v9;
  int32_t* v102 = &v8[v101];
  __riscv_vse32_v_i32m1(v102, v92, v12);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_2x7_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 2
//   NR         : 7
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_2x7_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v12);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v12);
  vint32m1_t v25 = __riscv_vmv_v_x_i32m1(v20, v12);
  vint32m1_t v26 = __riscv_vmv_v_x_i32m1(v20, v12);
  vint32m1_t v27 = __riscv_vmv_v_x_i32m1(v20, v12);
  vint32m1_t v28;
  v28 = v21;
  vint32m1_t v29;
  v29 = v22;
  vint32m1_t v30;
  v30 = v23;
  vint32m1_t v31;
  v31 = v24;
  vint32m1_t v32;
  v32 = v25;
  vint32m1_t v33;
  v33 = v26;
  vint32m1_t v34;
  v34 = v27;
  for (size_t i35 = v10; i35 < v1; i35 += v11) {
    size_t v36 = i35 * v4;
    size_t v37 = 2;
    int8_t* v38 = &v3[v36];
    vint8mf4_t v39 = __riscv_vle8_v_i8mf4(v38, v37);
    vint16mf2_t v40 = __riscv_vsext_vf2_i16mf2(v39, v37);
    size_t v41 = i35 * v6;
    size_t v42 = v41 + v11;
    size_t v43 = v41 + v12;
    size_t v44 = v41 + v13;
    size_t v45 = v41 + v14;
    size_t v46 = v41 + v15;
    size_t v47 = v41 + v16;
    int8_t v48 = v5[v41];
    int16_t v49 = (int16_t) v48;
    vint32m1_t v50 = v28;
    vint32m1_t v51 = __riscv_vwmacc_vx_i32m1(v50, v49, v40, v37);
    int8_t v52 = v5[v42];
    int16_t v53 = (int16_t) v52;
    vint32m1_t v54 = v29;
    vint32m1_t v55 = __riscv_vwmacc_vx_i32m1(v54, v53, v40, v37);
    int8_t v56 = v5[v43];
    int16_t v57 = (int16_t) v56;
    vint32m1_t v58 = v30;
    vint32m1_t v59 = __riscv_vwmacc_vx_i32m1(v58, v57, v40, v37);
    int8_t v60 = v5[v44];
    int16_t v61 = (int16_t) v60;
    vint32m1_t v62 = v31;
    vint32m1_t v63 = __riscv_vwmacc_vx_i32m1(v62, v61, v40, v37);
    int8_t v64 = v5[v45];
    int16_t v65 = (int16_t) v64;
    vint32m1_t v66 = v32;
    vint32m1_t v67 = __riscv_vwmacc_vx_i32m1(v66, v65, v40, v37);
    int8_t v68 = v5[v46];
    int16_t v69 = (int16_t) v68;
    vint32m1_t v70 = v33;
    vint32m1_t v71 = __riscv_vwmacc_vx_i32m1(v70, v69, v40, v37);
    int8_t v72 = v5[v47];
    int16_t v73 = (int16_t) v72;
    vint32m1_t v74 = v34;
    vint32m1_t v75 = __riscv_vwmacc_vx_i32m1(v74, v73, v40, v37);
    v28 = v51;
    v29 = v55;
    v30 = v59;
    v31 = v63;
    v32 = v67;
    v33 = v71;
    v34 = v75;
  }
  vint32m1_t v76 = v28;
  vint32m1_t v77 = v29;
  vint32m1_t v78 = v30;
  vint32m1_t v79 = v31;
  vint32m1_t v80 = v32;
  vint32m1_t v81 = v33;
  vint32m1_t v82 = v34;
  int32_t* v83 = &v8[v10];
  __riscv_vse32_v_i32m1(v83, v76, v12);
  int32_t* v84 = &v8[v9];
  __riscv_vse32_v_i32m1(v84, v77, v12);
  size_t v85 = v12 * v9;
  int32_t* v86 = &v8[v85];
  __riscv_vse32_v_i32m1(v86, v78, v12);
  size_t v87 = v13 * v9;
  int32_t* v88 = &v8[v87];
  __riscv_vse32_v_i32m1(v88, v79, v12);
  size_t v89 = v14 * v9;
  int32_t* v90 = &v8[v89];
  __riscv_vse32_v_i32m1(v90, v80, v12);
  size_t v91 = v15 * v9;
  int32_t* v92 = &v8[v91];
  __riscv_vse32_v_i32m1(v92, v81, v12);
  size_t v93 = v16 * v9;
  int32_t* v94 = &v8[v93];
  __riscv_vse32_v_i32m1(v94, v82, v12);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_2x7_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 2
//   NR         : 7
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_2x7_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v12);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v12);
  size_t v37 = v14 * v9;
  size_t v38 = v10 + v37;
  int32_t* v39 = &v8[v38];
  vint32m1_t v40 = __riscv_vle32_v_i32m1(v39, v12);
  size_t v41 = v15 * v9;
  size_t v42 = v10 + v41;
  int32_t* v43 = &v8[v42];
  vint32m1_t v44 = __riscv_vle32_v_i32m1(v43, v12);
  size_t v45 = v16 * v9;
  size_t v46 = v10 + v45;
  int32_t* v47 = &v8[v46];
  vint32m1_t v48 = __riscv_vle32_v_i32m1(v47, v12);
  vint32m1_t v49;
  v49 = v24;
  vint32m1_t v50;
  v50 = v28;
  vint32m1_t v51;
  v51 = v32;
  vint32m1_t v52;
  v52 = v36;
  vint32m1_t v53;
  v53 = v40;
  vint32m1_t v54;
  v54 = v44;
  vint32m1_t v55;
  v55 = v48;
  for (size_t i56 = v10; i56 < v1; i56 += v11) {
    size_t v57 = i56 * v4;
    size_t v58 = 2;
    int8_t* v59 = &v3[v57];
    vint8mf4_t v60 = __riscv_vle8_v_i8mf4(v59, v58);
    vint16mf2_t v61 = __riscv_vsext_vf2_i16mf2(v60, v58);
    size_t v62 = i56 * v6;
    size_t v63 = v62 + v11;
    size_t v64 = v62 + v12;
    size_t v65 = v62 + v13;
    size_t v66 = v62 + v14;
    size_t v67 = v62 + v15;
    size_t v68 = v62 + v16;
    int8_t v69 = v5[v62];
    int16_t v70 = (int16_t) v69;
    vint32m1_t v71 = v49;
    vint32m1_t v72 = __riscv_vwmacc_vx_i32m1(v71, v70, v61, v58);
    int8_t v73 = v5[v63];
    int16_t v74 = (int16_t) v73;
    vint32m1_t v75 = v50;
    vint32m1_t v76 = __riscv_vwmacc_vx_i32m1(v75, v74, v61, v58);
    int8_t v77 = v5[v64];
    int16_t v78 = (int16_t) v77;
    vint32m1_t v79 = v51;
    vint32m1_t v80 = __riscv_vwmacc_vx_i32m1(v79, v78, v61, v58);
    int8_t v81 = v5[v65];
    int16_t v82 = (int16_t) v81;
    vint32m1_t v83 = v52;
    vint32m1_t v84 = __riscv_vwmacc_vx_i32m1(v83, v82, v61, v58);
    int8_t v85 = v5[v66];
    int16_t v86 = (int16_t) v85;
    vint32m1_t v87 = v53;
    vint32m1_t v88 = __riscv_vwmacc_vx_i32m1(v87, v86, v61, v58);
    int8_t v89 = v5[v67];
    int16_t v90 = (int16_t) v89;
    vint32m1_t v91 = v54;
    vint32m1_t v92 = __riscv_vwmacc_vx_i32m1(v91, v90, v61, v58);
    int8_t v93 = v5[v68];
    int16_t v94 = (int16_t) v93;
    vint32m1_t v95 = v55;
    vint32m1_t v96 = __riscv_vwmacc_vx_i32m1(v95, v94, v61, v58);
    v49 = v72;
    v50 = v76;
    v51 = v80;
    v52 = v84;
    v53 = v88;
    v54 = v92;
    v55 = v96;
  }
  vint32m1_t v97 = v49;
  vint32m1_t v98 = v50;
  vint32m1_t v99 = v51;
  vint32m1_t v100 = v52;
  vint32m1_t v101 = v53;
  vint32m1_t v102 = v54;
  vint32m1_t v103 = v55;
  int32_t* v104 = &v8[v10];
  __riscv_vse32_v_i32m1(v104, v97, v12);
  int32_t* v105 = &v8[v9];
  __riscv_vse32_v_i32m1(v105, v98, v12);
  size_t v106 = v12 * v9;
  int32_t* v107 = &v8[v106];
  __riscv_vse32_v_i32m1(v107, v99, v12);
  size_t v108 = v13 * v9;
  int32_t* v109 = &v8[v108];
  __riscv_vse32_v_i32m1(v109, v100, v12);
  size_t v110 = v14 * v9;
  int32_t* v111 = &v8[v110];
  __riscv_vse32_v_i32m1(v111, v101, v12);
  size_t v112 = v15 * v9;
  int32_t* v113 = &v8[v112];
  __riscv_vse32_v_i32m1(v113, v102, v12);
  size_t v114 = v16 * v9;
  int32_t* v115 = &v8[v114];
  __riscv_vse32_v_i32m1(v115, v103, v12);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_2x8_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 2
//   NR         : 8
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_2x8_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v12);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v12);
  vint32m1_t v25 = __riscv_vmv_v_x_i32m1(v20, v12);
  vint32m1_t v26 = __riscv_vmv_v_x_i32m1(v20, v12);
  vint32m1_t v27 = __riscv_vmv_v_x_i32m1(v20, v12);
  vint32m1_t v28 = __riscv_vmv_v_x_i32m1(v20, v12);
  vint32m1_t v29;
  v29 = v21;
  vint32m1_t v30;
  v30 = v22;
  vint32m1_t v31;
  v31 = v23;
  vint32m1_t v32;
  v32 = v24;
  vint32m1_t v33;
  v33 = v25;
  vint32m1_t v34;
  v34 = v26;
  vint32m1_t v35;
  v35 = v27;
  vint32m1_t v36;
  v36 = v28;
  for (size_t i37 = v10; i37 < v1; i37 += v11) {
    size_t v38 = i37 * v4;
    size_t v39 = 2;
    int8_t* v40 = &v3[v38];
    vint8mf4_t v41 = __riscv_vle8_v_i8mf4(v40, v39);
    vint16mf2_t v42 = __riscv_vsext_vf2_i16mf2(v41, v39);
    size_t v43 = i37 * v6;
    size_t v44 = v43 + v11;
    size_t v45 = v43 + v12;
    size_t v46 = v43 + v13;
    size_t v47 = v43 + v14;
    size_t v48 = v43 + v15;
    size_t v49 = v43 + v16;
    size_t v50 = v43 + v17;
    int8_t v51 = v5[v43];
    int16_t v52 = (int16_t) v51;
    vint32m1_t v53 = v29;
    vint32m1_t v54 = __riscv_vwmacc_vx_i32m1(v53, v52, v42, v39);
    int8_t v55 = v5[v44];
    int16_t v56 = (int16_t) v55;
    vint32m1_t v57 = v30;
    vint32m1_t v58 = __riscv_vwmacc_vx_i32m1(v57, v56, v42, v39);
    int8_t v59 = v5[v45];
    int16_t v60 = (int16_t) v59;
    vint32m1_t v61 = v31;
    vint32m1_t v62 = __riscv_vwmacc_vx_i32m1(v61, v60, v42, v39);
    int8_t v63 = v5[v46];
    int16_t v64 = (int16_t) v63;
    vint32m1_t v65 = v32;
    vint32m1_t v66 = __riscv_vwmacc_vx_i32m1(v65, v64, v42, v39);
    int8_t v67 = v5[v47];
    int16_t v68 = (int16_t) v67;
    vint32m1_t v69 = v33;
    vint32m1_t v70 = __riscv_vwmacc_vx_i32m1(v69, v68, v42, v39);
    int8_t v71 = v5[v48];
    int16_t v72 = (int16_t) v71;
    vint32m1_t v73 = v34;
    vint32m1_t v74 = __riscv_vwmacc_vx_i32m1(v73, v72, v42, v39);
    int8_t v75 = v5[v49];
    int16_t v76 = (int16_t) v75;
    vint32m1_t v77 = v35;
    vint32m1_t v78 = __riscv_vwmacc_vx_i32m1(v77, v76, v42, v39);
    int8_t v79 = v5[v50];
    int16_t v80 = (int16_t) v79;
    vint32m1_t v81 = v36;
    vint32m1_t v82 = __riscv_vwmacc_vx_i32m1(v81, v80, v42, v39);
    v29 = v54;
    v30 = v58;
    v31 = v62;
    v32 = v66;
    v33 = v70;
    v34 = v74;
    v35 = v78;
    v36 = v82;
  }
  vint32m1_t v83 = v29;
  vint32m1_t v84 = v30;
  vint32m1_t v85 = v31;
  vint32m1_t v86 = v32;
  vint32m1_t v87 = v33;
  vint32m1_t v88 = v34;
  vint32m1_t v89 = v35;
  vint32m1_t v90 = v36;
  int32_t* v91 = &v8[v10];
  __riscv_vse32_v_i32m1(v91, v83, v12);
  int32_t* v92 = &v8[v9];
  __riscv_vse32_v_i32m1(v92, v84, v12);
  size_t v93 = v12 * v9;
  int32_t* v94 = &v8[v93];
  __riscv_vse32_v_i32m1(v94, v85, v12);
  size_t v95 = v13 * v9;
  int32_t* v96 = &v8[v95];
  __riscv_vse32_v_i32m1(v96, v86, v12);
  size_t v97 = v14 * v9;
  int32_t* v98 = &v8[v97];
  __riscv_vse32_v_i32m1(v98, v87, v12);
  size_t v99 = v15 * v9;
  int32_t* v100 = &v8[v99];
  __riscv_vse32_v_i32m1(v100, v88, v12);
  size_t v101 = v16 * v9;
  int32_t* v102 = &v8[v101];
  __riscv_vse32_v_i32m1(v102, v89, v12);
  size_t v103 = v17 * v9;
  int32_t* v104 = &v8[v103];
  __riscv_vse32_v_i32m1(v104, v90, v12);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_2x8_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 2
//   NR         : 8
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_2x8_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v12);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v12);
  size_t v37 = v14 * v9;
  size_t v38 = v10 + v37;
  int32_t* v39 = &v8[v38];
  vint32m1_t v40 = __riscv_vle32_v_i32m1(v39, v12);
  size_t v41 = v15 * v9;
  size_t v42 = v10 + v41;
  int32_t* v43 = &v8[v42];
  vint32m1_t v44 = __riscv_vle32_v_i32m1(v43, v12);
  size_t v45 = v16 * v9;
  size_t v46 = v10 + v45;
  int32_t* v47 = &v8[v46];
  vint32m1_t v48 = __riscv_vle32_v_i32m1(v47, v12);
  size_t v49 = v17 * v9;
  size_t v50 = v10 + v49;
  int32_t* v51 = &v8[v50];
  vint32m1_t v52 = __riscv_vle32_v_i32m1(v51, v12);
  vint32m1_t v53;
  v53 = v24;
  vint32m1_t v54;
  v54 = v28;
  vint32m1_t v55;
  v55 = v32;
  vint32m1_t v56;
  v56 = v36;
  vint32m1_t v57;
  v57 = v40;
  vint32m1_t v58;
  v58 = v44;
  vint32m1_t v59;
  v59 = v48;
  vint32m1_t v60;
  v60 = v52;
  for (size_t i61 = v10; i61 < v1; i61 += v11) {
    size_t v62 = i61 * v4;
    size_t v63 = 2;
    int8_t* v64 = &v3[v62];
    vint8mf4_t v65 = __riscv_vle8_v_i8mf4(v64, v63);
    vint16mf2_t v66 = __riscv_vsext_vf2_i16mf2(v65, v63);
    size_t v67 = i61 * v6;
    size_t v68 = v67 + v11;
    size_t v69 = v67 + v12;
    size_t v70 = v67 + v13;
    size_t v71 = v67 + v14;
    size_t v72 = v67 + v15;
    size_t v73 = v67 + v16;
    size_t v74 = v67 + v17;
    int8_t v75 = v5[v67];
    int16_t v76 = (int16_t) v75;
    vint32m1_t v77 = v53;
    vint32m1_t v78 = __riscv_vwmacc_vx_i32m1(v77, v76, v66, v63);
    int8_t v79 = v5[v68];
    int16_t v80 = (int16_t) v79;
    vint32m1_t v81 = v54;
    vint32m1_t v82 = __riscv_vwmacc_vx_i32m1(v81, v80, v66, v63);
    int8_t v83 = v5[v69];
    int16_t v84 = (int16_t) v83;
    vint32m1_t v85 = v55;
    vint32m1_t v86 = __riscv_vwmacc_vx_i32m1(v85, v84, v66, v63);
    int8_t v87 = v5[v70];
    int16_t v88 = (int16_t) v87;
    vint32m1_t v89 = v56;
    vint32m1_t v90 = __riscv_vwmacc_vx_i32m1(v89, v88, v66, v63);
    int8_t v91 = v5[v71];
    int16_t v92 = (int16_t) v91;
    vint32m1_t v93 = v57;
    vint32m1_t v94 = __riscv_vwmacc_vx_i32m1(v93, v92, v66, v63);
    int8_t v95 = v5[v72];
    int16_t v96 = (int16_t) v95;
    vint32m1_t v97 = v58;
    vint32m1_t v98 = __riscv_vwmacc_vx_i32m1(v97, v96, v66, v63);
    int8_t v99 = v5[v73];
    int16_t v100 = (int16_t) v99;
    vint32m1_t v101 = v59;
    vint32m1_t v102 = __riscv_vwmacc_vx_i32m1(v101, v100, v66, v63);
    int8_t v103 = v5[v74];
    int16_t v104 = (int16_t) v103;
    vint32m1_t v105 = v60;
    vint32m1_t v106 = __riscv_vwmacc_vx_i32m1(v105, v104, v66, v63);
    v53 = v78;
    v54 = v82;
    v55 = v86;
    v56 = v90;
    v57 = v94;
    v58 = v98;
    v59 = v102;
    v60 = v106;
  }
  vint32m1_t v107 = v53;
  vint32m1_t v108 = v54;
  vint32m1_t v109 = v55;
  vint32m1_t v110 = v56;
  vint32m1_t v111 = v57;
  vint32m1_t v112 = v58;
  vint32m1_t v113 = v59;
  vint32m1_t v114 = v60;
  int32_t* v115 = &v8[v10];
  __riscv_vse32_v_i32m1(v115, v107, v12);
  int32_t* v116 = &v8[v9];
  __riscv_vse32_v_i32m1(v116, v108, v12);
  size_t v117 = v12 * v9;
  int32_t* v118 = &v8[v117];
  __riscv_vse32_v_i32m1(v118, v109, v12);
  size_t v119 = v13 * v9;
  int32_t* v120 = &v8[v119];
  __riscv_vse32_v_i32m1(v120, v110, v12);
  size_t v121 = v14 * v9;
  int32_t* v122 = &v8[v121];
  __riscv_vse32_v_i32m1(v122, v111, v12);
  size_t v123 = v15 * v9;
  int32_t* v124 = &v8[v123];
  __riscv_vse32_v_i32m1(v124, v112, v12);
  size_t v125 = v16 * v9;
  int32_t* v126 = &v8[v125];
  __riscv_vse32_v_i32m1(v126, v113, v12);
  size_t v127 = v17 * v9;
  int32_t* v128 = &v8[v127];
  __riscv_vse32_v_i32m1(v128, v114, v12);
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
//   Generated   : 2026-06-29 11:41:10
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
//   Generated   : 2026-06-29 11:41:10
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
//   Generated   : 2026-06-29 11:41:10
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
//   Generated   : 2026-06-29 11:41:10
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
// Merged from xdsl_api_gemm_3x3_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 3
//   NR         : 3
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_3x3_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v13);
  vint32m1_t v24;
  v24 = v21;
  vint32m1_t v25;
  v25 = v22;
  vint32m1_t v26;
  v26 = v23;
  for (size_t i27 = v10; i27 < v1; i27 += v11) {
    size_t v28 = i27 * v4;
    size_t v29 = 3;
    int8_t* v30 = &v3[v28];
    vint8mf4_t v31 = __riscv_vle8_v_i8mf4(v30, v29);
    vint16mf2_t v32 = __riscv_vsext_vf2_i16mf2(v31, v29);
    size_t v33 = i27 * v6;
    size_t v34 = v33 + v11;
    size_t v35 = v33 + v12;
    int8_t v36 = v5[v33];
    int16_t v37 = (int16_t) v36;
    vint32m1_t v38 = v24;
    vint32m1_t v39 = __riscv_vwmacc_vx_i32m1(v38, v37, v32, v29);
    int8_t v40 = v5[v34];
    int16_t v41 = (int16_t) v40;
    vint32m1_t v42 = v25;
    vint32m1_t v43 = __riscv_vwmacc_vx_i32m1(v42, v41, v32, v29);
    int8_t v44 = v5[v35];
    int16_t v45 = (int16_t) v44;
    vint32m1_t v46 = v26;
    vint32m1_t v47 = __riscv_vwmacc_vx_i32m1(v46, v45, v32, v29);
    v24 = v39;
    v25 = v43;
    v26 = v47;
  }
  vint32m1_t v48 = v24;
  vint32m1_t v49 = v25;
  vint32m1_t v50 = v26;
  int32_t* v51 = &v8[v10];
  __riscv_vse32_v_i32m1(v51, v48, v13);
  int32_t* v52 = &v8[v9];
  __riscv_vse32_v_i32m1(v52, v49, v13);
  size_t v53 = v12 * v9;
  int32_t* v54 = &v8[v53];
  __riscv_vse32_v_i32m1(v54, v50, v13);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_3x3_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 3
//   NR         : 3
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_3x3_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v13);
  vint32m1_t v33;
  v33 = v24;
  vint32m1_t v34;
  v34 = v28;
  vint32m1_t v35;
  v35 = v32;
  for (size_t i36 = v10; i36 < v1; i36 += v11) {
    size_t v37 = i36 * v4;
    size_t v38 = 3;
    int8_t* v39 = &v3[v37];
    vint8mf4_t v40 = __riscv_vle8_v_i8mf4(v39, v38);
    vint16mf2_t v41 = __riscv_vsext_vf2_i16mf2(v40, v38);
    size_t v42 = i36 * v6;
    size_t v43 = v42 + v11;
    size_t v44 = v42 + v12;
    int8_t v45 = v5[v42];
    int16_t v46 = (int16_t) v45;
    vint32m1_t v47 = v33;
    vint32m1_t v48 = __riscv_vwmacc_vx_i32m1(v47, v46, v41, v38);
    int8_t v49 = v5[v43];
    int16_t v50 = (int16_t) v49;
    vint32m1_t v51 = v34;
    vint32m1_t v52 = __riscv_vwmacc_vx_i32m1(v51, v50, v41, v38);
    int8_t v53 = v5[v44];
    int16_t v54 = (int16_t) v53;
    vint32m1_t v55 = v35;
    vint32m1_t v56 = __riscv_vwmacc_vx_i32m1(v55, v54, v41, v38);
    v33 = v48;
    v34 = v52;
    v35 = v56;
  }
  vint32m1_t v57 = v33;
  vint32m1_t v58 = v34;
  vint32m1_t v59 = v35;
  int32_t* v60 = &v8[v10];
  __riscv_vse32_v_i32m1(v60, v57, v13);
  int32_t* v61 = &v8[v9];
  __riscv_vse32_v_i32m1(v61, v58, v13);
  size_t v62 = v12 * v9;
  int32_t* v63 = &v8[v62];
  __riscv_vse32_v_i32m1(v63, v59, v13);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_3x4_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 3
//   NR         : 4
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_3x4_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v13);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v13);
  vint32m1_t v25;
  v25 = v21;
  vint32m1_t v26;
  v26 = v22;
  vint32m1_t v27;
  v27 = v23;
  vint32m1_t v28;
  v28 = v24;
  for (size_t i29 = v10; i29 < v1; i29 += v11) {
    size_t v30 = i29 * v4;
    size_t v31 = 3;
    int8_t* v32 = &v3[v30];
    vint8mf4_t v33 = __riscv_vle8_v_i8mf4(v32, v31);
    vint16mf2_t v34 = __riscv_vsext_vf2_i16mf2(v33, v31);
    size_t v35 = i29 * v6;
    size_t v36 = v35 + v11;
    size_t v37 = v35 + v12;
    size_t v38 = v35 + v13;
    int8_t v39 = v5[v35];
    int16_t v40 = (int16_t) v39;
    vint32m1_t v41 = v25;
    vint32m1_t v42 = __riscv_vwmacc_vx_i32m1(v41, v40, v34, v31);
    int8_t v43 = v5[v36];
    int16_t v44 = (int16_t) v43;
    vint32m1_t v45 = v26;
    vint32m1_t v46 = __riscv_vwmacc_vx_i32m1(v45, v44, v34, v31);
    int8_t v47 = v5[v37];
    int16_t v48 = (int16_t) v47;
    vint32m1_t v49 = v27;
    vint32m1_t v50 = __riscv_vwmacc_vx_i32m1(v49, v48, v34, v31);
    int8_t v51 = v5[v38];
    int16_t v52 = (int16_t) v51;
    vint32m1_t v53 = v28;
    vint32m1_t v54 = __riscv_vwmacc_vx_i32m1(v53, v52, v34, v31);
    v25 = v42;
    v26 = v46;
    v27 = v50;
    v28 = v54;
  }
  vint32m1_t v55 = v25;
  vint32m1_t v56 = v26;
  vint32m1_t v57 = v27;
  vint32m1_t v58 = v28;
  int32_t* v59 = &v8[v10];
  __riscv_vse32_v_i32m1(v59, v55, v13);
  int32_t* v60 = &v8[v9];
  __riscv_vse32_v_i32m1(v60, v56, v13);
  size_t v61 = v12 * v9;
  int32_t* v62 = &v8[v61];
  __riscv_vse32_v_i32m1(v62, v57, v13);
  size_t v63 = v13 * v9;
  int32_t* v64 = &v8[v63];
  __riscv_vse32_v_i32m1(v64, v58, v13);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_3x4_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 3
//   NR         : 4
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_3x4_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v13);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v13);
  vint32m1_t v37;
  v37 = v24;
  vint32m1_t v38;
  v38 = v28;
  vint32m1_t v39;
  v39 = v32;
  vint32m1_t v40;
  v40 = v36;
  for (size_t i41 = v10; i41 < v1; i41 += v11) {
    size_t v42 = i41 * v4;
    size_t v43 = 3;
    int8_t* v44 = &v3[v42];
    vint8mf4_t v45 = __riscv_vle8_v_i8mf4(v44, v43);
    vint16mf2_t v46 = __riscv_vsext_vf2_i16mf2(v45, v43);
    size_t v47 = i41 * v6;
    size_t v48 = v47 + v11;
    size_t v49 = v47 + v12;
    size_t v50 = v47 + v13;
    int8_t v51 = v5[v47];
    int16_t v52 = (int16_t) v51;
    vint32m1_t v53 = v37;
    vint32m1_t v54 = __riscv_vwmacc_vx_i32m1(v53, v52, v46, v43);
    int8_t v55 = v5[v48];
    int16_t v56 = (int16_t) v55;
    vint32m1_t v57 = v38;
    vint32m1_t v58 = __riscv_vwmacc_vx_i32m1(v57, v56, v46, v43);
    int8_t v59 = v5[v49];
    int16_t v60 = (int16_t) v59;
    vint32m1_t v61 = v39;
    vint32m1_t v62 = __riscv_vwmacc_vx_i32m1(v61, v60, v46, v43);
    int8_t v63 = v5[v50];
    int16_t v64 = (int16_t) v63;
    vint32m1_t v65 = v40;
    vint32m1_t v66 = __riscv_vwmacc_vx_i32m1(v65, v64, v46, v43);
    v37 = v54;
    v38 = v58;
    v39 = v62;
    v40 = v66;
  }
  vint32m1_t v67 = v37;
  vint32m1_t v68 = v38;
  vint32m1_t v69 = v39;
  vint32m1_t v70 = v40;
  int32_t* v71 = &v8[v10];
  __riscv_vse32_v_i32m1(v71, v67, v13);
  int32_t* v72 = &v8[v9];
  __riscv_vse32_v_i32m1(v72, v68, v13);
  size_t v73 = v12 * v9;
  int32_t* v74 = &v8[v73];
  __riscv_vse32_v_i32m1(v74, v69, v13);
  size_t v75 = v13 * v9;
  int32_t* v76 = &v8[v75];
  __riscv_vse32_v_i32m1(v76, v70, v13);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_3x5_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 3
//   NR         : 5
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_3x5_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v13);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v13);
  vint32m1_t v25 = __riscv_vmv_v_x_i32m1(v20, v13);
  vint32m1_t v26;
  v26 = v21;
  vint32m1_t v27;
  v27 = v22;
  vint32m1_t v28;
  v28 = v23;
  vint32m1_t v29;
  v29 = v24;
  vint32m1_t v30;
  v30 = v25;
  for (size_t i31 = v10; i31 < v1; i31 += v11) {
    size_t v32 = i31 * v4;
    size_t v33 = 3;
    int8_t* v34 = &v3[v32];
    vint8mf4_t v35 = __riscv_vle8_v_i8mf4(v34, v33);
    vint16mf2_t v36 = __riscv_vsext_vf2_i16mf2(v35, v33);
    size_t v37 = i31 * v6;
    size_t v38 = v37 + v11;
    size_t v39 = v37 + v12;
    size_t v40 = v37 + v13;
    size_t v41 = v37 + v14;
    int8_t v42 = v5[v37];
    int16_t v43 = (int16_t) v42;
    vint32m1_t v44 = v26;
    vint32m1_t v45 = __riscv_vwmacc_vx_i32m1(v44, v43, v36, v33);
    int8_t v46 = v5[v38];
    int16_t v47 = (int16_t) v46;
    vint32m1_t v48 = v27;
    vint32m1_t v49 = __riscv_vwmacc_vx_i32m1(v48, v47, v36, v33);
    int8_t v50 = v5[v39];
    int16_t v51 = (int16_t) v50;
    vint32m1_t v52 = v28;
    vint32m1_t v53 = __riscv_vwmacc_vx_i32m1(v52, v51, v36, v33);
    int8_t v54 = v5[v40];
    int16_t v55 = (int16_t) v54;
    vint32m1_t v56 = v29;
    vint32m1_t v57 = __riscv_vwmacc_vx_i32m1(v56, v55, v36, v33);
    int8_t v58 = v5[v41];
    int16_t v59 = (int16_t) v58;
    vint32m1_t v60 = v30;
    vint32m1_t v61 = __riscv_vwmacc_vx_i32m1(v60, v59, v36, v33);
    v26 = v45;
    v27 = v49;
    v28 = v53;
    v29 = v57;
    v30 = v61;
  }
  vint32m1_t v62 = v26;
  vint32m1_t v63 = v27;
  vint32m1_t v64 = v28;
  vint32m1_t v65 = v29;
  vint32m1_t v66 = v30;
  int32_t* v67 = &v8[v10];
  __riscv_vse32_v_i32m1(v67, v62, v13);
  int32_t* v68 = &v8[v9];
  __riscv_vse32_v_i32m1(v68, v63, v13);
  size_t v69 = v12 * v9;
  int32_t* v70 = &v8[v69];
  __riscv_vse32_v_i32m1(v70, v64, v13);
  size_t v71 = v13 * v9;
  int32_t* v72 = &v8[v71];
  __riscv_vse32_v_i32m1(v72, v65, v13);
  size_t v73 = v14 * v9;
  int32_t* v74 = &v8[v73];
  __riscv_vse32_v_i32m1(v74, v66, v13);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_3x5_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 3
//   NR         : 5
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_3x5_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v13);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v13);
  size_t v37 = v14 * v9;
  size_t v38 = v10 + v37;
  int32_t* v39 = &v8[v38];
  vint32m1_t v40 = __riscv_vle32_v_i32m1(v39, v13);
  vint32m1_t v41;
  v41 = v24;
  vint32m1_t v42;
  v42 = v28;
  vint32m1_t v43;
  v43 = v32;
  vint32m1_t v44;
  v44 = v36;
  vint32m1_t v45;
  v45 = v40;
  for (size_t i46 = v10; i46 < v1; i46 += v11) {
    size_t v47 = i46 * v4;
    size_t v48 = 3;
    int8_t* v49 = &v3[v47];
    vint8mf4_t v50 = __riscv_vle8_v_i8mf4(v49, v48);
    vint16mf2_t v51 = __riscv_vsext_vf2_i16mf2(v50, v48);
    size_t v52 = i46 * v6;
    size_t v53 = v52 + v11;
    size_t v54 = v52 + v12;
    size_t v55 = v52 + v13;
    size_t v56 = v52 + v14;
    int8_t v57 = v5[v52];
    int16_t v58 = (int16_t) v57;
    vint32m1_t v59 = v41;
    vint32m1_t v60 = __riscv_vwmacc_vx_i32m1(v59, v58, v51, v48);
    int8_t v61 = v5[v53];
    int16_t v62 = (int16_t) v61;
    vint32m1_t v63 = v42;
    vint32m1_t v64 = __riscv_vwmacc_vx_i32m1(v63, v62, v51, v48);
    int8_t v65 = v5[v54];
    int16_t v66 = (int16_t) v65;
    vint32m1_t v67 = v43;
    vint32m1_t v68 = __riscv_vwmacc_vx_i32m1(v67, v66, v51, v48);
    int8_t v69 = v5[v55];
    int16_t v70 = (int16_t) v69;
    vint32m1_t v71 = v44;
    vint32m1_t v72 = __riscv_vwmacc_vx_i32m1(v71, v70, v51, v48);
    int8_t v73 = v5[v56];
    int16_t v74 = (int16_t) v73;
    vint32m1_t v75 = v45;
    vint32m1_t v76 = __riscv_vwmacc_vx_i32m1(v75, v74, v51, v48);
    v41 = v60;
    v42 = v64;
    v43 = v68;
    v44 = v72;
    v45 = v76;
  }
  vint32m1_t v77 = v41;
  vint32m1_t v78 = v42;
  vint32m1_t v79 = v43;
  vint32m1_t v80 = v44;
  vint32m1_t v81 = v45;
  int32_t* v82 = &v8[v10];
  __riscv_vse32_v_i32m1(v82, v77, v13);
  int32_t* v83 = &v8[v9];
  __riscv_vse32_v_i32m1(v83, v78, v13);
  size_t v84 = v12 * v9;
  int32_t* v85 = &v8[v84];
  __riscv_vse32_v_i32m1(v85, v79, v13);
  size_t v86 = v13 * v9;
  int32_t* v87 = &v8[v86];
  __riscv_vse32_v_i32m1(v87, v80, v13);
  size_t v88 = v14 * v9;
  int32_t* v89 = &v8[v88];
  __riscv_vse32_v_i32m1(v89, v81, v13);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_3x6_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 3
//   NR         : 6
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_3x6_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v13);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v13);
  vint32m1_t v25 = __riscv_vmv_v_x_i32m1(v20, v13);
  vint32m1_t v26 = __riscv_vmv_v_x_i32m1(v20, v13);
  vint32m1_t v27;
  v27 = v21;
  vint32m1_t v28;
  v28 = v22;
  vint32m1_t v29;
  v29 = v23;
  vint32m1_t v30;
  v30 = v24;
  vint32m1_t v31;
  v31 = v25;
  vint32m1_t v32;
  v32 = v26;
  for (size_t i33 = v10; i33 < v1; i33 += v11) {
    size_t v34 = i33 * v4;
    size_t v35 = 3;
    int8_t* v36 = &v3[v34];
    vint8mf4_t v37 = __riscv_vle8_v_i8mf4(v36, v35);
    vint16mf2_t v38 = __riscv_vsext_vf2_i16mf2(v37, v35);
    size_t v39 = i33 * v6;
    size_t v40 = v39 + v11;
    size_t v41 = v39 + v12;
    size_t v42 = v39 + v13;
    size_t v43 = v39 + v14;
    size_t v44 = v39 + v15;
    int8_t v45 = v5[v39];
    int16_t v46 = (int16_t) v45;
    vint32m1_t v47 = v27;
    vint32m1_t v48 = __riscv_vwmacc_vx_i32m1(v47, v46, v38, v35);
    int8_t v49 = v5[v40];
    int16_t v50 = (int16_t) v49;
    vint32m1_t v51 = v28;
    vint32m1_t v52 = __riscv_vwmacc_vx_i32m1(v51, v50, v38, v35);
    int8_t v53 = v5[v41];
    int16_t v54 = (int16_t) v53;
    vint32m1_t v55 = v29;
    vint32m1_t v56 = __riscv_vwmacc_vx_i32m1(v55, v54, v38, v35);
    int8_t v57 = v5[v42];
    int16_t v58 = (int16_t) v57;
    vint32m1_t v59 = v30;
    vint32m1_t v60 = __riscv_vwmacc_vx_i32m1(v59, v58, v38, v35);
    int8_t v61 = v5[v43];
    int16_t v62 = (int16_t) v61;
    vint32m1_t v63 = v31;
    vint32m1_t v64 = __riscv_vwmacc_vx_i32m1(v63, v62, v38, v35);
    int8_t v65 = v5[v44];
    int16_t v66 = (int16_t) v65;
    vint32m1_t v67 = v32;
    vint32m1_t v68 = __riscv_vwmacc_vx_i32m1(v67, v66, v38, v35);
    v27 = v48;
    v28 = v52;
    v29 = v56;
    v30 = v60;
    v31 = v64;
    v32 = v68;
  }
  vint32m1_t v69 = v27;
  vint32m1_t v70 = v28;
  vint32m1_t v71 = v29;
  vint32m1_t v72 = v30;
  vint32m1_t v73 = v31;
  vint32m1_t v74 = v32;
  int32_t* v75 = &v8[v10];
  __riscv_vse32_v_i32m1(v75, v69, v13);
  int32_t* v76 = &v8[v9];
  __riscv_vse32_v_i32m1(v76, v70, v13);
  size_t v77 = v12 * v9;
  int32_t* v78 = &v8[v77];
  __riscv_vse32_v_i32m1(v78, v71, v13);
  size_t v79 = v13 * v9;
  int32_t* v80 = &v8[v79];
  __riscv_vse32_v_i32m1(v80, v72, v13);
  size_t v81 = v14 * v9;
  int32_t* v82 = &v8[v81];
  __riscv_vse32_v_i32m1(v82, v73, v13);
  size_t v83 = v15 * v9;
  int32_t* v84 = &v8[v83];
  __riscv_vse32_v_i32m1(v84, v74, v13);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_3x6_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 3
//   NR         : 6
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_3x6_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v13);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v13);
  size_t v37 = v14 * v9;
  size_t v38 = v10 + v37;
  int32_t* v39 = &v8[v38];
  vint32m1_t v40 = __riscv_vle32_v_i32m1(v39, v13);
  size_t v41 = v15 * v9;
  size_t v42 = v10 + v41;
  int32_t* v43 = &v8[v42];
  vint32m1_t v44 = __riscv_vle32_v_i32m1(v43, v13);
  vint32m1_t v45;
  v45 = v24;
  vint32m1_t v46;
  v46 = v28;
  vint32m1_t v47;
  v47 = v32;
  vint32m1_t v48;
  v48 = v36;
  vint32m1_t v49;
  v49 = v40;
  vint32m1_t v50;
  v50 = v44;
  for (size_t i51 = v10; i51 < v1; i51 += v11) {
    size_t v52 = i51 * v4;
    size_t v53 = 3;
    int8_t* v54 = &v3[v52];
    vint8mf4_t v55 = __riscv_vle8_v_i8mf4(v54, v53);
    vint16mf2_t v56 = __riscv_vsext_vf2_i16mf2(v55, v53);
    size_t v57 = i51 * v6;
    size_t v58 = v57 + v11;
    size_t v59 = v57 + v12;
    size_t v60 = v57 + v13;
    size_t v61 = v57 + v14;
    size_t v62 = v57 + v15;
    int8_t v63 = v5[v57];
    int16_t v64 = (int16_t) v63;
    vint32m1_t v65 = v45;
    vint32m1_t v66 = __riscv_vwmacc_vx_i32m1(v65, v64, v56, v53);
    int8_t v67 = v5[v58];
    int16_t v68 = (int16_t) v67;
    vint32m1_t v69 = v46;
    vint32m1_t v70 = __riscv_vwmacc_vx_i32m1(v69, v68, v56, v53);
    int8_t v71 = v5[v59];
    int16_t v72 = (int16_t) v71;
    vint32m1_t v73 = v47;
    vint32m1_t v74 = __riscv_vwmacc_vx_i32m1(v73, v72, v56, v53);
    int8_t v75 = v5[v60];
    int16_t v76 = (int16_t) v75;
    vint32m1_t v77 = v48;
    vint32m1_t v78 = __riscv_vwmacc_vx_i32m1(v77, v76, v56, v53);
    int8_t v79 = v5[v61];
    int16_t v80 = (int16_t) v79;
    vint32m1_t v81 = v49;
    vint32m1_t v82 = __riscv_vwmacc_vx_i32m1(v81, v80, v56, v53);
    int8_t v83 = v5[v62];
    int16_t v84 = (int16_t) v83;
    vint32m1_t v85 = v50;
    vint32m1_t v86 = __riscv_vwmacc_vx_i32m1(v85, v84, v56, v53);
    v45 = v66;
    v46 = v70;
    v47 = v74;
    v48 = v78;
    v49 = v82;
    v50 = v86;
  }
  vint32m1_t v87 = v45;
  vint32m1_t v88 = v46;
  vint32m1_t v89 = v47;
  vint32m1_t v90 = v48;
  vint32m1_t v91 = v49;
  vint32m1_t v92 = v50;
  int32_t* v93 = &v8[v10];
  __riscv_vse32_v_i32m1(v93, v87, v13);
  int32_t* v94 = &v8[v9];
  __riscv_vse32_v_i32m1(v94, v88, v13);
  size_t v95 = v12 * v9;
  int32_t* v96 = &v8[v95];
  __riscv_vse32_v_i32m1(v96, v89, v13);
  size_t v97 = v13 * v9;
  int32_t* v98 = &v8[v97];
  __riscv_vse32_v_i32m1(v98, v90, v13);
  size_t v99 = v14 * v9;
  int32_t* v100 = &v8[v99];
  __riscv_vse32_v_i32m1(v100, v91, v13);
  size_t v101 = v15 * v9;
  int32_t* v102 = &v8[v101];
  __riscv_vse32_v_i32m1(v102, v92, v13);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_3x7_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 3
//   NR         : 7
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_3x7_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v13);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v13);
  vint32m1_t v25 = __riscv_vmv_v_x_i32m1(v20, v13);
  vint32m1_t v26 = __riscv_vmv_v_x_i32m1(v20, v13);
  vint32m1_t v27 = __riscv_vmv_v_x_i32m1(v20, v13);
  vint32m1_t v28;
  v28 = v21;
  vint32m1_t v29;
  v29 = v22;
  vint32m1_t v30;
  v30 = v23;
  vint32m1_t v31;
  v31 = v24;
  vint32m1_t v32;
  v32 = v25;
  vint32m1_t v33;
  v33 = v26;
  vint32m1_t v34;
  v34 = v27;
  for (size_t i35 = v10; i35 < v1; i35 += v11) {
    size_t v36 = i35 * v4;
    size_t v37 = 3;
    int8_t* v38 = &v3[v36];
    vint8mf4_t v39 = __riscv_vle8_v_i8mf4(v38, v37);
    vint16mf2_t v40 = __riscv_vsext_vf2_i16mf2(v39, v37);
    size_t v41 = i35 * v6;
    size_t v42 = v41 + v11;
    size_t v43 = v41 + v12;
    size_t v44 = v41 + v13;
    size_t v45 = v41 + v14;
    size_t v46 = v41 + v15;
    size_t v47 = v41 + v16;
    int8_t v48 = v5[v41];
    int16_t v49 = (int16_t) v48;
    vint32m1_t v50 = v28;
    vint32m1_t v51 = __riscv_vwmacc_vx_i32m1(v50, v49, v40, v37);
    int8_t v52 = v5[v42];
    int16_t v53 = (int16_t) v52;
    vint32m1_t v54 = v29;
    vint32m1_t v55 = __riscv_vwmacc_vx_i32m1(v54, v53, v40, v37);
    int8_t v56 = v5[v43];
    int16_t v57 = (int16_t) v56;
    vint32m1_t v58 = v30;
    vint32m1_t v59 = __riscv_vwmacc_vx_i32m1(v58, v57, v40, v37);
    int8_t v60 = v5[v44];
    int16_t v61 = (int16_t) v60;
    vint32m1_t v62 = v31;
    vint32m1_t v63 = __riscv_vwmacc_vx_i32m1(v62, v61, v40, v37);
    int8_t v64 = v5[v45];
    int16_t v65 = (int16_t) v64;
    vint32m1_t v66 = v32;
    vint32m1_t v67 = __riscv_vwmacc_vx_i32m1(v66, v65, v40, v37);
    int8_t v68 = v5[v46];
    int16_t v69 = (int16_t) v68;
    vint32m1_t v70 = v33;
    vint32m1_t v71 = __riscv_vwmacc_vx_i32m1(v70, v69, v40, v37);
    int8_t v72 = v5[v47];
    int16_t v73 = (int16_t) v72;
    vint32m1_t v74 = v34;
    vint32m1_t v75 = __riscv_vwmacc_vx_i32m1(v74, v73, v40, v37);
    v28 = v51;
    v29 = v55;
    v30 = v59;
    v31 = v63;
    v32 = v67;
    v33 = v71;
    v34 = v75;
  }
  vint32m1_t v76 = v28;
  vint32m1_t v77 = v29;
  vint32m1_t v78 = v30;
  vint32m1_t v79 = v31;
  vint32m1_t v80 = v32;
  vint32m1_t v81 = v33;
  vint32m1_t v82 = v34;
  int32_t* v83 = &v8[v10];
  __riscv_vse32_v_i32m1(v83, v76, v13);
  int32_t* v84 = &v8[v9];
  __riscv_vse32_v_i32m1(v84, v77, v13);
  size_t v85 = v12 * v9;
  int32_t* v86 = &v8[v85];
  __riscv_vse32_v_i32m1(v86, v78, v13);
  size_t v87 = v13 * v9;
  int32_t* v88 = &v8[v87];
  __riscv_vse32_v_i32m1(v88, v79, v13);
  size_t v89 = v14 * v9;
  int32_t* v90 = &v8[v89];
  __riscv_vse32_v_i32m1(v90, v80, v13);
  size_t v91 = v15 * v9;
  int32_t* v92 = &v8[v91];
  __riscv_vse32_v_i32m1(v92, v81, v13);
  size_t v93 = v16 * v9;
  int32_t* v94 = &v8[v93];
  __riscv_vse32_v_i32m1(v94, v82, v13);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_3x7_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 3
//   NR         : 7
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_3x7_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v13);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v13);
  size_t v37 = v14 * v9;
  size_t v38 = v10 + v37;
  int32_t* v39 = &v8[v38];
  vint32m1_t v40 = __riscv_vle32_v_i32m1(v39, v13);
  size_t v41 = v15 * v9;
  size_t v42 = v10 + v41;
  int32_t* v43 = &v8[v42];
  vint32m1_t v44 = __riscv_vle32_v_i32m1(v43, v13);
  size_t v45 = v16 * v9;
  size_t v46 = v10 + v45;
  int32_t* v47 = &v8[v46];
  vint32m1_t v48 = __riscv_vle32_v_i32m1(v47, v13);
  vint32m1_t v49;
  v49 = v24;
  vint32m1_t v50;
  v50 = v28;
  vint32m1_t v51;
  v51 = v32;
  vint32m1_t v52;
  v52 = v36;
  vint32m1_t v53;
  v53 = v40;
  vint32m1_t v54;
  v54 = v44;
  vint32m1_t v55;
  v55 = v48;
  for (size_t i56 = v10; i56 < v1; i56 += v11) {
    size_t v57 = i56 * v4;
    size_t v58 = 3;
    int8_t* v59 = &v3[v57];
    vint8mf4_t v60 = __riscv_vle8_v_i8mf4(v59, v58);
    vint16mf2_t v61 = __riscv_vsext_vf2_i16mf2(v60, v58);
    size_t v62 = i56 * v6;
    size_t v63 = v62 + v11;
    size_t v64 = v62 + v12;
    size_t v65 = v62 + v13;
    size_t v66 = v62 + v14;
    size_t v67 = v62 + v15;
    size_t v68 = v62 + v16;
    int8_t v69 = v5[v62];
    int16_t v70 = (int16_t) v69;
    vint32m1_t v71 = v49;
    vint32m1_t v72 = __riscv_vwmacc_vx_i32m1(v71, v70, v61, v58);
    int8_t v73 = v5[v63];
    int16_t v74 = (int16_t) v73;
    vint32m1_t v75 = v50;
    vint32m1_t v76 = __riscv_vwmacc_vx_i32m1(v75, v74, v61, v58);
    int8_t v77 = v5[v64];
    int16_t v78 = (int16_t) v77;
    vint32m1_t v79 = v51;
    vint32m1_t v80 = __riscv_vwmacc_vx_i32m1(v79, v78, v61, v58);
    int8_t v81 = v5[v65];
    int16_t v82 = (int16_t) v81;
    vint32m1_t v83 = v52;
    vint32m1_t v84 = __riscv_vwmacc_vx_i32m1(v83, v82, v61, v58);
    int8_t v85 = v5[v66];
    int16_t v86 = (int16_t) v85;
    vint32m1_t v87 = v53;
    vint32m1_t v88 = __riscv_vwmacc_vx_i32m1(v87, v86, v61, v58);
    int8_t v89 = v5[v67];
    int16_t v90 = (int16_t) v89;
    vint32m1_t v91 = v54;
    vint32m1_t v92 = __riscv_vwmacc_vx_i32m1(v91, v90, v61, v58);
    int8_t v93 = v5[v68];
    int16_t v94 = (int16_t) v93;
    vint32m1_t v95 = v55;
    vint32m1_t v96 = __riscv_vwmacc_vx_i32m1(v95, v94, v61, v58);
    v49 = v72;
    v50 = v76;
    v51 = v80;
    v52 = v84;
    v53 = v88;
    v54 = v92;
    v55 = v96;
  }
  vint32m1_t v97 = v49;
  vint32m1_t v98 = v50;
  vint32m1_t v99 = v51;
  vint32m1_t v100 = v52;
  vint32m1_t v101 = v53;
  vint32m1_t v102 = v54;
  vint32m1_t v103 = v55;
  int32_t* v104 = &v8[v10];
  __riscv_vse32_v_i32m1(v104, v97, v13);
  int32_t* v105 = &v8[v9];
  __riscv_vse32_v_i32m1(v105, v98, v13);
  size_t v106 = v12 * v9;
  int32_t* v107 = &v8[v106];
  __riscv_vse32_v_i32m1(v107, v99, v13);
  size_t v108 = v13 * v9;
  int32_t* v109 = &v8[v108];
  __riscv_vse32_v_i32m1(v109, v100, v13);
  size_t v110 = v14 * v9;
  int32_t* v111 = &v8[v110];
  __riscv_vse32_v_i32m1(v111, v101, v13);
  size_t v112 = v15 * v9;
  int32_t* v113 = &v8[v112];
  __riscv_vse32_v_i32m1(v113, v102, v13);
  size_t v114 = v16 * v9;
  int32_t* v115 = &v8[v114];
  __riscv_vse32_v_i32m1(v115, v103, v13);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_3x8_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 3
//   NR         : 8
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_3x8_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v13);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v13);
  vint32m1_t v25 = __riscv_vmv_v_x_i32m1(v20, v13);
  vint32m1_t v26 = __riscv_vmv_v_x_i32m1(v20, v13);
  vint32m1_t v27 = __riscv_vmv_v_x_i32m1(v20, v13);
  vint32m1_t v28 = __riscv_vmv_v_x_i32m1(v20, v13);
  vint32m1_t v29;
  v29 = v21;
  vint32m1_t v30;
  v30 = v22;
  vint32m1_t v31;
  v31 = v23;
  vint32m1_t v32;
  v32 = v24;
  vint32m1_t v33;
  v33 = v25;
  vint32m1_t v34;
  v34 = v26;
  vint32m1_t v35;
  v35 = v27;
  vint32m1_t v36;
  v36 = v28;
  for (size_t i37 = v10; i37 < v1; i37 += v11) {
    size_t v38 = i37 * v4;
    size_t v39 = 3;
    int8_t* v40 = &v3[v38];
    vint8mf4_t v41 = __riscv_vle8_v_i8mf4(v40, v39);
    vint16mf2_t v42 = __riscv_vsext_vf2_i16mf2(v41, v39);
    size_t v43 = i37 * v6;
    size_t v44 = v43 + v11;
    size_t v45 = v43 + v12;
    size_t v46 = v43 + v13;
    size_t v47 = v43 + v14;
    size_t v48 = v43 + v15;
    size_t v49 = v43 + v16;
    size_t v50 = v43 + v17;
    int8_t v51 = v5[v43];
    int16_t v52 = (int16_t) v51;
    vint32m1_t v53 = v29;
    vint32m1_t v54 = __riscv_vwmacc_vx_i32m1(v53, v52, v42, v39);
    int8_t v55 = v5[v44];
    int16_t v56 = (int16_t) v55;
    vint32m1_t v57 = v30;
    vint32m1_t v58 = __riscv_vwmacc_vx_i32m1(v57, v56, v42, v39);
    int8_t v59 = v5[v45];
    int16_t v60 = (int16_t) v59;
    vint32m1_t v61 = v31;
    vint32m1_t v62 = __riscv_vwmacc_vx_i32m1(v61, v60, v42, v39);
    int8_t v63 = v5[v46];
    int16_t v64 = (int16_t) v63;
    vint32m1_t v65 = v32;
    vint32m1_t v66 = __riscv_vwmacc_vx_i32m1(v65, v64, v42, v39);
    int8_t v67 = v5[v47];
    int16_t v68 = (int16_t) v67;
    vint32m1_t v69 = v33;
    vint32m1_t v70 = __riscv_vwmacc_vx_i32m1(v69, v68, v42, v39);
    int8_t v71 = v5[v48];
    int16_t v72 = (int16_t) v71;
    vint32m1_t v73 = v34;
    vint32m1_t v74 = __riscv_vwmacc_vx_i32m1(v73, v72, v42, v39);
    int8_t v75 = v5[v49];
    int16_t v76 = (int16_t) v75;
    vint32m1_t v77 = v35;
    vint32m1_t v78 = __riscv_vwmacc_vx_i32m1(v77, v76, v42, v39);
    int8_t v79 = v5[v50];
    int16_t v80 = (int16_t) v79;
    vint32m1_t v81 = v36;
    vint32m1_t v82 = __riscv_vwmacc_vx_i32m1(v81, v80, v42, v39);
    v29 = v54;
    v30 = v58;
    v31 = v62;
    v32 = v66;
    v33 = v70;
    v34 = v74;
    v35 = v78;
    v36 = v82;
  }
  vint32m1_t v83 = v29;
  vint32m1_t v84 = v30;
  vint32m1_t v85 = v31;
  vint32m1_t v86 = v32;
  vint32m1_t v87 = v33;
  vint32m1_t v88 = v34;
  vint32m1_t v89 = v35;
  vint32m1_t v90 = v36;
  int32_t* v91 = &v8[v10];
  __riscv_vse32_v_i32m1(v91, v83, v13);
  int32_t* v92 = &v8[v9];
  __riscv_vse32_v_i32m1(v92, v84, v13);
  size_t v93 = v12 * v9;
  int32_t* v94 = &v8[v93];
  __riscv_vse32_v_i32m1(v94, v85, v13);
  size_t v95 = v13 * v9;
  int32_t* v96 = &v8[v95];
  __riscv_vse32_v_i32m1(v96, v86, v13);
  size_t v97 = v14 * v9;
  int32_t* v98 = &v8[v97];
  __riscv_vse32_v_i32m1(v98, v87, v13);
  size_t v99 = v15 * v9;
  int32_t* v100 = &v8[v99];
  __riscv_vse32_v_i32m1(v100, v88, v13);
  size_t v101 = v16 * v9;
  int32_t* v102 = &v8[v101];
  __riscv_vse32_v_i32m1(v102, v89, v13);
  size_t v103 = v17 * v9;
  int32_t* v104 = &v8[v103];
  __riscv_vse32_v_i32m1(v104, v90, v13);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_3x8_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 3
//   NR         : 8
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_3x8_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v13);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v13);
  size_t v37 = v14 * v9;
  size_t v38 = v10 + v37;
  int32_t* v39 = &v8[v38];
  vint32m1_t v40 = __riscv_vle32_v_i32m1(v39, v13);
  size_t v41 = v15 * v9;
  size_t v42 = v10 + v41;
  int32_t* v43 = &v8[v42];
  vint32m1_t v44 = __riscv_vle32_v_i32m1(v43, v13);
  size_t v45 = v16 * v9;
  size_t v46 = v10 + v45;
  int32_t* v47 = &v8[v46];
  vint32m1_t v48 = __riscv_vle32_v_i32m1(v47, v13);
  size_t v49 = v17 * v9;
  size_t v50 = v10 + v49;
  int32_t* v51 = &v8[v50];
  vint32m1_t v52 = __riscv_vle32_v_i32m1(v51, v13);
  vint32m1_t v53;
  v53 = v24;
  vint32m1_t v54;
  v54 = v28;
  vint32m1_t v55;
  v55 = v32;
  vint32m1_t v56;
  v56 = v36;
  vint32m1_t v57;
  v57 = v40;
  vint32m1_t v58;
  v58 = v44;
  vint32m1_t v59;
  v59 = v48;
  vint32m1_t v60;
  v60 = v52;
  for (size_t i61 = v10; i61 < v1; i61 += v11) {
    size_t v62 = i61 * v4;
    size_t v63 = 3;
    int8_t* v64 = &v3[v62];
    vint8mf4_t v65 = __riscv_vle8_v_i8mf4(v64, v63);
    vint16mf2_t v66 = __riscv_vsext_vf2_i16mf2(v65, v63);
    size_t v67 = i61 * v6;
    size_t v68 = v67 + v11;
    size_t v69 = v67 + v12;
    size_t v70 = v67 + v13;
    size_t v71 = v67 + v14;
    size_t v72 = v67 + v15;
    size_t v73 = v67 + v16;
    size_t v74 = v67 + v17;
    int8_t v75 = v5[v67];
    int16_t v76 = (int16_t) v75;
    vint32m1_t v77 = v53;
    vint32m1_t v78 = __riscv_vwmacc_vx_i32m1(v77, v76, v66, v63);
    int8_t v79 = v5[v68];
    int16_t v80 = (int16_t) v79;
    vint32m1_t v81 = v54;
    vint32m1_t v82 = __riscv_vwmacc_vx_i32m1(v81, v80, v66, v63);
    int8_t v83 = v5[v69];
    int16_t v84 = (int16_t) v83;
    vint32m1_t v85 = v55;
    vint32m1_t v86 = __riscv_vwmacc_vx_i32m1(v85, v84, v66, v63);
    int8_t v87 = v5[v70];
    int16_t v88 = (int16_t) v87;
    vint32m1_t v89 = v56;
    vint32m1_t v90 = __riscv_vwmacc_vx_i32m1(v89, v88, v66, v63);
    int8_t v91 = v5[v71];
    int16_t v92 = (int16_t) v91;
    vint32m1_t v93 = v57;
    vint32m1_t v94 = __riscv_vwmacc_vx_i32m1(v93, v92, v66, v63);
    int8_t v95 = v5[v72];
    int16_t v96 = (int16_t) v95;
    vint32m1_t v97 = v58;
    vint32m1_t v98 = __riscv_vwmacc_vx_i32m1(v97, v96, v66, v63);
    int8_t v99 = v5[v73];
    int16_t v100 = (int16_t) v99;
    vint32m1_t v101 = v59;
    vint32m1_t v102 = __riscv_vwmacc_vx_i32m1(v101, v100, v66, v63);
    int8_t v103 = v5[v74];
    int16_t v104 = (int16_t) v103;
    vint32m1_t v105 = v60;
    vint32m1_t v106 = __riscv_vwmacc_vx_i32m1(v105, v104, v66, v63);
    v53 = v78;
    v54 = v82;
    v55 = v86;
    v56 = v90;
    v57 = v94;
    v58 = v98;
    v59 = v102;
    v60 = v106;
  }
  vint32m1_t v107 = v53;
  vint32m1_t v108 = v54;
  vint32m1_t v109 = v55;
  vint32m1_t v110 = v56;
  vint32m1_t v111 = v57;
  vint32m1_t v112 = v58;
  vint32m1_t v113 = v59;
  vint32m1_t v114 = v60;
  int32_t* v115 = &v8[v10];
  __riscv_vse32_v_i32m1(v115, v107, v13);
  int32_t* v116 = &v8[v9];
  __riscv_vse32_v_i32m1(v116, v108, v13);
  size_t v117 = v12 * v9;
  int32_t* v118 = &v8[v117];
  __riscv_vse32_v_i32m1(v118, v109, v13);
  size_t v119 = v13 * v9;
  int32_t* v120 = &v8[v119];
  __riscv_vse32_v_i32m1(v120, v110, v13);
  size_t v121 = v14 * v9;
  int32_t* v122 = &v8[v121];
  __riscv_vse32_v_i32m1(v122, v111, v13);
  size_t v123 = v15 * v9;
  int32_t* v124 = &v8[v123];
  __riscv_vse32_v_i32m1(v124, v112, v13);
  size_t v125 = v16 * v9;
  int32_t* v126 = &v8[v125];
  __riscv_vse32_v_i32m1(v126, v113, v13);
  size_t v127 = v17 * v9;
  int32_t* v128 = &v8[v127];
  __riscv_vse32_v_i32m1(v128, v114, v13);
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
//   Generated   : 2026-06-29 11:41:10
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
//   Generated   : 2026-06-29 11:41:10
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
//   Generated   : 2026-06-29 11:41:10
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
//   Generated   : 2026-06-29 11:41:10
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



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_4x3_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 4
//   NR         : 3
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_4x3_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v14);
  vint32m1_t v24;
  v24 = v21;
  vint32m1_t v25;
  v25 = v22;
  vint32m1_t v26;
  v26 = v23;
  for (size_t i27 = v10; i27 < v1; i27 += v11) {
    size_t v28 = i27 * v4;
    size_t v29 = 4;
    int8_t* v30 = &v3[v28];
    vint8mf4_t v31 = __riscv_vle8_v_i8mf4(v30, v29);
    vint16mf2_t v32 = __riscv_vsext_vf2_i16mf2(v31, v29);
    size_t v33 = i27 * v6;
    size_t v34 = v33 + v11;
    size_t v35 = v33 + v12;
    int8_t v36 = v5[v33];
    int16_t v37 = (int16_t) v36;
    vint32m1_t v38 = v24;
    vint32m1_t v39 = __riscv_vwmacc_vx_i32m1(v38, v37, v32, v29);
    int8_t v40 = v5[v34];
    int16_t v41 = (int16_t) v40;
    vint32m1_t v42 = v25;
    vint32m1_t v43 = __riscv_vwmacc_vx_i32m1(v42, v41, v32, v29);
    int8_t v44 = v5[v35];
    int16_t v45 = (int16_t) v44;
    vint32m1_t v46 = v26;
    vint32m1_t v47 = __riscv_vwmacc_vx_i32m1(v46, v45, v32, v29);
    v24 = v39;
    v25 = v43;
    v26 = v47;
  }
  vint32m1_t v48 = v24;
  vint32m1_t v49 = v25;
  vint32m1_t v50 = v26;
  int32_t* v51 = &v8[v10];
  __riscv_vse32_v_i32m1(v51, v48, v14);
  int32_t* v52 = &v8[v9];
  __riscv_vse32_v_i32m1(v52, v49, v14);
  size_t v53 = v12 * v9;
  int32_t* v54 = &v8[v53];
  __riscv_vse32_v_i32m1(v54, v50, v14);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_4x3_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 4
//   NR         : 3
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_4x3_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v14);
  vint32m1_t v33;
  v33 = v24;
  vint32m1_t v34;
  v34 = v28;
  vint32m1_t v35;
  v35 = v32;
  for (size_t i36 = v10; i36 < v1; i36 += v11) {
    size_t v37 = i36 * v4;
    size_t v38 = 4;
    int8_t* v39 = &v3[v37];
    vint8mf4_t v40 = __riscv_vle8_v_i8mf4(v39, v38);
    vint16mf2_t v41 = __riscv_vsext_vf2_i16mf2(v40, v38);
    size_t v42 = i36 * v6;
    size_t v43 = v42 + v11;
    size_t v44 = v42 + v12;
    int8_t v45 = v5[v42];
    int16_t v46 = (int16_t) v45;
    vint32m1_t v47 = v33;
    vint32m1_t v48 = __riscv_vwmacc_vx_i32m1(v47, v46, v41, v38);
    int8_t v49 = v5[v43];
    int16_t v50 = (int16_t) v49;
    vint32m1_t v51 = v34;
    vint32m1_t v52 = __riscv_vwmacc_vx_i32m1(v51, v50, v41, v38);
    int8_t v53 = v5[v44];
    int16_t v54 = (int16_t) v53;
    vint32m1_t v55 = v35;
    vint32m1_t v56 = __riscv_vwmacc_vx_i32m1(v55, v54, v41, v38);
    v33 = v48;
    v34 = v52;
    v35 = v56;
  }
  vint32m1_t v57 = v33;
  vint32m1_t v58 = v34;
  vint32m1_t v59 = v35;
  int32_t* v60 = &v8[v10];
  __riscv_vse32_v_i32m1(v60, v57, v14);
  int32_t* v61 = &v8[v9];
  __riscv_vse32_v_i32m1(v61, v58, v14);
  size_t v62 = v12 * v9;
  int32_t* v63 = &v8[v62];
  __riscv_vse32_v_i32m1(v63, v59, v14);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_4x4_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 4
//   NR         : 4
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_4x4_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v14);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v14);
  vint32m1_t v25;
  v25 = v21;
  vint32m1_t v26;
  v26 = v22;
  vint32m1_t v27;
  v27 = v23;
  vint32m1_t v28;
  v28 = v24;
  for (size_t i29 = v10; i29 < v1; i29 += v11) {
    size_t v30 = i29 * v4;
    size_t v31 = 4;
    int8_t* v32 = &v3[v30];
    vint8mf4_t v33 = __riscv_vle8_v_i8mf4(v32, v31);
    vint16mf2_t v34 = __riscv_vsext_vf2_i16mf2(v33, v31);
    size_t v35 = i29 * v6;
    size_t v36 = v35 + v11;
    size_t v37 = v35 + v12;
    size_t v38 = v35 + v13;
    int8_t v39 = v5[v35];
    int16_t v40 = (int16_t) v39;
    vint32m1_t v41 = v25;
    vint32m1_t v42 = __riscv_vwmacc_vx_i32m1(v41, v40, v34, v31);
    int8_t v43 = v5[v36];
    int16_t v44 = (int16_t) v43;
    vint32m1_t v45 = v26;
    vint32m1_t v46 = __riscv_vwmacc_vx_i32m1(v45, v44, v34, v31);
    int8_t v47 = v5[v37];
    int16_t v48 = (int16_t) v47;
    vint32m1_t v49 = v27;
    vint32m1_t v50 = __riscv_vwmacc_vx_i32m1(v49, v48, v34, v31);
    int8_t v51 = v5[v38];
    int16_t v52 = (int16_t) v51;
    vint32m1_t v53 = v28;
    vint32m1_t v54 = __riscv_vwmacc_vx_i32m1(v53, v52, v34, v31);
    v25 = v42;
    v26 = v46;
    v27 = v50;
    v28 = v54;
  }
  vint32m1_t v55 = v25;
  vint32m1_t v56 = v26;
  vint32m1_t v57 = v27;
  vint32m1_t v58 = v28;
  int32_t* v59 = &v8[v10];
  __riscv_vse32_v_i32m1(v59, v55, v14);
  int32_t* v60 = &v8[v9];
  __riscv_vse32_v_i32m1(v60, v56, v14);
  size_t v61 = v12 * v9;
  int32_t* v62 = &v8[v61];
  __riscv_vse32_v_i32m1(v62, v57, v14);
  size_t v63 = v13 * v9;
  int32_t* v64 = &v8[v63];
  __riscv_vse32_v_i32m1(v64, v58, v14);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_4x4_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 4
//   NR         : 4
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_4x4_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v14);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v14);
  vint32m1_t v37;
  v37 = v24;
  vint32m1_t v38;
  v38 = v28;
  vint32m1_t v39;
  v39 = v32;
  vint32m1_t v40;
  v40 = v36;
  for (size_t i41 = v10; i41 < v1; i41 += v11) {
    size_t v42 = i41 * v4;
    size_t v43 = 4;
    int8_t* v44 = &v3[v42];
    vint8mf4_t v45 = __riscv_vle8_v_i8mf4(v44, v43);
    vint16mf2_t v46 = __riscv_vsext_vf2_i16mf2(v45, v43);
    size_t v47 = i41 * v6;
    size_t v48 = v47 + v11;
    size_t v49 = v47 + v12;
    size_t v50 = v47 + v13;
    int8_t v51 = v5[v47];
    int16_t v52 = (int16_t) v51;
    vint32m1_t v53 = v37;
    vint32m1_t v54 = __riscv_vwmacc_vx_i32m1(v53, v52, v46, v43);
    int8_t v55 = v5[v48];
    int16_t v56 = (int16_t) v55;
    vint32m1_t v57 = v38;
    vint32m1_t v58 = __riscv_vwmacc_vx_i32m1(v57, v56, v46, v43);
    int8_t v59 = v5[v49];
    int16_t v60 = (int16_t) v59;
    vint32m1_t v61 = v39;
    vint32m1_t v62 = __riscv_vwmacc_vx_i32m1(v61, v60, v46, v43);
    int8_t v63 = v5[v50];
    int16_t v64 = (int16_t) v63;
    vint32m1_t v65 = v40;
    vint32m1_t v66 = __riscv_vwmacc_vx_i32m1(v65, v64, v46, v43);
    v37 = v54;
    v38 = v58;
    v39 = v62;
    v40 = v66;
  }
  vint32m1_t v67 = v37;
  vint32m1_t v68 = v38;
  vint32m1_t v69 = v39;
  vint32m1_t v70 = v40;
  int32_t* v71 = &v8[v10];
  __riscv_vse32_v_i32m1(v71, v67, v14);
  int32_t* v72 = &v8[v9];
  __riscv_vse32_v_i32m1(v72, v68, v14);
  size_t v73 = v12 * v9;
  int32_t* v74 = &v8[v73];
  __riscv_vse32_v_i32m1(v74, v69, v14);
  size_t v75 = v13 * v9;
  int32_t* v76 = &v8[v75];
  __riscv_vse32_v_i32m1(v76, v70, v14);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_4x5_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 4
//   NR         : 5
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_4x5_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v14);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v14);
  vint32m1_t v25 = __riscv_vmv_v_x_i32m1(v20, v14);
  vint32m1_t v26;
  v26 = v21;
  vint32m1_t v27;
  v27 = v22;
  vint32m1_t v28;
  v28 = v23;
  vint32m1_t v29;
  v29 = v24;
  vint32m1_t v30;
  v30 = v25;
  for (size_t i31 = v10; i31 < v1; i31 += v11) {
    size_t v32 = i31 * v4;
    size_t v33 = 4;
    int8_t* v34 = &v3[v32];
    vint8mf4_t v35 = __riscv_vle8_v_i8mf4(v34, v33);
    vint16mf2_t v36 = __riscv_vsext_vf2_i16mf2(v35, v33);
    size_t v37 = i31 * v6;
    size_t v38 = v37 + v11;
    size_t v39 = v37 + v12;
    size_t v40 = v37 + v13;
    size_t v41 = v37 + v14;
    int8_t v42 = v5[v37];
    int16_t v43 = (int16_t) v42;
    vint32m1_t v44 = v26;
    vint32m1_t v45 = __riscv_vwmacc_vx_i32m1(v44, v43, v36, v33);
    int8_t v46 = v5[v38];
    int16_t v47 = (int16_t) v46;
    vint32m1_t v48 = v27;
    vint32m1_t v49 = __riscv_vwmacc_vx_i32m1(v48, v47, v36, v33);
    int8_t v50 = v5[v39];
    int16_t v51 = (int16_t) v50;
    vint32m1_t v52 = v28;
    vint32m1_t v53 = __riscv_vwmacc_vx_i32m1(v52, v51, v36, v33);
    int8_t v54 = v5[v40];
    int16_t v55 = (int16_t) v54;
    vint32m1_t v56 = v29;
    vint32m1_t v57 = __riscv_vwmacc_vx_i32m1(v56, v55, v36, v33);
    int8_t v58 = v5[v41];
    int16_t v59 = (int16_t) v58;
    vint32m1_t v60 = v30;
    vint32m1_t v61 = __riscv_vwmacc_vx_i32m1(v60, v59, v36, v33);
    v26 = v45;
    v27 = v49;
    v28 = v53;
    v29 = v57;
    v30 = v61;
  }
  vint32m1_t v62 = v26;
  vint32m1_t v63 = v27;
  vint32m1_t v64 = v28;
  vint32m1_t v65 = v29;
  vint32m1_t v66 = v30;
  int32_t* v67 = &v8[v10];
  __riscv_vse32_v_i32m1(v67, v62, v14);
  int32_t* v68 = &v8[v9];
  __riscv_vse32_v_i32m1(v68, v63, v14);
  size_t v69 = v12 * v9;
  int32_t* v70 = &v8[v69];
  __riscv_vse32_v_i32m1(v70, v64, v14);
  size_t v71 = v13 * v9;
  int32_t* v72 = &v8[v71];
  __riscv_vse32_v_i32m1(v72, v65, v14);
  size_t v73 = v14 * v9;
  int32_t* v74 = &v8[v73];
  __riscv_vse32_v_i32m1(v74, v66, v14);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_4x5_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 4
//   NR         : 5
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_4x5_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v14);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v14);
  size_t v37 = v14 * v9;
  size_t v38 = v10 + v37;
  int32_t* v39 = &v8[v38];
  vint32m1_t v40 = __riscv_vle32_v_i32m1(v39, v14);
  vint32m1_t v41;
  v41 = v24;
  vint32m1_t v42;
  v42 = v28;
  vint32m1_t v43;
  v43 = v32;
  vint32m1_t v44;
  v44 = v36;
  vint32m1_t v45;
  v45 = v40;
  for (size_t i46 = v10; i46 < v1; i46 += v11) {
    size_t v47 = i46 * v4;
    size_t v48 = 4;
    int8_t* v49 = &v3[v47];
    vint8mf4_t v50 = __riscv_vle8_v_i8mf4(v49, v48);
    vint16mf2_t v51 = __riscv_vsext_vf2_i16mf2(v50, v48);
    size_t v52 = i46 * v6;
    size_t v53 = v52 + v11;
    size_t v54 = v52 + v12;
    size_t v55 = v52 + v13;
    size_t v56 = v52 + v14;
    int8_t v57 = v5[v52];
    int16_t v58 = (int16_t) v57;
    vint32m1_t v59 = v41;
    vint32m1_t v60 = __riscv_vwmacc_vx_i32m1(v59, v58, v51, v48);
    int8_t v61 = v5[v53];
    int16_t v62 = (int16_t) v61;
    vint32m1_t v63 = v42;
    vint32m1_t v64 = __riscv_vwmacc_vx_i32m1(v63, v62, v51, v48);
    int8_t v65 = v5[v54];
    int16_t v66 = (int16_t) v65;
    vint32m1_t v67 = v43;
    vint32m1_t v68 = __riscv_vwmacc_vx_i32m1(v67, v66, v51, v48);
    int8_t v69 = v5[v55];
    int16_t v70 = (int16_t) v69;
    vint32m1_t v71 = v44;
    vint32m1_t v72 = __riscv_vwmacc_vx_i32m1(v71, v70, v51, v48);
    int8_t v73 = v5[v56];
    int16_t v74 = (int16_t) v73;
    vint32m1_t v75 = v45;
    vint32m1_t v76 = __riscv_vwmacc_vx_i32m1(v75, v74, v51, v48);
    v41 = v60;
    v42 = v64;
    v43 = v68;
    v44 = v72;
    v45 = v76;
  }
  vint32m1_t v77 = v41;
  vint32m1_t v78 = v42;
  vint32m1_t v79 = v43;
  vint32m1_t v80 = v44;
  vint32m1_t v81 = v45;
  int32_t* v82 = &v8[v10];
  __riscv_vse32_v_i32m1(v82, v77, v14);
  int32_t* v83 = &v8[v9];
  __riscv_vse32_v_i32m1(v83, v78, v14);
  size_t v84 = v12 * v9;
  int32_t* v85 = &v8[v84];
  __riscv_vse32_v_i32m1(v85, v79, v14);
  size_t v86 = v13 * v9;
  int32_t* v87 = &v8[v86];
  __riscv_vse32_v_i32m1(v87, v80, v14);
  size_t v88 = v14 * v9;
  int32_t* v89 = &v8[v88];
  __riscv_vse32_v_i32m1(v89, v81, v14);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_4x6_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 4
//   NR         : 6
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_4x6_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v14);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v14);
  vint32m1_t v25 = __riscv_vmv_v_x_i32m1(v20, v14);
  vint32m1_t v26 = __riscv_vmv_v_x_i32m1(v20, v14);
  vint32m1_t v27;
  v27 = v21;
  vint32m1_t v28;
  v28 = v22;
  vint32m1_t v29;
  v29 = v23;
  vint32m1_t v30;
  v30 = v24;
  vint32m1_t v31;
  v31 = v25;
  vint32m1_t v32;
  v32 = v26;
  for (size_t i33 = v10; i33 < v1; i33 += v11) {
    size_t v34 = i33 * v4;
    size_t v35 = 4;
    int8_t* v36 = &v3[v34];
    vint8mf4_t v37 = __riscv_vle8_v_i8mf4(v36, v35);
    vint16mf2_t v38 = __riscv_vsext_vf2_i16mf2(v37, v35);
    size_t v39 = i33 * v6;
    size_t v40 = v39 + v11;
    size_t v41 = v39 + v12;
    size_t v42 = v39 + v13;
    size_t v43 = v39 + v14;
    size_t v44 = v39 + v15;
    int8_t v45 = v5[v39];
    int16_t v46 = (int16_t) v45;
    vint32m1_t v47 = v27;
    vint32m1_t v48 = __riscv_vwmacc_vx_i32m1(v47, v46, v38, v35);
    int8_t v49 = v5[v40];
    int16_t v50 = (int16_t) v49;
    vint32m1_t v51 = v28;
    vint32m1_t v52 = __riscv_vwmacc_vx_i32m1(v51, v50, v38, v35);
    int8_t v53 = v5[v41];
    int16_t v54 = (int16_t) v53;
    vint32m1_t v55 = v29;
    vint32m1_t v56 = __riscv_vwmacc_vx_i32m1(v55, v54, v38, v35);
    int8_t v57 = v5[v42];
    int16_t v58 = (int16_t) v57;
    vint32m1_t v59 = v30;
    vint32m1_t v60 = __riscv_vwmacc_vx_i32m1(v59, v58, v38, v35);
    int8_t v61 = v5[v43];
    int16_t v62 = (int16_t) v61;
    vint32m1_t v63 = v31;
    vint32m1_t v64 = __riscv_vwmacc_vx_i32m1(v63, v62, v38, v35);
    int8_t v65 = v5[v44];
    int16_t v66 = (int16_t) v65;
    vint32m1_t v67 = v32;
    vint32m1_t v68 = __riscv_vwmacc_vx_i32m1(v67, v66, v38, v35);
    v27 = v48;
    v28 = v52;
    v29 = v56;
    v30 = v60;
    v31 = v64;
    v32 = v68;
  }
  vint32m1_t v69 = v27;
  vint32m1_t v70 = v28;
  vint32m1_t v71 = v29;
  vint32m1_t v72 = v30;
  vint32m1_t v73 = v31;
  vint32m1_t v74 = v32;
  int32_t* v75 = &v8[v10];
  __riscv_vse32_v_i32m1(v75, v69, v14);
  int32_t* v76 = &v8[v9];
  __riscv_vse32_v_i32m1(v76, v70, v14);
  size_t v77 = v12 * v9;
  int32_t* v78 = &v8[v77];
  __riscv_vse32_v_i32m1(v78, v71, v14);
  size_t v79 = v13 * v9;
  int32_t* v80 = &v8[v79];
  __riscv_vse32_v_i32m1(v80, v72, v14);
  size_t v81 = v14 * v9;
  int32_t* v82 = &v8[v81];
  __riscv_vse32_v_i32m1(v82, v73, v14);
  size_t v83 = v15 * v9;
  int32_t* v84 = &v8[v83];
  __riscv_vse32_v_i32m1(v84, v74, v14);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_4x6_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 4
//   NR         : 6
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_4x6_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v14);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v14);
  size_t v37 = v14 * v9;
  size_t v38 = v10 + v37;
  int32_t* v39 = &v8[v38];
  vint32m1_t v40 = __riscv_vle32_v_i32m1(v39, v14);
  size_t v41 = v15 * v9;
  size_t v42 = v10 + v41;
  int32_t* v43 = &v8[v42];
  vint32m1_t v44 = __riscv_vle32_v_i32m1(v43, v14);
  vint32m1_t v45;
  v45 = v24;
  vint32m1_t v46;
  v46 = v28;
  vint32m1_t v47;
  v47 = v32;
  vint32m1_t v48;
  v48 = v36;
  vint32m1_t v49;
  v49 = v40;
  vint32m1_t v50;
  v50 = v44;
  for (size_t i51 = v10; i51 < v1; i51 += v11) {
    size_t v52 = i51 * v4;
    size_t v53 = 4;
    int8_t* v54 = &v3[v52];
    vint8mf4_t v55 = __riscv_vle8_v_i8mf4(v54, v53);
    vint16mf2_t v56 = __riscv_vsext_vf2_i16mf2(v55, v53);
    size_t v57 = i51 * v6;
    size_t v58 = v57 + v11;
    size_t v59 = v57 + v12;
    size_t v60 = v57 + v13;
    size_t v61 = v57 + v14;
    size_t v62 = v57 + v15;
    int8_t v63 = v5[v57];
    int16_t v64 = (int16_t) v63;
    vint32m1_t v65 = v45;
    vint32m1_t v66 = __riscv_vwmacc_vx_i32m1(v65, v64, v56, v53);
    int8_t v67 = v5[v58];
    int16_t v68 = (int16_t) v67;
    vint32m1_t v69 = v46;
    vint32m1_t v70 = __riscv_vwmacc_vx_i32m1(v69, v68, v56, v53);
    int8_t v71 = v5[v59];
    int16_t v72 = (int16_t) v71;
    vint32m1_t v73 = v47;
    vint32m1_t v74 = __riscv_vwmacc_vx_i32m1(v73, v72, v56, v53);
    int8_t v75 = v5[v60];
    int16_t v76 = (int16_t) v75;
    vint32m1_t v77 = v48;
    vint32m1_t v78 = __riscv_vwmacc_vx_i32m1(v77, v76, v56, v53);
    int8_t v79 = v5[v61];
    int16_t v80 = (int16_t) v79;
    vint32m1_t v81 = v49;
    vint32m1_t v82 = __riscv_vwmacc_vx_i32m1(v81, v80, v56, v53);
    int8_t v83 = v5[v62];
    int16_t v84 = (int16_t) v83;
    vint32m1_t v85 = v50;
    vint32m1_t v86 = __riscv_vwmacc_vx_i32m1(v85, v84, v56, v53);
    v45 = v66;
    v46 = v70;
    v47 = v74;
    v48 = v78;
    v49 = v82;
    v50 = v86;
  }
  vint32m1_t v87 = v45;
  vint32m1_t v88 = v46;
  vint32m1_t v89 = v47;
  vint32m1_t v90 = v48;
  vint32m1_t v91 = v49;
  vint32m1_t v92 = v50;
  int32_t* v93 = &v8[v10];
  __riscv_vse32_v_i32m1(v93, v87, v14);
  int32_t* v94 = &v8[v9];
  __riscv_vse32_v_i32m1(v94, v88, v14);
  size_t v95 = v12 * v9;
  int32_t* v96 = &v8[v95];
  __riscv_vse32_v_i32m1(v96, v89, v14);
  size_t v97 = v13 * v9;
  int32_t* v98 = &v8[v97];
  __riscv_vse32_v_i32m1(v98, v90, v14);
  size_t v99 = v14 * v9;
  int32_t* v100 = &v8[v99];
  __riscv_vse32_v_i32m1(v100, v91, v14);
  size_t v101 = v15 * v9;
  int32_t* v102 = &v8[v101];
  __riscv_vse32_v_i32m1(v102, v92, v14);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_4x7_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 4
//   NR         : 7
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_4x7_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v14);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v14);
  vint32m1_t v25 = __riscv_vmv_v_x_i32m1(v20, v14);
  vint32m1_t v26 = __riscv_vmv_v_x_i32m1(v20, v14);
  vint32m1_t v27 = __riscv_vmv_v_x_i32m1(v20, v14);
  vint32m1_t v28;
  v28 = v21;
  vint32m1_t v29;
  v29 = v22;
  vint32m1_t v30;
  v30 = v23;
  vint32m1_t v31;
  v31 = v24;
  vint32m1_t v32;
  v32 = v25;
  vint32m1_t v33;
  v33 = v26;
  vint32m1_t v34;
  v34 = v27;
  for (size_t i35 = v10; i35 < v1; i35 += v11) {
    size_t v36 = i35 * v4;
    size_t v37 = 4;
    int8_t* v38 = &v3[v36];
    vint8mf4_t v39 = __riscv_vle8_v_i8mf4(v38, v37);
    vint16mf2_t v40 = __riscv_vsext_vf2_i16mf2(v39, v37);
    size_t v41 = i35 * v6;
    size_t v42 = v41 + v11;
    size_t v43 = v41 + v12;
    size_t v44 = v41 + v13;
    size_t v45 = v41 + v14;
    size_t v46 = v41 + v15;
    size_t v47 = v41 + v16;
    int8_t v48 = v5[v41];
    int16_t v49 = (int16_t) v48;
    vint32m1_t v50 = v28;
    vint32m1_t v51 = __riscv_vwmacc_vx_i32m1(v50, v49, v40, v37);
    int8_t v52 = v5[v42];
    int16_t v53 = (int16_t) v52;
    vint32m1_t v54 = v29;
    vint32m1_t v55 = __riscv_vwmacc_vx_i32m1(v54, v53, v40, v37);
    int8_t v56 = v5[v43];
    int16_t v57 = (int16_t) v56;
    vint32m1_t v58 = v30;
    vint32m1_t v59 = __riscv_vwmacc_vx_i32m1(v58, v57, v40, v37);
    int8_t v60 = v5[v44];
    int16_t v61 = (int16_t) v60;
    vint32m1_t v62 = v31;
    vint32m1_t v63 = __riscv_vwmacc_vx_i32m1(v62, v61, v40, v37);
    int8_t v64 = v5[v45];
    int16_t v65 = (int16_t) v64;
    vint32m1_t v66 = v32;
    vint32m1_t v67 = __riscv_vwmacc_vx_i32m1(v66, v65, v40, v37);
    int8_t v68 = v5[v46];
    int16_t v69 = (int16_t) v68;
    vint32m1_t v70 = v33;
    vint32m1_t v71 = __riscv_vwmacc_vx_i32m1(v70, v69, v40, v37);
    int8_t v72 = v5[v47];
    int16_t v73 = (int16_t) v72;
    vint32m1_t v74 = v34;
    vint32m1_t v75 = __riscv_vwmacc_vx_i32m1(v74, v73, v40, v37);
    v28 = v51;
    v29 = v55;
    v30 = v59;
    v31 = v63;
    v32 = v67;
    v33 = v71;
    v34 = v75;
  }
  vint32m1_t v76 = v28;
  vint32m1_t v77 = v29;
  vint32m1_t v78 = v30;
  vint32m1_t v79 = v31;
  vint32m1_t v80 = v32;
  vint32m1_t v81 = v33;
  vint32m1_t v82 = v34;
  int32_t* v83 = &v8[v10];
  __riscv_vse32_v_i32m1(v83, v76, v14);
  int32_t* v84 = &v8[v9];
  __riscv_vse32_v_i32m1(v84, v77, v14);
  size_t v85 = v12 * v9;
  int32_t* v86 = &v8[v85];
  __riscv_vse32_v_i32m1(v86, v78, v14);
  size_t v87 = v13 * v9;
  int32_t* v88 = &v8[v87];
  __riscv_vse32_v_i32m1(v88, v79, v14);
  size_t v89 = v14 * v9;
  int32_t* v90 = &v8[v89];
  __riscv_vse32_v_i32m1(v90, v80, v14);
  size_t v91 = v15 * v9;
  int32_t* v92 = &v8[v91];
  __riscv_vse32_v_i32m1(v92, v81, v14);
  size_t v93 = v16 * v9;
  int32_t* v94 = &v8[v93];
  __riscv_vse32_v_i32m1(v94, v82, v14);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_4x7_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 4
//   NR         : 7
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_4x7_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v14);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v14);
  size_t v37 = v14 * v9;
  size_t v38 = v10 + v37;
  int32_t* v39 = &v8[v38];
  vint32m1_t v40 = __riscv_vle32_v_i32m1(v39, v14);
  size_t v41 = v15 * v9;
  size_t v42 = v10 + v41;
  int32_t* v43 = &v8[v42];
  vint32m1_t v44 = __riscv_vle32_v_i32m1(v43, v14);
  size_t v45 = v16 * v9;
  size_t v46 = v10 + v45;
  int32_t* v47 = &v8[v46];
  vint32m1_t v48 = __riscv_vle32_v_i32m1(v47, v14);
  vint32m1_t v49;
  v49 = v24;
  vint32m1_t v50;
  v50 = v28;
  vint32m1_t v51;
  v51 = v32;
  vint32m1_t v52;
  v52 = v36;
  vint32m1_t v53;
  v53 = v40;
  vint32m1_t v54;
  v54 = v44;
  vint32m1_t v55;
  v55 = v48;
  for (size_t i56 = v10; i56 < v1; i56 += v11) {
    size_t v57 = i56 * v4;
    size_t v58 = 4;
    int8_t* v59 = &v3[v57];
    vint8mf4_t v60 = __riscv_vle8_v_i8mf4(v59, v58);
    vint16mf2_t v61 = __riscv_vsext_vf2_i16mf2(v60, v58);
    size_t v62 = i56 * v6;
    size_t v63 = v62 + v11;
    size_t v64 = v62 + v12;
    size_t v65 = v62 + v13;
    size_t v66 = v62 + v14;
    size_t v67 = v62 + v15;
    size_t v68 = v62 + v16;
    int8_t v69 = v5[v62];
    int16_t v70 = (int16_t) v69;
    vint32m1_t v71 = v49;
    vint32m1_t v72 = __riscv_vwmacc_vx_i32m1(v71, v70, v61, v58);
    int8_t v73 = v5[v63];
    int16_t v74 = (int16_t) v73;
    vint32m1_t v75 = v50;
    vint32m1_t v76 = __riscv_vwmacc_vx_i32m1(v75, v74, v61, v58);
    int8_t v77 = v5[v64];
    int16_t v78 = (int16_t) v77;
    vint32m1_t v79 = v51;
    vint32m1_t v80 = __riscv_vwmacc_vx_i32m1(v79, v78, v61, v58);
    int8_t v81 = v5[v65];
    int16_t v82 = (int16_t) v81;
    vint32m1_t v83 = v52;
    vint32m1_t v84 = __riscv_vwmacc_vx_i32m1(v83, v82, v61, v58);
    int8_t v85 = v5[v66];
    int16_t v86 = (int16_t) v85;
    vint32m1_t v87 = v53;
    vint32m1_t v88 = __riscv_vwmacc_vx_i32m1(v87, v86, v61, v58);
    int8_t v89 = v5[v67];
    int16_t v90 = (int16_t) v89;
    vint32m1_t v91 = v54;
    vint32m1_t v92 = __riscv_vwmacc_vx_i32m1(v91, v90, v61, v58);
    int8_t v93 = v5[v68];
    int16_t v94 = (int16_t) v93;
    vint32m1_t v95 = v55;
    vint32m1_t v96 = __riscv_vwmacc_vx_i32m1(v95, v94, v61, v58);
    v49 = v72;
    v50 = v76;
    v51 = v80;
    v52 = v84;
    v53 = v88;
    v54 = v92;
    v55 = v96;
  }
  vint32m1_t v97 = v49;
  vint32m1_t v98 = v50;
  vint32m1_t v99 = v51;
  vint32m1_t v100 = v52;
  vint32m1_t v101 = v53;
  vint32m1_t v102 = v54;
  vint32m1_t v103 = v55;
  int32_t* v104 = &v8[v10];
  __riscv_vse32_v_i32m1(v104, v97, v14);
  int32_t* v105 = &v8[v9];
  __riscv_vse32_v_i32m1(v105, v98, v14);
  size_t v106 = v12 * v9;
  int32_t* v107 = &v8[v106];
  __riscv_vse32_v_i32m1(v107, v99, v14);
  size_t v108 = v13 * v9;
  int32_t* v109 = &v8[v108];
  __riscv_vse32_v_i32m1(v109, v100, v14);
  size_t v110 = v14 * v9;
  int32_t* v111 = &v8[v110];
  __riscv_vse32_v_i32m1(v111, v101, v14);
  size_t v112 = v15 * v9;
  int32_t* v113 = &v8[v112];
  __riscv_vse32_v_i32m1(v113, v102, v14);
  size_t v114 = v16 * v9;
  int32_t* v115 = &v8[v114];
  __riscv_vse32_v_i32m1(v115, v103, v14);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_4x8_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 4
//   NR         : 8
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_4x8_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v14);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v14);
  vint32m1_t v25 = __riscv_vmv_v_x_i32m1(v20, v14);
  vint32m1_t v26 = __riscv_vmv_v_x_i32m1(v20, v14);
  vint32m1_t v27 = __riscv_vmv_v_x_i32m1(v20, v14);
  vint32m1_t v28 = __riscv_vmv_v_x_i32m1(v20, v14);
  vint32m1_t v29;
  v29 = v21;
  vint32m1_t v30;
  v30 = v22;
  vint32m1_t v31;
  v31 = v23;
  vint32m1_t v32;
  v32 = v24;
  vint32m1_t v33;
  v33 = v25;
  vint32m1_t v34;
  v34 = v26;
  vint32m1_t v35;
  v35 = v27;
  vint32m1_t v36;
  v36 = v28;
  for (size_t i37 = v10; i37 < v1; i37 += v11) {
    size_t v38 = i37 * v4;
    size_t v39 = 4;
    int8_t* v40 = &v3[v38];
    vint8mf4_t v41 = __riscv_vle8_v_i8mf4(v40, v39);
    vint16mf2_t v42 = __riscv_vsext_vf2_i16mf2(v41, v39);
    size_t v43 = i37 * v6;
    size_t v44 = v43 + v11;
    size_t v45 = v43 + v12;
    size_t v46 = v43 + v13;
    size_t v47 = v43 + v14;
    size_t v48 = v43 + v15;
    size_t v49 = v43 + v16;
    size_t v50 = v43 + v17;
    int8_t v51 = v5[v43];
    int16_t v52 = (int16_t) v51;
    vint32m1_t v53 = v29;
    vint32m1_t v54 = __riscv_vwmacc_vx_i32m1(v53, v52, v42, v39);
    int8_t v55 = v5[v44];
    int16_t v56 = (int16_t) v55;
    vint32m1_t v57 = v30;
    vint32m1_t v58 = __riscv_vwmacc_vx_i32m1(v57, v56, v42, v39);
    int8_t v59 = v5[v45];
    int16_t v60 = (int16_t) v59;
    vint32m1_t v61 = v31;
    vint32m1_t v62 = __riscv_vwmacc_vx_i32m1(v61, v60, v42, v39);
    int8_t v63 = v5[v46];
    int16_t v64 = (int16_t) v63;
    vint32m1_t v65 = v32;
    vint32m1_t v66 = __riscv_vwmacc_vx_i32m1(v65, v64, v42, v39);
    int8_t v67 = v5[v47];
    int16_t v68 = (int16_t) v67;
    vint32m1_t v69 = v33;
    vint32m1_t v70 = __riscv_vwmacc_vx_i32m1(v69, v68, v42, v39);
    int8_t v71 = v5[v48];
    int16_t v72 = (int16_t) v71;
    vint32m1_t v73 = v34;
    vint32m1_t v74 = __riscv_vwmacc_vx_i32m1(v73, v72, v42, v39);
    int8_t v75 = v5[v49];
    int16_t v76 = (int16_t) v75;
    vint32m1_t v77 = v35;
    vint32m1_t v78 = __riscv_vwmacc_vx_i32m1(v77, v76, v42, v39);
    int8_t v79 = v5[v50];
    int16_t v80 = (int16_t) v79;
    vint32m1_t v81 = v36;
    vint32m1_t v82 = __riscv_vwmacc_vx_i32m1(v81, v80, v42, v39);
    v29 = v54;
    v30 = v58;
    v31 = v62;
    v32 = v66;
    v33 = v70;
    v34 = v74;
    v35 = v78;
    v36 = v82;
  }
  vint32m1_t v83 = v29;
  vint32m1_t v84 = v30;
  vint32m1_t v85 = v31;
  vint32m1_t v86 = v32;
  vint32m1_t v87 = v33;
  vint32m1_t v88 = v34;
  vint32m1_t v89 = v35;
  vint32m1_t v90 = v36;
  int32_t* v91 = &v8[v10];
  __riscv_vse32_v_i32m1(v91, v83, v14);
  int32_t* v92 = &v8[v9];
  __riscv_vse32_v_i32m1(v92, v84, v14);
  size_t v93 = v12 * v9;
  int32_t* v94 = &v8[v93];
  __riscv_vse32_v_i32m1(v94, v85, v14);
  size_t v95 = v13 * v9;
  int32_t* v96 = &v8[v95];
  __riscv_vse32_v_i32m1(v96, v86, v14);
  size_t v97 = v14 * v9;
  int32_t* v98 = &v8[v97];
  __riscv_vse32_v_i32m1(v98, v87, v14);
  size_t v99 = v15 * v9;
  int32_t* v100 = &v8[v99];
  __riscv_vse32_v_i32m1(v100, v88, v14);
  size_t v101 = v16 * v9;
  int32_t* v102 = &v8[v101];
  __riscv_vse32_v_i32m1(v102, v89, v14);
  size_t v103 = v17 * v9;
  int32_t* v104 = &v8[v103];
  __riscv_vse32_v_i32m1(v104, v90, v14);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_4x8_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 4
//   NR         : 8
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_4x8_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v14);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v14);
  size_t v37 = v14 * v9;
  size_t v38 = v10 + v37;
  int32_t* v39 = &v8[v38];
  vint32m1_t v40 = __riscv_vle32_v_i32m1(v39, v14);
  size_t v41 = v15 * v9;
  size_t v42 = v10 + v41;
  int32_t* v43 = &v8[v42];
  vint32m1_t v44 = __riscv_vle32_v_i32m1(v43, v14);
  size_t v45 = v16 * v9;
  size_t v46 = v10 + v45;
  int32_t* v47 = &v8[v46];
  vint32m1_t v48 = __riscv_vle32_v_i32m1(v47, v14);
  size_t v49 = v17 * v9;
  size_t v50 = v10 + v49;
  int32_t* v51 = &v8[v50];
  vint32m1_t v52 = __riscv_vle32_v_i32m1(v51, v14);
  vint32m1_t v53;
  v53 = v24;
  vint32m1_t v54;
  v54 = v28;
  vint32m1_t v55;
  v55 = v32;
  vint32m1_t v56;
  v56 = v36;
  vint32m1_t v57;
  v57 = v40;
  vint32m1_t v58;
  v58 = v44;
  vint32m1_t v59;
  v59 = v48;
  vint32m1_t v60;
  v60 = v52;
  for (size_t i61 = v10; i61 < v1; i61 += v11) {
    size_t v62 = i61 * v4;
    size_t v63 = 4;
    int8_t* v64 = &v3[v62];
    vint8mf4_t v65 = __riscv_vle8_v_i8mf4(v64, v63);
    vint16mf2_t v66 = __riscv_vsext_vf2_i16mf2(v65, v63);
    size_t v67 = i61 * v6;
    size_t v68 = v67 + v11;
    size_t v69 = v67 + v12;
    size_t v70 = v67 + v13;
    size_t v71 = v67 + v14;
    size_t v72 = v67 + v15;
    size_t v73 = v67 + v16;
    size_t v74 = v67 + v17;
    int8_t v75 = v5[v67];
    int16_t v76 = (int16_t) v75;
    vint32m1_t v77 = v53;
    vint32m1_t v78 = __riscv_vwmacc_vx_i32m1(v77, v76, v66, v63);
    int8_t v79 = v5[v68];
    int16_t v80 = (int16_t) v79;
    vint32m1_t v81 = v54;
    vint32m1_t v82 = __riscv_vwmacc_vx_i32m1(v81, v80, v66, v63);
    int8_t v83 = v5[v69];
    int16_t v84 = (int16_t) v83;
    vint32m1_t v85 = v55;
    vint32m1_t v86 = __riscv_vwmacc_vx_i32m1(v85, v84, v66, v63);
    int8_t v87 = v5[v70];
    int16_t v88 = (int16_t) v87;
    vint32m1_t v89 = v56;
    vint32m1_t v90 = __riscv_vwmacc_vx_i32m1(v89, v88, v66, v63);
    int8_t v91 = v5[v71];
    int16_t v92 = (int16_t) v91;
    vint32m1_t v93 = v57;
    vint32m1_t v94 = __riscv_vwmacc_vx_i32m1(v93, v92, v66, v63);
    int8_t v95 = v5[v72];
    int16_t v96 = (int16_t) v95;
    vint32m1_t v97 = v58;
    vint32m1_t v98 = __riscv_vwmacc_vx_i32m1(v97, v96, v66, v63);
    int8_t v99 = v5[v73];
    int16_t v100 = (int16_t) v99;
    vint32m1_t v101 = v59;
    vint32m1_t v102 = __riscv_vwmacc_vx_i32m1(v101, v100, v66, v63);
    int8_t v103 = v5[v74];
    int16_t v104 = (int16_t) v103;
    vint32m1_t v105 = v60;
    vint32m1_t v106 = __riscv_vwmacc_vx_i32m1(v105, v104, v66, v63);
    v53 = v78;
    v54 = v82;
    v55 = v86;
    v56 = v90;
    v57 = v94;
    v58 = v98;
    v59 = v102;
    v60 = v106;
  }
  vint32m1_t v107 = v53;
  vint32m1_t v108 = v54;
  vint32m1_t v109 = v55;
  vint32m1_t v110 = v56;
  vint32m1_t v111 = v57;
  vint32m1_t v112 = v58;
  vint32m1_t v113 = v59;
  vint32m1_t v114 = v60;
  int32_t* v115 = &v8[v10];
  __riscv_vse32_v_i32m1(v115, v107, v14);
  int32_t* v116 = &v8[v9];
  __riscv_vse32_v_i32m1(v116, v108, v14);
  size_t v117 = v12 * v9;
  int32_t* v118 = &v8[v117];
  __riscv_vse32_v_i32m1(v118, v109, v14);
  size_t v119 = v13 * v9;
  int32_t* v120 = &v8[v119];
  __riscv_vse32_v_i32m1(v120, v110, v14);
  size_t v121 = v14 * v9;
  int32_t* v122 = &v8[v121];
  __riscv_vse32_v_i32m1(v122, v111, v14);
  size_t v123 = v15 * v9;
  int32_t* v124 = &v8[v123];
  __riscv_vse32_v_i32m1(v124, v112, v14);
  size_t v125 = v16 * v9;
  int32_t* v126 = &v8[v125];
  __riscv_vse32_v_i32m1(v126, v113, v14);
  size_t v127 = v17 * v9;
  int32_t* v128 = &v8[v127];
  __riscv_vse32_v_i32m1(v128, v114, v14);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_5x1_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 5
//   NR         : 1
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_5x1_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v22;
  v22 = v21;
  for (size_t i23 = v10; i23 < v1; i23 += v11) {
    size_t v24 = i23 * v4;
    size_t v25 = 5;
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
  __riscv_vse32_v_i32m1(v35, v34, v15);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_5x1_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 5
//   NR         : 1
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_5x1_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v15);
  vint32m1_t v25;
  v25 = v24;
  for (size_t i26 = v10; i26 < v1; i26 += v11) {
    size_t v27 = i26 * v4;
    size_t v28 = 5;
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
  __riscv_vse32_v_i32m1(v38, v37, v15);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_5x2_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 5
//   NR         : 2
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_5x2_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v22 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v23;
  v23 = v21;
  vint32m1_t v24;
  v24 = v22;
  for (size_t i25 = v10; i25 < v1; i25 += v11) {
    size_t v26 = i25 * v4;
    size_t v27 = 5;
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
  __riscv_vse32_v_i32m1(v43, v41, v15);
  int32_t* v44 = &v8[v9];
  __riscv_vse32_v_i32m1(v44, v42, v15);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_5x2_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 5
//   NR         : 2
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_5x2_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v15);
  size_t v25 = v11 * v9;
  size_t v26 = v10 + v25;
  int32_t* v27 = &v8[v26];
  vint32m1_t v28 = __riscv_vle32_v_i32m1(v27, v15);
  vint32m1_t v29;
  v29 = v24;
  vint32m1_t v30;
  v30 = v28;
  for (size_t i31 = v10; i31 < v1; i31 += v11) {
    size_t v32 = i31 * v4;
    size_t v33 = 5;
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
  __riscv_vse32_v_i32m1(v49, v47, v15);
  int32_t* v50 = &v8[v9];
  __riscv_vse32_v_i32m1(v50, v48, v15);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_5x3_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 5
//   NR         : 3
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_5x3_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v22 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v24;
  v24 = v21;
  vint32m1_t v25;
  v25 = v22;
  vint32m1_t v26;
  v26 = v23;
  for (size_t i27 = v10; i27 < v1; i27 += v11) {
    size_t v28 = i27 * v4;
    size_t v29 = 5;
    int8_t* v30 = &v3[v28];
    vint8mf4_t v31 = __riscv_vle8_v_i8mf4(v30, v29);
    vint16mf2_t v32 = __riscv_vsext_vf2_i16mf2(v31, v29);
    size_t v33 = i27 * v6;
    size_t v34 = v33 + v11;
    size_t v35 = v33 + v12;
    int8_t v36 = v5[v33];
    int16_t v37 = (int16_t) v36;
    vint32m1_t v38 = v24;
    vint32m1_t v39 = __riscv_vwmacc_vx_i32m1(v38, v37, v32, v29);
    int8_t v40 = v5[v34];
    int16_t v41 = (int16_t) v40;
    vint32m1_t v42 = v25;
    vint32m1_t v43 = __riscv_vwmacc_vx_i32m1(v42, v41, v32, v29);
    int8_t v44 = v5[v35];
    int16_t v45 = (int16_t) v44;
    vint32m1_t v46 = v26;
    vint32m1_t v47 = __riscv_vwmacc_vx_i32m1(v46, v45, v32, v29);
    v24 = v39;
    v25 = v43;
    v26 = v47;
  }
  vint32m1_t v48 = v24;
  vint32m1_t v49 = v25;
  vint32m1_t v50 = v26;
  int32_t* v51 = &v8[v10];
  __riscv_vse32_v_i32m1(v51, v48, v15);
  int32_t* v52 = &v8[v9];
  __riscv_vse32_v_i32m1(v52, v49, v15);
  size_t v53 = v12 * v9;
  int32_t* v54 = &v8[v53];
  __riscv_vse32_v_i32m1(v54, v50, v15);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_5x3_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 5
//   NR         : 3
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_5x3_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v15);
  size_t v25 = v11 * v9;
  size_t v26 = v10 + v25;
  int32_t* v27 = &v8[v26];
  vint32m1_t v28 = __riscv_vle32_v_i32m1(v27, v15);
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v15);
  vint32m1_t v33;
  v33 = v24;
  vint32m1_t v34;
  v34 = v28;
  vint32m1_t v35;
  v35 = v32;
  for (size_t i36 = v10; i36 < v1; i36 += v11) {
    size_t v37 = i36 * v4;
    size_t v38 = 5;
    int8_t* v39 = &v3[v37];
    vint8mf4_t v40 = __riscv_vle8_v_i8mf4(v39, v38);
    vint16mf2_t v41 = __riscv_vsext_vf2_i16mf2(v40, v38);
    size_t v42 = i36 * v6;
    size_t v43 = v42 + v11;
    size_t v44 = v42 + v12;
    int8_t v45 = v5[v42];
    int16_t v46 = (int16_t) v45;
    vint32m1_t v47 = v33;
    vint32m1_t v48 = __riscv_vwmacc_vx_i32m1(v47, v46, v41, v38);
    int8_t v49 = v5[v43];
    int16_t v50 = (int16_t) v49;
    vint32m1_t v51 = v34;
    vint32m1_t v52 = __riscv_vwmacc_vx_i32m1(v51, v50, v41, v38);
    int8_t v53 = v5[v44];
    int16_t v54 = (int16_t) v53;
    vint32m1_t v55 = v35;
    vint32m1_t v56 = __riscv_vwmacc_vx_i32m1(v55, v54, v41, v38);
    v33 = v48;
    v34 = v52;
    v35 = v56;
  }
  vint32m1_t v57 = v33;
  vint32m1_t v58 = v34;
  vint32m1_t v59 = v35;
  int32_t* v60 = &v8[v10];
  __riscv_vse32_v_i32m1(v60, v57, v15);
  int32_t* v61 = &v8[v9];
  __riscv_vse32_v_i32m1(v61, v58, v15);
  size_t v62 = v12 * v9;
  int32_t* v63 = &v8[v62];
  __riscv_vse32_v_i32m1(v63, v59, v15);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_5x4_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 5
//   NR         : 4
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_5x4_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v22 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v25;
  v25 = v21;
  vint32m1_t v26;
  v26 = v22;
  vint32m1_t v27;
  v27 = v23;
  vint32m1_t v28;
  v28 = v24;
  for (size_t i29 = v10; i29 < v1; i29 += v11) {
    size_t v30 = i29 * v4;
    size_t v31 = 5;
    int8_t* v32 = &v3[v30];
    vint8mf4_t v33 = __riscv_vle8_v_i8mf4(v32, v31);
    vint16mf2_t v34 = __riscv_vsext_vf2_i16mf2(v33, v31);
    size_t v35 = i29 * v6;
    size_t v36 = v35 + v11;
    size_t v37 = v35 + v12;
    size_t v38 = v35 + v13;
    int8_t v39 = v5[v35];
    int16_t v40 = (int16_t) v39;
    vint32m1_t v41 = v25;
    vint32m1_t v42 = __riscv_vwmacc_vx_i32m1(v41, v40, v34, v31);
    int8_t v43 = v5[v36];
    int16_t v44 = (int16_t) v43;
    vint32m1_t v45 = v26;
    vint32m1_t v46 = __riscv_vwmacc_vx_i32m1(v45, v44, v34, v31);
    int8_t v47 = v5[v37];
    int16_t v48 = (int16_t) v47;
    vint32m1_t v49 = v27;
    vint32m1_t v50 = __riscv_vwmacc_vx_i32m1(v49, v48, v34, v31);
    int8_t v51 = v5[v38];
    int16_t v52 = (int16_t) v51;
    vint32m1_t v53 = v28;
    vint32m1_t v54 = __riscv_vwmacc_vx_i32m1(v53, v52, v34, v31);
    v25 = v42;
    v26 = v46;
    v27 = v50;
    v28 = v54;
  }
  vint32m1_t v55 = v25;
  vint32m1_t v56 = v26;
  vint32m1_t v57 = v27;
  vint32m1_t v58 = v28;
  int32_t* v59 = &v8[v10];
  __riscv_vse32_v_i32m1(v59, v55, v15);
  int32_t* v60 = &v8[v9];
  __riscv_vse32_v_i32m1(v60, v56, v15);
  size_t v61 = v12 * v9;
  int32_t* v62 = &v8[v61];
  __riscv_vse32_v_i32m1(v62, v57, v15);
  size_t v63 = v13 * v9;
  int32_t* v64 = &v8[v63];
  __riscv_vse32_v_i32m1(v64, v58, v15);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_5x4_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:10
//   MR         : 5
//   NR         : 4
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_5x4_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v15);
  size_t v25 = v11 * v9;
  size_t v26 = v10 + v25;
  int32_t* v27 = &v8[v26];
  vint32m1_t v28 = __riscv_vle32_v_i32m1(v27, v15);
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v15);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v15);
  vint32m1_t v37;
  v37 = v24;
  vint32m1_t v38;
  v38 = v28;
  vint32m1_t v39;
  v39 = v32;
  vint32m1_t v40;
  v40 = v36;
  for (size_t i41 = v10; i41 < v1; i41 += v11) {
    size_t v42 = i41 * v4;
    size_t v43 = 5;
    int8_t* v44 = &v3[v42];
    vint8mf4_t v45 = __riscv_vle8_v_i8mf4(v44, v43);
    vint16mf2_t v46 = __riscv_vsext_vf2_i16mf2(v45, v43);
    size_t v47 = i41 * v6;
    size_t v48 = v47 + v11;
    size_t v49 = v47 + v12;
    size_t v50 = v47 + v13;
    int8_t v51 = v5[v47];
    int16_t v52 = (int16_t) v51;
    vint32m1_t v53 = v37;
    vint32m1_t v54 = __riscv_vwmacc_vx_i32m1(v53, v52, v46, v43);
    int8_t v55 = v5[v48];
    int16_t v56 = (int16_t) v55;
    vint32m1_t v57 = v38;
    vint32m1_t v58 = __riscv_vwmacc_vx_i32m1(v57, v56, v46, v43);
    int8_t v59 = v5[v49];
    int16_t v60 = (int16_t) v59;
    vint32m1_t v61 = v39;
    vint32m1_t v62 = __riscv_vwmacc_vx_i32m1(v61, v60, v46, v43);
    int8_t v63 = v5[v50];
    int16_t v64 = (int16_t) v63;
    vint32m1_t v65 = v40;
    vint32m1_t v66 = __riscv_vwmacc_vx_i32m1(v65, v64, v46, v43);
    v37 = v54;
    v38 = v58;
    v39 = v62;
    v40 = v66;
  }
  vint32m1_t v67 = v37;
  vint32m1_t v68 = v38;
  vint32m1_t v69 = v39;
  vint32m1_t v70 = v40;
  int32_t* v71 = &v8[v10];
  __riscv_vse32_v_i32m1(v71, v67, v15);
  int32_t* v72 = &v8[v9];
  __riscv_vse32_v_i32m1(v72, v68, v15);
  size_t v73 = v12 * v9;
  int32_t* v74 = &v8[v73];
  __riscv_vse32_v_i32m1(v74, v69, v15);
  size_t v75 = v13 * v9;
  int32_t* v76 = &v8[v75];
  __riscv_vse32_v_i32m1(v76, v70, v15);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_5x5_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 5
//   NR         : 5
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_5x5_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v22 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v25 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v26;
  v26 = v21;
  vint32m1_t v27;
  v27 = v22;
  vint32m1_t v28;
  v28 = v23;
  vint32m1_t v29;
  v29 = v24;
  vint32m1_t v30;
  v30 = v25;
  for (size_t i31 = v10; i31 < v1; i31 += v11) {
    size_t v32 = i31 * v4;
    size_t v33 = 5;
    int8_t* v34 = &v3[v32];
    vint8mf4_t v35 = __riscv_vle8_v_i8mf4(v34, v33);
    vint16mf2_t v36 = __riscv_vsext_vf2_i16mf2(v35, v33);
    size_t v37 = i31 * v6;
    size_t v38 = v37 + v11;
    size_t v39 = v37 + v12;
    size_t v40 = v37 + v13;
    size_t v41 = v37 + v14;
    int8_t v42 = v5[v37];
    int16_t v43 = (int16_t) v42;
    vint32m1_t v44 = v26;
    vint32m1_t v45 = __riscv_vwmacc_vx_i32m1(v44, v43, v36, v33);
    int8_t v46 = v5[v38];
    int16_t v47 = (int16_t) v46;
    vint32m1_t v48 = v27;
    vint32m1_t v49 = __riscv_vwmacc_vx_i32m1(v48, v47, v36, v33);
    int8_t v50 = v5[v39];
    int16_t v51 = (int16_t) v50;
    vint32m1_t v52 = v28;
    vint32m1_t v53 = __riscv_vwmacc_vx_i32m1(v52, v51, v36, v33);
    int8_t v54 = v5[v40];
    int16_t v55 = (int16_t) v54;
    vint32m1_t v56 = v29;
    vint32m1_t v57 = __riscv_vwmacc_vx_i32m1(v56, v55, v36, v33);
    int8_t v58 = v5[v41];
    int16_t v59 = (int16_t) v58;
    vint32m1_t v60 = v30;
    vint32m1_t v61 = __riscv_vwmacc_vx_i32m1(v60, v59, v36, v33);
    v26 = v45;
    v27 = v49;
    v28 = v53;
    v29 = v57;
    v30 = v61;
  }
  vint32m1_t v62 = v26;
  vint32m1_t v63 = v27;
  vint32m1_t v64 = v28;
  vint32m1_t v65 = v29;
  vint32m1_t v66 = v30;
  int32_t* v67 = &v8[v10];
  __riscv_vse32_v_i32m1(v67, v62, v15);
  int32_t* v68 = &v8[v9];
  __riscv_vse32_v_i32m1(v68, v63, v15);
  size_t v69 = v12 * v9;
  int32_t* v70 = &v8[v69];
  __riscv_vse32_v_i32m1(v70, v64, v15);
  size_t v71 = v13 * v9;
  int32_t* v72 = &v8[v71];
  __riscv_vse32_v_i32m1(v72, v65, v15);
  size_t v73 = v14 * v9;
  int32_t* v74 = &v8[v73];
  __riscv_vse32_v_i32m1(v74, v66, v15);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_5x5_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 5
//   NR         : 5
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_5x5_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v15);
  size_t v25 = v11 * v9;
  size_t v26 = v10 + v25;
  int32_t* v27 = &v8[v26];
  vint32m1_t v28 = __riscv_vle32_v_i32m1(v27, v15);
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v15);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v15);
  size_t v37 = v14 * v9;
  size_t v38 = v10 + v37;
  int32_t* v39 = &v8[v38];
  vint32m1_t v40 = __riscv_vle32_v_i32m1(v39, v15);
  vint32m1_t v41;
  v41 = v24;
  vint32m1_t v42;
  v42 = v28;
  vint32m1_t v43;
  v43 = v32;
  vint32m1_t v44;
  v44 = v36;
  vint32m1_t v45;
  v45 = v40;
  for (size_t i46 = v10; i46 < v1; i46 += v11) {
    size_t v47 = i46 * v4;
    size_t v48 = 5;
    int8_t* v49 = &v3[v47];
    vint8mf4_t v50 = __riscv_vle8_v_i8mf4(v49, v48);
    vint16mf2_t v51 = __riscv_vsext_vf2_i16mf2(v50, v48);
    size_t v52 = i46 * v6;
    size_t v53 = v52 + v11;
    size_t v54 = v52 + v12;
    size_t v55 = v52 + v13;
    size_t v56 = v52 + v14;
    int8_t v57 = v5[v52];
    int16_t v58 = (int16_t) v57;
    vint32m1_t v59 = v41;
    vint32m1_t v60 = __riscv_vwmacc_vx_i32m1(v59, v58, v51, v48);
    int8_t v61 = v5[v53];
    int16_t v62 = (int16_t) v61;
    vint32m1_t v63 = v42;
    vint32m1_t v64 = __riscv_vwmacc_vx_i32m1(v63, v62, v51, v48);
    int8_t v65 = v5[v54];
    int16_t v66 = (int16_t) v65;
    vint32m1_t v67 = v43;
    vint32m1_t v68 = __riscv_vwmacc_vx_i32m1(v67, v66, v51, v48);
    int8_t v69 = v5[v55];
    int16_t v70 = (int16_t) v69;
    vint32m1_t v71 = v44;
    vint32m1_t v72 = __riscv_vwmacc_vx_i32m1(v71, v70, v51, v48);
    int8_t v73 = v5[v56];
    int16_t v74 = (int16_t) v73;
    vint32m1_t v75 = v45;
    vint32m1_t v76 = __riscv_vwmacc_vx_i32m1(v75, v74, v51, v48);
    v41 = v60;
    v42 = v64;
    v43 = v68;
    v44 = v72;
    v45 = v76;
  }
  vint32m1_t v77 = v41;
  vint32m1_t v78 = v42;
  vint32m1_t v79 = v43;
  vint32m1_t v80 = v44;
  vint32m1_t v81 = v45;
  int32_t* v82 = &v8[v10];
  __riscv_vse32_v_i32m1(v82, v77, v15);
  int32_t* v83 = &v8[v9];
  __riscv_vse32_v_i32m1(v83, v78, v15);
  size_t v84 = v12 * v9;
  int32_t* v85 = &v8[v84];
  __riscv_vse32_v_i32m1(v85, v79, v15);
  size_t v86 = v13 * v9;
  int32_t* v87 = &v8[v86];
  __riscv_vse32_v_i32m1(v87, v80, v15);
  size_t v88 = v14 * v9;
  int32_t* v89 = &v8[v88];
  __riscv_vse32_v_i32m1(v89, v81, v15);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_5x6_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 5
//   NR         : 6
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_5x6_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v22 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v25 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v26 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v27;
  v27 = v21;
  vint32m1_t v28;
  v28 = v22;
  vint32m1_t v29;
  v29 = v23;
  vint32m1_t v30;
  v30 = v24;
  vint32m1_t v31;
  v31 = v25;
  vint32m1_t v32;
  v32 = v26;
  for (size_t i33 = v10; i33 < v1; i33 += v11) {
    size_t v34 = i33 * v4;
    size_t v35 = 5;
    int8_t* v36 = &v3[v34];
    vint8mf4_t v37 = __riscv_vle8_v_i8mf4(v36, v35);
    vint16mf2_t v38 = __riscv_vsext_vf2_i16mf2(v37, v35);
    size_t v39 = i33 * v6;
    size_t v40 = v39 + v11;
    size_t v41 = v39 + v12;
    size_t v42 = v39 + v13;
    size_t v43 = v39 + v14;
    size_t v44 = v39 + v15;
    int8_t v45 = v5[v39];
    int16_t v46 = (int16_t) v45;
    vint32m1_t v47 = v27;
    vint32m1_t v48 = __riscv_vwmacc_vx_i32m1(v47, v46, v38, v35);
    int8_t v49 = v5[v40];
    int16_t v50 = (int16_t) v49;
    vint32m1_t v51 = v28;
    vint32m1_t v52 = __riscv_vwmacc_vx_i32m1(v51, v50, v38, v35);
    int8_t v53 = v5[v41];
    int16_t v54 = (int16_t) v53;
    vint32m1_t v55 = v29;
    vint32m1_t v56 = __riscv_vwmacc_vx_i32m1(v55, v54, v38, v35);
    int8_t v57 = v5[v42];
    int16_t v58 = (int16_t) v57;
    vint32m1_t v59 = v30;
    vint32m1_t v60 = __riscv_vwmacc_vx_i32m1(v59, v58, v38, v35);
    int8_t v61 = v5[v43];
    int16_t v62 = (int16_t) v61;
    vint32m1_t v63 = v31;
    vint32m1_t v64 = __riscv_vwmacc_vx_i32m1(v63, v62, v38, v35);
    int8_t v65 = v5[v44];
    int16_t v66 = (int16_t) v65;
    vint32m1_t v67 = v32;
    vint32m1_t v68 = __riscv_vwmacc_vx_i32m1(v67, v66, v38, v35);
    v27 = v48;
    v28 = v52;
    v29 = v56;
    v30 = v60;
    v31 = v64;
    v32 = v68;
  }
  vint32m1_t v69 = v27;
  vint32m1_t v70 = v28;
  vint32m1_t v71 = v29;
  vint32m1_t v72 = v30;
  vint32m1_t v73 = v31;
  vint32m1_t v74 = v32;
  int32_t* v75 = &v8[v10];
  __riscv_vse32_v_i32m1(v75, v69, v15);
  int32_t* v76 = &v8[v9];
  __riscv_vse32_v_i32m1(v76, v70, v15);
  size_t v77 = v12 * v9;
  int32_t* v78 = &v8[v77];
  __riscv_vse32_v_i32m1(v78, v71, v15);
  size_t v79 = v13 * v9;
  int32_t* v80 = &v8[v79];
  __riscv_vse32_v_i32m1(v80, v72, v15);
  size_t v81 = v14 * v9;
  int32_t* v82 = &v8[v81];
  __riscv_vse32_v_i32m1(v82, v73, v15);
  size_t v83 = v15 * v9;
  int32_t* v84 = &v8[v83];
  __riscv_vse32_v_i32m1(v84, v74, v15);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_5x6_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 5
//   NR         : 6
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_5x6_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v15);
  size_t v25 = v11 * v9;
  size_t v26 = v10 + v25;
  int32_t* v27 = &v8[v26];
  vint32m1_t v28 = __riscv_vle32_v_i32m1(v27, v15);
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v15);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v15);
  size_t v37 = v14 * v9;
  size_t v38 = v10 + v37;
  int32_t* v39 = &v8[v38];
  vint32m1_t v40 = __riscv_vle32_v_i32m1(v39, v15);
  size_t v41 = v15 * v9;
  size_t v42 = v10 + v41;
  int32_t* v43 = &v8[v42];
  vint32m1_t v44 = __riscv_vle32_v_i32m1(v43, v15);
  vint32m1_t v45;
  v45 = v24;
  vint32m1_t v46;
  v46 = v28;
  vint32m1_t v47;
  v47 = v32;
  vint32m1_t v48;
  v48 = v36;
  vint32m1_t v49;
  v49 = v40;
  vint32m1_t v50;
  v50 = v44;
  for (size_t i51 = v10; i51 < v1; i51 += v11) {
    size_t v52 = i51 * v4;
    size_t v53 = 5;
    int8_t* v54 = &v3[v52];
    vint8mf4_t v55 = __riscv_vle8_v_i8mf4(v54, v53);
    vint16mf2_t v56 = __riscv_vsext_vf2_i16mf2(v55, v53);
    size_t v57 = i51 * v6;
    size_t v58 = v57 + v11;
    size_t v59 = v57 + v12;
    size_t v60 = v57 + v13;
    size_t v61 = v57 + v14;
    size_t v62 = v57 + v15;
    int8_t v63 = v5[v57];
    int16_t v64 = (int16_t) v63;
    vint32m1_t v65 = v45;
    vint32m1_t v66 = __riscv_vwmacc_vx_i32m1(v65, v64, v56, v53);
    int8_t v67 = v5[v58];
    int16_t v68 = (int16_t) v67;
    vint32m1_t v69 = v46;
    vint32m1_t v70 = __riscv_vwmacc_vx_i32m1(v69, v68, v56, v53);
    int8_t v71 = v5[v59];
    int16_t v72 = (int16_t) v71;
    vint32m1_t v73 = v47;
    vint32m1_t v74 = __riscv_vwmacc_vx_i32m1(v73, v72, v56, v53);
    int8_t v75 = v5[v60];
    int16_t v76 = (int16_t) v75;
    vint32m1_t v77 = v48;
    vint32m1_t v78 = __riscv_vwmacc_vx_i32m1(v77, v76, v56, v53);
    int8_t v79 = v5[v61];
    int16_t v80 = (int16_t) v79;
    vint32m1_t v81 = v49;
    vint32m1_t v82 = __riscv_vwmacc_vx_i32m1(v81, v80, v56, v53);
    int8_t v83 = v5[v62];
    int16_t v84 = (int16_t) v83;
    vint32m1_t v85 = v50;
    vint32m1_t v86 = __riscv_vwmacc_vx_i32m1(v85, v84, v56, v53);
    v45 = v66;
    v46 = v70;
    v47 = v74;
    v48 = v78;
    v49 = v82;
    v50 = v86;
  }
  vint32m1_t v87 = v45;
  vint32m1_t v88 = v46;
  vint32m1_t v89 = v47;
  vint32m1_t v90 = v48;
  vint32m1_t v91 = v49;
  vint32m1_t v92 = v50;
  int32_t* v93 = &v8[v10];
  __riscv_vse32_v_i32m1(v93, v87, v15);
  int32_t* v94 = &v8[v9];
  __riscv_vse32_v_i32m1(v94, v88, v15);
  size_t v95 = v12 * v9;
  int32_t* v96 = &v8[v95];
  __riscv_vse32_v_i32m1(v96, v89, v15);
  size_t v97 = v13 * v9;
  int32_t* v98 = &v8[v97];
  __riscv_vse32_v_i32m1(v98, v90, v15);
  size_t v99 = v14 * v9;
  int32_t* v100 = &v8[v99];
  __riscv_vse32_v_i32m1(v100, v91, v15);
  size_t v101 = v15 * v9;
  int32_t* v102 = &v8[v101];
  __riscv_vse32_v_i32m1(v102, v92, v15);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_5x7_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 5
//   NR         : 7
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_5x7_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v22 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v25 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v26 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v27 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v28;
  v28 = v21;
  vint32m1_t v29;
  v29 = v22;
  vint32m1_t v30;
  v30 = v23;
  vint32m1_t v31;
  v31 = v24;
  vint32m1_t v32;
  v32 = v25;
  vint32m1_t v33;
  v33 = v26;
  vint32m1_t v34;
  v34 = v27;
  for (size_t i35 = v10; i35 < v1; i35 += v11) {
    size_t v36 = i35 * v4;
    size_t v37 = 5;
    int8_t* v38 = &v3[v36];
    vint8mf4_t v39 = __riscv_vle8_v_i8mf4(v38, v37);
    vint16mf2_t v40 = __riscv_vsext_vf2_i16mf2(v39, v37);
    size_t v41 = i35 * v6;
    size_t v42 = v41 + v11;
    size_t v43 = v41 + v12;
    size_t v44 = v41 + v13;
    size_t v45 = v41 + v14;
    size_t v46 = v41 + v15;
    size_t v47 = v41 + v16;
    int8_t v48 = v5[v41];
    int16_t v49 = (int16_t) v48;
    vint32m1_t v50 = v28;
    vint32m1_t v51 = __riscv_vwmacc_vx_i32m1(v50, v49, v40, v37);
    int8_t v52 = v5[v42];
    int16_t v53 = (int16_t) v52;
    vint32m1_t v54 = v29;
    vint32m1_t v55 = __riscv_vwmacc_vx_i32m1(v54, v53, v40, v37);
    int8_t v56 = v5[v43];
    int16_t v57 = (int16_t) v56;
    vint32m1_t v58 = v30;
    vint32m1_t v59 = __riscv_vwmacc_vx_i32m1(v58, v57, v40, v37);
    int8_t v60 = v5[v44];
    int16_t v61 = (int16_t) v60;
    vint32m1_t v62 = v31;
    vint32m1_t v63 = __riscv_vwmacc_vx_i32m1(v62, v61, v40, v37);
    int8_t v64 = v5[v45];
    int16_t v65 = (int16_t) v64;
    vint32m1_t v66 = v32;
    vint32m1_t v67 = __riscv_vwmacc_vx_i32m1(v66, v65, v40, v37);
    int8_t v68 = v5[v46];
    int16_t v69 = (int16_t) v68;
    vint32m1_t v70 = v33;
    vint32m1_t v71 = __riscv_vwmacc_vx_i32m1(v70, v69, v40, v37);
    int8_t v72 = v5[v47];
    int16_t v73 = (int16_t) v72;
    vint32m1_t v74 = v34;
    vint32m1_t v75 = __riscv_vwmacc_vx_i32m1(v74, v73, v40, v37);
    v28 = v51;
    v29 = v55;
    v30 = v59;
    v31 = v63;
    v32 = v67;
    v33 = v71;
    v34 = v75;
  }
  vint32m1_t v76 = v28;
  vint32m1_t v77 = v29;
  vint32m1_t v78 = v30;
  vint32m1_t v79 = v31;
  vint32m1_t v80 = v32;
  vint32m1_t v81 = v33;
  vint32m1_t v82 = v34;
  int32_t* v83 = &v8[v10];
  __riscv_vse32_v_i32m1(v83, v76, v15);
  int32_t* v84 = &v8[v9];
  __riscv_vse32_v_i32m1(v84, v77, v15);
  size_t v85 = v12 * v9;
  int32_t* v86 = &v8[v85];
  __riscv_vse32_v_i32m1(v86, v78, v15);
  size_t v87 = v13 * v9;
  int32_t* v88 = &v8[v87];
  __riscv_vse32_v_i32m1(v88, v79, v15);
  size_t v89 = v14 * v9;
  int32_t* v90 = &v8[v89];
  __riscv_vse32_v_i32m1(v90, v80, v15);
  size_t v91 = v15 * v9;
  int32_t* v92 = &v8[v91];
  __riscv_vse32_v_i32m1(v92, v81, v15);
  size_t v93 = v16 * v9;
  int32_t* v94 = &v8[v93];
  __riscv_vse32_v_i32m1(v94, v82, v15);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_5x7_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 5
//   NR         : 7
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_5x7_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v15);
  size_t v25 = v11 * v9;
  size_t v26 = v10 + v25;
  int32_t* v27 = &v8[v26];
  vint32m1_t v28 = __riscv_vle32_v_i32m1(v27, v15);
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v15);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v15);
  size_t v37 = v14 * v9;
  size_t v38 = v10 + v37;
  int32_t* v39 = &v8[v38];
  vint32m1_t v40 = __riscv_vle32_v_i32m1(v39, v15);
  size_t v41 = v15 * v9;
  size_t v42 = v10 + v41;
  int32_t* v43 = &v8[v42];
  vint32m1_t v44 = __riscv_vle32_v_i32m1(v43, v15);
  size_t v45 = v16 * v9;
  size_t v46 = v10 + v45;
  int32_t* v47 = &v8[v46];
  vint32m1_t v48 = __riscv_vle32_v_i32m1(v47, v15);
  vint32m1_t v49;
  v49 = v24;
  vint32m1_t v50;
  v50 = v28;
  vint32m1_t v51;
  v51 = v32;
  vint32m1_t v52;
  v52 = v36;
  vint32m1_t v53;
  v53 = v40;
  vint32m1_t v54;
  v54 = v44;
  vint32m1_t v55;
  v55 = v48;
  for (size_t i56 = v10; i56 < v1; i56 += v11) {
    size_t v57 = i56 * v4;
    size_t v58 = 5;
    int8_t* v59 = &v3[v57];
    vint8mf4_t v60 = __riscv_vle8_v_i8mf4(v59, v58);
    vint16mf2_t v61 = __riscv_vsext_vf2_i16mf2(v60, v58);
    size_t v62 = i56 * v6;
    size_t v63 = v62 + v11;
    size_t v64 = v62 + v12;
    size_t v65 = v62 + v13;
    size_t v66 = v62 + v14;
    size_t v67 = v62 + v15;
    size_t v68 = v62 + v16;
    int8_t v69 = v5[v62];
    int16_t v70 = (int16_t) v69;
    vint32m1_t v71 = v49;
    vint32m1_t v72 = __riscv_vwmacc_vx_i32m1(v71, v70, v61, v58);
    int8_t v73 = v5[v63];
    int16_t v74 = (int16_t) v73;
    vint32m1_t v75 = v50;
    vint32m1_t v76 = __riscv_vwmacc_vx_i32m1(v75, v74, v61, v58);
    int8_t v77 = v5[v64];
    int16_t v78 = (int16_t) v77;
    vint32m1_t v79 = v51;
    vint32m1_t v80 = __riscv_vwmacc_vx_i32m1(v79, v78, v61, v58);
    int8_t v81 = v5[v65];
    int16_t v82 = (int16_t) v81;
    vint32m1_t v83 = v52;
    vint32m1_t v84 = __riscv_vwmacc_vx_i32m1(v83, v82, v61, v58);
    int8_t v85 = v5[v66];
    int16_t v86 = (int16_t) v85;
    vint32m1_t v87 = v53;
    vint32m1_t v88 = __riscv_vwmacc_vx_i32m1(v87, v86, v61, v58);
    int8_t v89 = v5[v67];
    int16_t v90 = (int16_t) v89;
    vint32m1_t v91 = v54;
    vint32m1_t v92 = __riscv_vwmacc_vx_i32m1(v91, v90, v61, v58);
    int8_t v93 = v5[v68];
    int16_t v94 = (int16_t) v93;
    vint32m1_t v95 = v55;
    vint32m1_t v96 = __riscv_vwmacc_vx_i32m1(v95, v94, v61, v58);
    v49 = v72;
    v50 = v76;
    v51 = v80;
    v52 = v84;
    v53 = v88;
    v54 = v92;
    v55 = v96;
  }
  vint32m1_t v97 = v49;
  vint32m1_t v98 = v50;
  vint32m1_t v99 = v51;
  vint32m1_t v100 = v52;
  vint32m1_t v101 = v53;
  vint32m1_t v102 = v54;
  vint32m1_t v103 = v55;
  int32_t* v104 = &v8[v10];
  __riscv_vse32_v_i32m1(v104, v97, v15);
  int32_t* v105 = &v8[v9];
  __riscv_vse32_v_i32m1(v105, v98, v15);
  size_t v106 = v12 * v9;
  int32_t* v107 = &v8[v106];
  __riscv_vse32_v_i32m1(v107, v99, v15);
  size_t v108 = v13 * v9;
  int32_t* v109 = &v8[v108];
  __riscv_vse32_v_i32m1(v109, v100, v15);
  size_t v110 = v14 * v9;
  int32_t* v111 = &v8[v110];
  __riscv_vse32_v_i32m1(v111, v101, v15);
  size_t v112 = v15 * v9;
  int32_t* v113 = &v8[v112];
  __riscv_vse32_v_i32m1(v113, v102, v15);
  size_t v114 = v16 * v9;
  int32_t* v115 = &v8[v114];
  __riscv_vse32_v_i32m1(v115, v103, v15);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_5x8_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 5
//   NR         : 8
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_5x8_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v22 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v25 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v26 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v27 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v28 = __riscv_vmv_v_x_i32m1(v20, v15);
  vint32m1_t v29;
  v29 = v21;
  vint32m1_t v30;
  v30 = v22;
  vint32m1_t v31;
  v31 = v23;
  vint32m1_t v32;
  v32 = v24;
  vint32m1_t v33;
  v33 = v25;
  vint32m1_t v34;
  v34 = v26;
  vint32m1_t v35;
  v35 = v27;
  vint32m1_t v36;
  v36 = v28;
  for (size_t i37 = v10; i37 < v1; i37 += v11) {
    size_t v38 = i37 * v4;
    size_t v39 = 5;
    int8_t* v40 = &v3[v38];
    vint8mf4_t v41 = __riscv_vle8_v_i8mf4(v40, v39);
    vint16mf2_t v42 = __riscv_vsext_vf2_i16mf2(v41, v39);
    size_t v43 = i37 * v6;
    size_t v44 = v43 + v11;
    size_t v45 = v43 + v12;
    size_t v46 = v43 + v13;
    size_t v47 = v43 + v14;
    size_t v48 = v43 + v15;
    size_t v49 = v43 + v16;
    size_t v50 = v43 + v17;
    int8_t v51 = v5[v43];
    int16_t v52 = (int16_t) v51;
    vint32m1_t v53 = v29;
    vint32m1_t v54 = __riscv_vwmacc_vx_i32m1(v53, v52, v42, v39);
    int8_t v55 = v5[v44];
    int16_t v56 = (int16_t) v55;
    vint32m1_t v57 = v30;
    vint32m1_t v58 = __riscv_vwmacc_vx_i32m1(v57, v56, v42, v39);
    int8_t v59 = v5[v45];
    int16_t v60 = (int16_t) v59;
    vint32m1_t v61 = v31;
    vint32m1_t v62 = __riscv_vwmacc_vx_i32m1(v61, v60, v42, v39);
    int8_t v63 = v5[v46];
    int16_t v64 = (int16_t) v63;
    vint32m1_t v65 = v32;
    vint32m1_t v66 = __riscv_vwmacc_vx_i32m1(v65, v64, v42, v39);
    int8_t v67 = v5[v47];
    int16_t v68 = (int16_t) v67;
    vint32m1_t v69 = v33;
    vint32m1_t v70 = __riscv_vwmacc_vx_i32m1(v69, v68, v42, v39);
    int8_t v71 = v5[v48];
    int16_t v72 = (int16_t) v71;
    vint32m1_t v73 = v34;
    vint32m1_t v74 = __riscv_vwmacc_vx_i32m1(v73, v72, v42, v39);
    int8_t v75 = v5[v49];
    int16_t v76 = (int16_t) v75;
    vint32m1_t v77 = v35;
    vint32m1_t v78 = __riscv_vwmacc_vx_i32m1(v77, v76, v42, v39);
    int8_t v79 = v5[v50];
    int16_t v80 = (int16_t) v79;
    vint32m1_t v81 = v36;
    vint32m1_t v82 = __riscv_vwmacc_vx_i32m1(v81, v80, v42, v39);
    v29 = v54;
    v30 = v58;
    v31 = v62;
    v32 = v66;
    v33 = v70;
    v34 = v74;
    v35 = v78;
    v36 = v82;
  }
  vint32m1_t v83 = v29;
  vint32m1_t v84 = v30;
  vint32m1_t v85 = v31;
  vint32m1_t v86 = v32;
  vint32m1_t v87 = v33;
  vint32m1_t v88 = v34;
  vint32m1_t v89 = v35;
  vint32m1_t v90 = v36;
  int32_t* v91 = &v8[v10];
  __riscv_vse32_v_i32m1(v91, v83, v15);
  int32_t* v92 = &v8[v9];
  __riscv_vse32_v_i32m1(v92, v84, v15);
  size_t v93 = v12 * v9;
  int32_t* v94 = &v8[v93];
  __riscv_vse32_v_i32m1(v94, v85, v15);
  size_t v95 = v13 * v9;
  int32_t* v96 = &v8[v95];
  __riscv_vse32_v_i32m1(v96, v86, v15);
  size_t v97 = v14 * v9;
  int32_t* v98 = &v8[v97];
  __riscv_vse32_v_i32m1(v98, v87, v15);
  size_t v99 = v15 * v9;
  int32_t* v100 = &v8[v99];
  __riscv_vse32_v_i32m1(v100, v88, v15);
  size_t v101 = v16 * v9;
  int32_t* v102 = &v8[v101];
  __riscv_vse32_v_i32m1(v102, v89, v15);
  size_t v103 = v17 * v9;
  int32_t* v104 = &v8[v103];
  __riscv_vse32_v_i32m1(v104, v90, v15);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_5x8_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 5
//   NR         : 8
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_5x8_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v15);
  size_t v25 = v11 * v9;
  size_t v26 = v10 + v25;
  int32_t* v27 = &v8[v26];
  vint32m1_t v28 = __riscv_vle32_v_i32m1(v27, v15);
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v15);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v15);
  size_t v37 = v14 * v9;
  size_t v38 = v10 + v37;
  int32_t* v39 = &v8[v38];
  vint32m1_t v40 = __riscv_vle32_v_i32m1(v39, v15);
  size_t v41 = v15 * v9;
  size_t v42 = v10 + v41;
  int32_t* v43 = &v8[v42];
  vint32m1_t v44 = __riscv_vle32_v_i32m1(v43, v15);
  size_t v45 = v16 * v9;
  size_t v46 = v10 + v45;
  int32_t* v47 = &v8[v46];
  vint32m1_t v48 = __riscv_vle32_v_i32m1(v47, v15);
  size_t v49 = v17 * v9;
  size_t v50 = v10 + v49;
  int32_t* v51 = &v8[v50];
  vint32m1_t v52 = __riscv_vle32_v_i32m1(v51, v15);
  vint32m1_t v53;
  v53 = v24;
  vint32m1_t v54;
  v54 = v28;
  vint32m1_t v55;
  v55 = v32;
  vint32m1_t v56;
  v56 = v36;
  vint32m1_t v57;
  v57 = v40;
  vint32m1_t v58;
  v58 = v44;
  vint32m1_t v59;
  v59 = v48;
  vint32m1_t v60;
  v60 = v52;
  for (size_t i61 = v10; i61 < v1; i61 += v11) {
    size_t v62 = i61 * v4;
    size_t v63 = 5;
    int8_t* v64 = &v3[v62];
    vint8mf4_t v65 = __riscv_vle8_v_i8mf4(v64, v63);
    vint16mf2_t v66 = __riscv_vsext_vf2_i16mf2(v65, v63);
    size_t v67 = i61 * v6;
    size_t v68 = v67 + v11;
    size_t v69 = v67 + v12;
    size_t v70 = v67 + v13;
    size_t v71 = v67 + v14;
    size_t v72 = v67 + v15;
    size_t v73 = v67 + v16;
    size_t v74 = v67 + v17;
    int8_t v75 = v5[v67];
    int16_t v76 = (int16_t) v75;
    vint32m1_t v77 = v53;
    vint32m1_t v78 = __riscv_vwmacc_vx_i32m1(v77, v76, v66, v63);
    int8_t v79 = v5[v68];
    int16_t v80 = (int16_t) v79;
    vint32m1_t v81 = v54;
    vint32m1_t v82 = __riscv_vwmacc_vx_i32m1(v81, v80, v66, v63);
    int8_t v83 = v5[v69];
    int16_t v84 = (int16_t) v83;
    vint32m1_t v85 = v55;
    vint32m1_t v86 = __riscv_vwmacc_vx_i32m1(v85, v84, v66, v63);
    int8_t v87 = v5[v70];
    int16_t v88 = (int16_t) v87;
    vint32m1_t v89 = v56;
    vint32m1_t v90 = __riscv_vwmacc_vx_i32m1(v89, v88, v66, v63);
    int8_t v91 = v5[v71];
    int16_t v92 = (int16_t) v91;
    vint32m1_t v93 = v57;
    vint32m1_t v94 = __riscv_vwmacc_vx_i32m1(v93, v92, v66, v63);
    int8_t v95 = v5[v72];
    int16_t v96 = (int16_t) v95;
    vint32m1_t v97 = v58;
    vint32m1_t v98 = __riscv_vwmacc_vx_i32m1(v97, v96, v66, v63);
    int8_t v99 = v5[v73];
    int16_t v100 = (int16_t) v99;
    vint32m1_t v101 = v59;
    vint32m1_t v102 = __riscv_vwmacc_vx_i32m1(v101, v100, v66, v63);
    int8_t v103 = v5[v74];
    int16_t v104 = (int16_t) v103;
    vint32m1_t v105 = v60;
    vint32m1_t v106 = __riscv_vwmacc_vx_i32m1(v105, v104, v66, v63);
    v53 = v78;
    v54 = v82;
    v55 = v86;
    v56 = v90;
    v57 = v94;
    v58 = v98;
    v59 = v102;
    v60 = v106;
  }
  vint32m1_t v107 = v53;
  vint32m1_t v108 = v54;
  vint32m1_t v109 = v55;
  vint32m1_t v110 = v56;
  vint32m1_t v111 = v57;
  vint32m1_t v112 = v58;
  vint32m1_t v113 = v59;
  vint32m1_t v114 = v60;
  int32_t* v115 = &v8[v10];
  __riscv_vse32_v_i32m1(v115, v107, v15);
  int32_t* v116 = &v8[v9];
  __riscv_vse32_v_i32m1(v116, v108, v15);
  size_t v117 = v12 * v9;
  int32_t* v118 = &v8[v117];
  __riscv_vse32_v_i32m1(v118, v109, v15);
  size_t v119 = v13 * v9;
  int32_t* v120 = &v8[v119];
  __riscv_vse32_v_i32m1(v120, v110, v15);
  size_t v121 = v14 * v9;
  int32_t* v122 = &v8[v121];
  __riscv_vse32_v_i32m1(v122, v111, v15);
  size_t v123 = v15 * v9;
  int32_t* v124 = &v8[v123];
  __riscv_vse32_v_i32m1(v124, v112, v15);
  size_t v125 = v16 * v9;
  int32_t* v126 = &v8[v125];
  __riscv_vse32_v_i32m1(v126, v113, v15);
  size_t v127 = v17 * v9;
  int32_t* v128 = &v8[v127];
  __riscv_vse32_v_i32m1(v128, v114, v15);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_6x1_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 6
//   NR         : 1
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_6x1_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v22;
  v22 = v21;
  for (size_t i23 = v10; i23 < v1; i23 += v11) {
    size_t v24 = i23 * v4;
    size_t v25 = 6;
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
  __riscv_vse32_v_i32m1(v35, v34, v16);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_6x1_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 6
//   NR         : 1
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_6x1_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v16);
  vint32m1_t v25;
  v25 = v24;
  for (size_t i26 = v10; i26 < v1; i26 += v11) {
    size_t v27 = i26 * v4;
    size_t v28 = 6;
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
  __riscv_vse32_v_i32m1(v38, v37, v16);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_6x2_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 6
//   NR         : 2
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_6x2_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v22 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v23;
  v23 = v21;
  vint32m1_t v24;
  v24 = v22;
  for (size_t i25 = v10; i25 < v1; i25 += v11) {
    size_t v26 = i25 * v4;
    size_t v27 = 6;
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
  __riscv_vse32_v_i32m1(v43, v41, v16);
  int32_t* v44 = &v8[v9];
  __riscv_vse32_v_i32m1(v44, v42, v16);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_6x2_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 6
//   NR         : 2
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_6x2_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v16);
  size_t v25 = v11 * v9;
  size_t v26 = v10 + v25;
  int32_t* v27 = &v8[v26];
  vint32m1_t v28 = __riscv_vle32_v_i32m1(v27, v16);
  vint32m1_t v29;
  v29 = v24;
  vint32m1_t v30;
  v30 = v28;
  for (size_t i31 = v10; i31 < v1; i31 += v11) {
    size_t v32 = i31 * v4;
    size_t v33 = 6;
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
  __riscv_vse32_v_i32m1(v49, v47, v16);
  int32_t* v50 = &v8[v9];
  __riscv_vse32_v_i32m1(v50, v48, v16);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_6x3_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 6
//   NR         : 3
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_6x3_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v22 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v24;
  v24 = v21;
  vint32m1_t v25;
  v25 = v22;
  vint32m1_t v26;
  v26 = v23;
  for (size_t i27 = v10; i27 < v1; i27 += v11) {
    size_t v28 = i27 * v4;
    size_t v29 = 6;
    int8_t* v30 = &v3[v28];
    vint8mf4_t v31 = __riscv_vle8_v_i8mf4(v30, v29);
    vint16mf2_t v32 = __riscv_vsext_vf2_i16mf2(v31, v29);
    size_t v33 = i27 * v6;
    size_t v34 = v33 + v11;
    size_t v35 = v33 + v12;
    int8_t v36 = v5[v33];
    int16_t v37 = (int16_t) v36;
    vint32m1_t v38 = v24;
    vint32m1_t v39 = __riscv_vwmacc_vx_i32m1(v38, v37, v32, v29);
    int8_t v40 = v5[v34];
    int16_t v41 = (int16_t) v40;
    vint32m1_t v42 = v25;
    vint32m1_t v43 = __riscv_vwmacc_vx_i32m1(v42, v41, v32, v29);
    int8_t v44 = v5[v35];
    int16_t v45 = (int16_t) v44;
    vint32m1_t v46 = v26;
    vint32m1_t v47 = __riscv_vwmacc_vx_i32m1(v46, v45, v32, v29);
    v24 = v39;
    v25 = v43;
    v26 = v47;
  }
  vint32m1_t v48 = v24;
  vint32m1_t v49 = v25;
  vint32m1_t v50 = v26;
  int32_t* v51 = &v8[v10];
  __riscv_vse32_v_i32m1(v51, v48, v16);
  int32_t* v52 = &v8[v9];
  __riscv_vse32_v_i32m1(v52, v49, v16);
  size_t v53 = v12 * v9;
  int32_t* v54 = &v8[v53];
  __riscv_vse32_v_i32m1(v54, v50, v16);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_6x3_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 6
//   NR         : 3
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_6x3_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v16);
  size_t v25 = v11 * v9;
  size_t v26 = v10 + v25;
  int32_t* v27 = &v8[v26];
  vint32m1_t v28 = __riscv_vle32_v_i32m1(v27, v16);
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v16);
  vint32m1_t v33;
  v33 = v24;
  vint32m1_t v34;
  v34 = v28;
  vint32m1_t v35;
  v35 = v32;
  for (size_t i36 = v10; i36 < v1; i36 += v11) {
    size_t v37 = i36 * v4;
    size_t v38 = 6;
    int8_t* v39 = &v3[v37];
    vint8mf4_t v40 = __riscv_vle8_v_i8mf4(v39, v38);
    vint16mf2_t v41 = __riscv_vsext_vf2_i16mf2(v40, v38);
    size_t v42 = i36 * v6;
    size_t v43 = v42 + v11;
    size_t v44 = v42 + v12;
    int8_t v45 = v5[v42];
    int16_t v46 = (int16_t) v45;
    vint32m1_t v47 = v33;
    vint32m1_t v48 = __riscv_vwmacc_vx_i32m1(v47, v46, v41, v38);
    int8_t v49 = v5[v43];
    int16_t v50 = (int16_t) v49;
    vint32m1_t v51 = v34;
    vint32m1_t v52 = __riscv_vwmacc_vx_i32m1(v51, v50, v41, v38);
    int8_t v53 = v5[v44];
    int16_t v54 = (int16_t) v53;
    vint32m1_t v55 = v35;
    vint32m1_t v56 = __riscv_vwmacc_vx_i32m1(v55, v54, v41, v38);
    v33 = v48;
    v34 = v52;
    v35 = v56;
  }
  vint32m1_t v57 = v33;
  vint32m1_t v58 = v34;
  vint32m1_t v59 = v35;
  int32_t* v60 = &v8[v10];
  __riscv_vse32_v_i32m1(v60, v57, v16);
  int32_t* v61 = &v8[v9];
  __riscv_vse32_v_i32m1(v61, v58, v16);
  size_t v62 = v12 * v9;
  int32_t* v63 = &v8[v62];
  __riscv_vse32_v_i32m1(v63, v59, v16);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_6x4_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 6
//   NR         : 4
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_6x4_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v22 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v25;
  v25 = v21;
  vint32m1_t v26;
  v26 = v22;
  vint32m1_t v27;
  v27 = v23;
  vint32m1_t v28;
  v28 = v24;
  for (size_t i29 = v10; i29 < v1; i29 += v11) {
    size_t v30 = i29 * v4;
    size_t v31 = 6;
    int8_t* v32 = &v3[v30];
    vint8mf4_t v33 = __riscv_vle8_v_i8mf4(v32, v31);
    vint16mf2_t v34 = __riscv_vsext_vf2_i16mf2(v33, v31);
    size_t v35 = i29 * v6;
    size_t v36 = v35 + v11;
    size_t v37 = v35 + v12;
    size_t v38 = v35 + v13;
    int8_t v39 = v5[v35];
    int16_t v40 = (int16_t) v39;
    vint32m1_t v41 = v25;
    vint32m1_t v42 = __riscv_vwmacc_vx_i32m1(v41, v40, v34, v31);
    int8_t v43 = v5[v36];
    int16_t v44 = (int16_t) v43;
    vint32m1_t v45 = v26;
    vint32m1_t v46 = __riscv_vwmacc_vx_i32m1(v45, v44, v34, v31);
    int8_t v47 = v5[v37];
    int16_t v48 = (int16_t) v47;
    vint32m1_t v49 = v27;
    vint32m1_t v50 = __riscv_vwmacc_vx_i32m1(v49, v48, v34, v31);
    int8_t v51 = v5[v38];
    int16_t v52 = (int16_t) v51;
    vint32m1_t v53 = v28;
    vint32m1_t v54 = __riscv_vwmacc_vx_i32m1(v53, v52, v34, v31);
    v25 = v42;
    v26 = v46;
    v27 = v50;
    v28 = v54;
  }
  vint32m1_t v55 = v25;
  vint32m1_t v56 = v26;
  vint32m1_t v57 = v27;
  vint32m1_t v58 = v28;
  int32_t* v59 = &v8[v10];
  __riscv_vse32_v_i32m1(v59, v55, v16);
  int32_t* v60 = &v8[v9];
  __riscv_vse32_v_i32m1(v60, v56, v16);
  size_t v61 = v12 * v9;
  int32_t* v62 = &v8[v61];
  __riscv_vse32_v_i32m1(v62, v57, v16);
  size_t v63 = v13 * v9;
  int32_t* v64 = &v8[v63];
  __riscv_vse32_v_i32m1(v64, v58, v16);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_6x4_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 6
//   NR         : 4
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_6x4_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v16);
  size_t v25 = v11 * v9;
  size_t v26 = v10 + v25;
  int32_t* v27 = &v8[v26];
  vint32m1_t v28 = __riscv_vle32_v_i32m1(v27, v16);
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v16);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v16);
  vint32m1_t v37;
  v37 = v24;
  vint32m1_t v38;
  v38 = v28;
  vint32m1_t v39;
  v39 = v32;
  vint32m1_t v40;
  v40 = v36;
  for (size_t i41 = v10; i41 < v1; i41 += v11) {
    size_t v42 = i41 * v4;
    size_t v43 = 6;
    int8_t* v44 = &v3[v42];
    vint8mf4_t v45 = __riscv_vle8_v_i8mf4(v44, v43);
    vint16mf2_t v46 = __riscv_vsext_vf2_i16mf2(v45, v43);
    size_t v47 = i41 * v6;
    size_t v48 = v47 + v11;
    size_t v49 = v47 + v12;
    size_t v50 = v47 + v13;
    int8_t v51 = v5[v47];
    int16_t v52 = (int16_t) v51;
    vint32m1_t v53 = v37;
    vint32m1_t v54 = __riscv_vwmacc_vx_i32m1(v53, v52, v46, v43);
    int8_t v55 = v5[v48];
    int16_t v56 = (int16_t) v55;
    vint32m1_t v57 = v38;
    vint32m1_t v58 = __riscv_vwmacc_vx_i32m1(v57, v56, v46, v43);
    int8_t v59 = v5[v49];
    int16_t v60 = (int16_t) v59;
    vint32m1_t v61 = v39;
    vint32m1_t v62 = __riscv_vwmacc_vx_i32m1(v61, v60, v46, v43);
    int8_t v63 = v5[v50];
    int16_t v64 = (int16_t) v63;
    vint32m1_t v65 = v40;
    vint32m1_t v66 = __riscv_vwmacc_vx_i32m1(v65, v64, v46, v43);
    v37 = v54;
    v38 = v58;
    v39 = v62;
    v40 = v66;
  }
  vint32m1_t v67 = v37;
  vint32m1_t v68 = v38;
  vint32m1_t v69 = v39;
  vint32m1_t v70 = v40;
  int32_t* v71 = &v8[v10];
  __riscv_vse32_v_i32m1(v71, v67, v16);
  int32_t* v72 = &v8[v9];
  __riscv_vse32_v_i32m1(v72, v68, v16);
  size_t v73 = v12 * v9;
  int32_t* v74 = &v8[v73];
  __riscv_vse32_v_i32m1(v74, v69, v16);
  size_t v75 = v13 * v9;
  int32_t* v76 = &v8[v75];
  __riscv_vse32_v_i32m1(v76, v70, v16);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_6x5_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 6
//   NR         : 5
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_6x5_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v22 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v25 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v26;
  v26 = v21;
  vint32m1_t v27;
  v27 = v22;
  vint32m1_t v28;
  v28 = v23;
  vint32m1_t v29;
  v29 = v24;
  vint32m1_t v30;
  v30 = v25;
  for (size_t i31 = v10; i31 < v1; i31 += v11) {
    size_t v32 = i31 * v4;
    size_t v33 = 6;
    int8_t* v34 = &v3[v32];
    vint8mf4_t v35 = __riscv_vle8_v_i8mf4(v34, v33);
    vint16mf2_t v36 = __riscv_vsext_vf2_i16mf2(v35, v33);
    size_t v37 = i31 * v6;
    size_t v38 = v37 + v11;
    size_t v39 = v37 + v12;
    size_t v40 = v37 + v13;
    size_t v41 = v37 + v14;
    int8_t v42 = v5[v37];
    int16_t v43 = (int16_t) v42;
    vint32m1_t v44 = v26;
    vint32m1_t v45 = __riscv_vwmacc_vx_i32m1(v44, v43, v36, v33);
    int8_t v46 = v5[v38];
    int16_t v47 = (int16_t) v46;
    vint32m1_t v48 = v27;
    vint32m1_t v49 = __riscv_vwmacc_vx_i32m1(v48, v47, v36, v33);
    int8_t v50 = v5[v39];
    int16_t v51 = (int16_t) v50;
    vint32m1_t v52 = v28;
    vint32m1_t v53 = __riscv_vwmacc_vx_i32m1(v52, v51, v36, v33);
    int8_t v54 = v5[v40];
    int16_t v55 = (int16_t) v54;
    vint32m1_t v56 = v29;
    vint32m1_t v57 = __riscv_vwmacc_vx_i32m1(v56, v55, v36, v33);
    int8_t v58 = v5[v41];
    int16_t v59 = (int16_t) v58;
    vint32m1_t v60 = v30;
    vint32m1_t v61 = __riscv_vwmacc_vx_i32m1(v60, v59, v36, v33);
    v26 = v45;
    v27 = v49;
    v28 = v53;
    v29 = v57;
    v30 = v61;
  }
  vint32m1_t v62 = v26;
  vint32m1_t v63 = v27;
  vint32m1_t v64 = v28;
  vint32m1_t v65 = v29;
  vint32m1_t v66 = v30;
  int32_t* v67 = &v8[v10];
  __riscv_vse32_v_i32m1(v67, v62, v16);
  int32_t* v68 = &v8[v9];
  __riscv_vse32_v_i32m1(v68, v63, v16);
  size_t v69 = v12 * v9;
  int32_t* v70 = &v8[v69];
  __riscv_vse32_v_i32m1(v70, v64, v16);
  size_t v71 = v13 * v9;
  int32_t* v72 = &v8[v71];
  __riscv_vse32_v_i32m1(v72, v65, v16);
  size_t v73 = v14 * v9;
  int32_t* v74 = &v8[v73];
  __riscv_vse32_v_i32m1(v74, v66, v16);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_6x5_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 6
//   NR         : 5
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_6x5_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v16);
  size_t v25 = v11 * v9;
  size_t v26 = v10 + v25;
  int32_t* v27 = &v8[v26];
  vint32m1_t v28 = __riscv_vle32_v_i32m1(v27, v16);
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v16);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v16);
  size_t v37 = v14 * v9;
  size_t v38 = v10 + v37;
  int32_t* v39 = &v8[v38];
  vint32m1_t v40 = __riscv_vle32_v_i32m1(v39, v16);
  vint32m1_t v41;
  v41 = v24;
  vint32m1_t v42;
  v42 = v28;
  vint32m1_t v43;
  v43 = v32;
  vint32m1_t v44;
  v44 = v36;
  vint32m1_t v45;
  v45 = v40;
  for (size_t i46 = v10; i46 < v1; i46 += v11) {
    size_t v47 = i46 * v4;
    size_t v48 = 6;
    int8_t* v49 = &v3[v47];
    vint8mf4_t v50 = __riscv_vle8_v_i8mf4(v49, v48);
    vint16mf2_t v51 = __riscv_vsext_vf2_i16mf2(v50, v48);
    size_t v52 = i46 * v6;
    size_t v53 = v52 + v11;
    size_t v54 = v52 + v12;
    size_t v55 = v52 + v13;
    size_t v56 = v52 + v14;
    int8_t v57 = v5[v52];
    int16_t v58 = (int16_t) v57;
    vint32m1_t v59 = v41;
    vint32m1_t v60 = __riscv_vwmacc_vx_i32m1(v59, v58, v51, v48);
    int8_t v61 = v5[v53];
    int16_t v62 = (int16_t) v61;
    vint32m1_t v63 = v42;
    vint32m1_t v64 = __riscv_vwmacc_vx_i32m1(v63, v62, v51, v48);
    int8_t v65 = v5[v54];
    int16_t v66 = (int16_t) v65;
    vint32m1_t v67 = v43;
    vint32m1_t v68 = __riscv_vwmacc_vx_i32m1(v67, v66, v51, v48);
    int8_t v69 = v5[v55];
    int16_t v70 = (int16_t) v69;
    vint32m1_t v71 = v44;
    vint32m1_t v72 = __riscv_vwmacc_vx_i32m1(v71, v70, v51, v48);
    int8_t v73 = v5[v56];
    int16_t v74 = (int16_t) v73;
    vint32m1_t v75 = v45;
    vint32m1_t v76 = __riscv_vwmacc_vx_i32m1(v75, v74, v51, v48);
    v41 = v60;
    v42 = v64;
    v43 = v68;
    v44 = v72;
    v45 = v76;
  }
  vint32m1_t v77 = v41;
  vint32m1_t v78 = v42;
  vint32m1_t v79 = v43;
  vint32m1_t v80 = v44;
  vint32m1_t v81 = v45;
  int32_t* v82 = &v8[v10];
  __riscv_vse32_v_i32m1(v82, v77, v16);
  int32_t* v83 = &v8[v9];
  __riscv_vse32_v_i32m1(v83, v78, v16);
  size_t v84 = v12 * v9;
  int32_t* v85 = &v8[v84];
  __riscv_vse32_v_i32m1(v85, v79, v16);
  size_t v86 = v13 * v9;
  int32_t* v87 = &v8[v86];
  __riscv_vse32_v_i32m1(v87, v80, v16);
  size_t v88 = v14 * v9;
  int32_t* v89 = &v8[v88];
  __riscv_vse32_v_i32m1(v89, v81, v16);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_6x6_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 6
//   NR         : 6
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_6x6_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v22 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v25 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v26 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v27;
  v27 = v21;
  vint32m1_t v28;
  v28 = v22;
  vint32m1_t v29;
  v29 = v23;
  vint32m1_t v30;
  v30 = v24;
  vint32m1_t v31;
  v31 = v25;
  vint32m1_t v32;
  v32 = v26;
  for (size_t i33 = v10; i33 < v1; i33 += v11) {
    size_t v34 = i33 * v4;
    size_t v35 = 6;
    int8_t* v36 = &v3[v34];
    vint8mf4_t v37 = __riscv_vle8_v_i8mf4(v36, v35);
    vint16mf2_t v38 = __riscv_vsext_vf2_i16mf2(v37, v35);
    size_t v39 = i33 * v6;
    size_t v40 = v39 + v11;
    size_t v41 = v39 + v12;
    size_t v42 = v39 + v13;
    size_t v43 = v39 + v14;
    size_t v44 = v39 + v15;
    int8_t v45 = v5[v39];
    int16_t v46 = (int16_t) v45;
    vint32m1_t v47 = v27;
    vint32m1_t v48 = __riscv_vwmacc_vx_i32m1(v47, v46, v38, v35);
    int8_t v49 = v5[v40];
    int16_t v50 = (int16_t) v49;
    vint32m1_t v51 = v28;
    vint32m1_t v52 = __riscv_vwmacc_vx_i32m1(v51, v50, v38, v35);
    int8_t v53 = v5[v41];
    int16_t v54 = (int16_t) v53;
    vint32m1_t v55 = v29;
    vint32m1_t v56 = __riscv_vwmacc_vx_i32m1(v55, v54, v38, v35);
    int8_t v57 = v5[v42];
    int16_t v58 = (int16_t) v57;
    vint32m1_t v59 = v30;
    vint32m1_t v60 = __riscv_vwmacc_vx_i32m1(v59, v58, v38, v35);
    int8_t v61 = v5[v43];
    int16_t v62 = (int16_t) v61;
    vint32m1_t v63 = v31;
    vint32m1_t v64 = __riscv_vwmacc_vx_i32m1(v63, v62, v38, v35);
    int8_t v65 = v5[v44];
    int16_t v66 = (int16_t) v65;
    vint32m1_t v67 = v32;
    vint32m1_t v68 = __riscv_vwmacc_vx_i32m1(v67, v66, v38, v35);
    v27 = v48;
    v28 = v52;
    v29 = v56;
    v30 = v60;
    v31 = v64;
    v32 = v68;
  }
  vint32m1_t v69 = v27;
  vint32m1_t v70 = v28;
  vint32m1_t v71 = v29;
  vint32m1_t v72 = v30;
  vint32m1_t v73 = v31;
  vint32m1_t v74 = v32;
  int32_t* v75 = &v8[v10];
  __riscv_vse32_v_i32m1(v75, v69, v16);
  int32_t* v76 = &v8[v9];
  __riscv_vse32_v_i32m1(v76, v70, v16);
  size_t v77 = v12 * v9;
  int32_t* v78 = &v8[v77];
  __riscv_vse32_v_i32m1(v78, v71, v16);
  size_t v79 = v13 * v9;
  int32_t* v80 = &v8[v79];
  __riscv_vse32_v_i32m1(v80, v72, v16);
  size_t v81 = v14 * v9;
  int32_t* v82 = &v8[v81];
  __riscv_vse32_v_i32m1(v82, v73, v16);
  size_t v83 = v15 * v9;
  int32_t* v84 = &v8[v83];
  __riscv_vse32_v_i32m1(v84, v74, v16);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_6x6_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 6
//   NR         : 6
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_6x6_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v16);
  size_t v25 = v11 * v9;
  size_t v26 = v10 + v25;
  int32_t* v27 = &v8[v26];
  vint32m1_t v28 = __riscv_vle32_v_i32m1(v27, v16);
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v16);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v16);
  size_t v37 = v14 * v9;
  size_t v38 = v10 + v37;
  int32_t* v39 = &v8[v38];
  vint32m1_t v40 = __riscv_vle32_v_i32m1(v39, v16);
  size_t v41 = v15 * v9;
  size_t v42 = v10 + v41;
  int32_t* v43 = &v8[v42];
  vint32m1_t v44 = __riscv_vle32_v_i32m1(v43, v16);
  vint32m1_t v45;
  v45 = v24;
  vint32m1_t v46;
  v46 = v28;
  vint32m1_t v47;
  v47 = v32;
  vint32m1_t v48;
  v48 = v36;
  vint32m1_t v49;
  v49 = v40;
  vint32m1_t v50;
  v50 = v44;
  for (size_t i51 = v10; i51 < v1; i51 += v11) {
    size_t v52 = i51 * v4;
    size_t v53 = 6;
    int8_t* v54 = &v3[v52];
    vint8mf4_t v55 = __riscv_vle8_v_i8mf4(v54, v53);
    vint16mf2_t v56 = __riscv_vsext_vf2_i16mf2(v55, v53);
    size_t v57 = i51 * v6;
    size_t v58 = v57 + v11;
    size_t v59 = v57 + v12;
    size_t v60 = v57 + v13;
    size_t v61 = v57 + v14;
    size_t v62 = v57 + v15;
    int8_t v63 = v5[v57];
    int16_t v64 = (int16_t) v63;
    vint32m1_t v65 = v45;
    vint32m1_t v66 = __riscv_vwmacc_vx_i32m1(v65, v64, v56, v53);
    int8_t v67 = v5[v58];
    int16_t v68 = (int16_t) v67;
    vint32m1_t v69 = v46;
    vint32m1_t v70 = __riscv_vwmacc_vx_i32m1(v69, v68, v56, v53);
    int8_t v71 = v5[v59];
    int16_t v72 = (int16_t) v71;
    vint32m1_t v73 = v47;
    vint32m1_t v74 = __riscv_vwmacc_vx_i32m1(v73, v72, v56, v53);
    int8_t v75 = v5[v60];
    int16_t v76 = (int16_t) v75;
    vint32m1_t v77 = v48;
    vint32m1_t v78 = __riscv_vwmacc_vx_i32m1(v77, v76, v56, v53);
    int8_t v79 = v5[v61];
    int16_t v80 = (int16_t) v79;
    vint32m1_t v81 = v49;
    vint32m1_t v82 = __riscv_vwmacc_vx_i32m1(v81, v80, v56, v53);
    int8_t v83 = v5[v62];
    int16_t v84 = (int16_t) v83;
    vint32m1_t v85 = v50;
    vint32m1_t v86 = __riscv_vwmacc_vx_i32m1(v85, v84, v56, v53);
    v45 = v66;
    v46 = v70;
    v47 = v74;
    v48 = v78;
    v49 = v82;
    v50 = v86;
  }
  vint32m1_t v87 = v45;
  vint32m1_t v88 = v46;
  vint32m1_t v89 = v47;
  vint32m1_t v90 = v48;
  vint32m1_t v91 = v49;
  vint32m1_t v92 = v50;
  int32_t* v93 = &v8[v10];
  __riscv_vse32_v_i32m1(v93, v87, v16);
  int32_t* v94 = &v8[v9];
  __riscv_vse32_v_i32m1(v94, v88, v16);
  size_t v95 = v12 * v9;
  int32_t* v96 = &v8[v95];
  __riscv_vse32_v_i32m1(v96, v89, v16);
  size_t v97 = v13 * v9;
  int32_t* v98 = &v8[v97];
  __riscv_vse32_v_i32m1(v98, v90, v16);
  size_t v99 = v14 * v9;
  int32_t* v100 = &v8[v99];
  __riscv_vse32_v_i32m1(v100, v91, v16);
  size_t v101 = v15 * v9;
  int32_t* v102 = &v8[v101];
  __riscv_vse32_v_i32m1(v102, v92, v16);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_6x7_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 6
//   NR         : 7
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_6x7_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v22 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v25 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v26 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v27 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v28;
  v28 = v21;
  vint32m1_t v29;
  v29 = v22;
  vint32m1_t v30;
  v30 = v23;
  vint32m1_t v31;
  v31 = v24;
  vint32m1_t v32;
  v32 = v25;
  vint32m1_t v33;
  v33 = v26;
  vint32m1_t v34;
  v34 = v27;
  for (size_t i35 = v10; i35 < v1; i35 += v11) {
    size_t v36 = i35 * v4;
    size_t v37 = 6;
    int8_t* v38 = &v3[v36];
    vint8mf4_t v39 = __riscv_vle8_v_i8mf4(v38, v37);
    vint16mf2_t v40 = __riscv_vsext_vf2_i16mf2(v39, v37);
    size_t v41 = i35 * v6;
    size_t v42 = v41 + v11;
    size_t v43 = v41 + v12;
    size_t v44 = v41 + v13;
    size_t v45 = v41 + v14;
    size_t v46 = v41 + v15;
    size_t v47 = v41 + v16;
    int8_t v48 = v5[v41];
    int16_t v49 = (int16_t) v48;
    vint32m1_t v50 = v28;
    vint32m1_t v51 = __riscv_vwmacc_vx_i32m1(v50, v49, v40, v37);
    int8_t v52 = v5[v42];
    int16_t v53 = (int16_t) v52;
    vint32m1_t v54 = v29;
    vint32m1_t v55 = __riscv_vwmacc_vx_i32m1(v54, v53, v40, v37);
    int8_t v56 = v5[v43];
    int16_t v57 = (int16_t) v56;
    vint32m1_t v58 = v30;
    vint32m1_t v59 = __riscv_vwmacc_vx_i32m1(v58, v57, v40, v37);
    int8_t v60 = v5[v44];
    int16_t v61 = (int16_t) v60;
    vint32m1_t v62 = v31;
    vint32m1_t v63 = __riscv_vwmacc_vx_i32m1(v62, v61, v40, v37);
    int8_t v64 = v5[v45];
    int16_t v65 = (int16_t) v64;
    vint32m1_t v66 = v32;
    vint32m1_t v67 = __riscv_vwmacc_vx_i32m1(v66, v65, v40, v37);
    int8_t v68 = v5[v46];
    int16_t v69 = (int16_t) v68;
    vint32m1_t v70 = v33;
    vint32m1_t v71 = __riscv_vwmacc_vx_i32m1(v70, v69, v40, v37);
    int8_t v72 = v5[v47];
    int16_t v73 = (int16_t) v72;
    vint32m1_t v74 = v34;
    vint32m1_t v75 = __riscv_vwmacc_vx_i32m1(v74, v73, v40, v37);
    v28 = v51;
    v29 = v55;
    v30 = v59;
    v31 = v63;
    v32 = v67;
    v33 = v71;
    v34 = v75;
  }
  vint32m1_t v76 = v28;
  vint32m1_t v77 = v29;
  vint32m1_t v78 = v30;
  vint32m1_t v79 = v31;
  vint32m1_t v80 = v32;
  vint32m1_t v81 = v33;
  vint32m1_t v82 = v34;
  int32_t* v83 = &v8[v10];
  __riscv_vse32_v_i32m1(v83, v76, v16);
  int32_t* v84 = &v8[v9];
  __riscv_vse32_v_i32m1(v84, v77, v16);
  size_t v85 = v12 * v9;
  int32_t* v86 = &v8[v85];
  __riscv_vse32_v_i32m1(v86, v78, v16);
  size_t v87 = v13 * v9;
  int32_t* v88 = &v8[v87];
  __riscv_vse32_v_i32m1(v88, v79, v16);
  size_t v89 = v14 * v9;
  int32_t* v90 = &v8[v89];
  __riscv_vse32_v_i32m1(v90, v80, v16);
  size_t v91 = v15 * v9;
  int32_t* v92 = &v8[v91];
  __riscv_vse32_v_i32m1(v92, v81, v16);
  size_t v93 = v16 * v9;
  int32_t* v94 = &v8[v93];
  __riscv_vse32_v_i32m1(v94, v82, v16);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_6x7_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 6
//   NR         : 7
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_6x7_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v16);
  size_t v25 = v11 * v9;
  size_t v26 = v10 + v25;
  int32_t* v27 = &v8[v26];
  vint32m1_t v28 = __riscv_vle32_v_i32m1(v27, v16);
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v16);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v16);
  size_t v37 = v14 * v9;
  size_t v38 = v10 + v37;
  int32_t* v39 = &v8[v38];
  vint32m1_t v40 = __riscv_vle32_v_i32m1(v39, v16);
  size_t v41 = v15 * v9;
  size_t v42 = v10 + v41;
  int32_t* v43 = &v8[v42];
  vint32m1_t v44 = __riscv_vle32_v_i32m1(v43, v16);
  size_t v45 = v16 * v9;
  size_t v46 = v10 + v45;
  int32_t* v47 = &v8[v46];
  vint32m1_t v48 = __riscv_vle32_v_i32m1(v47, v16);
  vint32m1_t v49;
  v49 = v24;
  vint32m1_t v50;
  v50 = v28;
  vint32m1_t v51;
  v51 = v32;
  vint32m1_t v52;
  v52 = v36;
  vint32m1_t v53;
  v53 = v40;
  vint32m1_t v54;
  v54 = v44;
  vint32m1_t v55;
  v55 = v48;
  for (size_t i56 = v10; i56 < v1; i56 += v11) {
    size_t v57 = i56 * v4;
    size_t v58 = 6;
    int8_t* v59 = &v3[v57];
    vint8mf4_t v60 = __riscv_vle8_v_i8mf4(v59, v58);
    vint16mf2_t v61 = __riscv_vsext_vf2_i16mf2(v60, v58);
    size_t v62 = i56 * v6;
    size_t v63 = v62 + v11;
    size_t v64 = v62 + v12;
    size_t v65 = v62 + v13;
    size_t v66 = v62 + v14;
    size_t v67 = v62 + v15;
    size_t v68 = v62 + v16;
    int8_t v69 = v5[v62];
    int16_t v70 = (int16_t) v69;
    vint32m1_t v71 = v49;
    vint32m1_t v72 = __riscv_vwmacc_vx_i32m1(v71, v70, v61, v58);
    int8_t v73 = v5[v63];
    int16_t v74 = (int16_t) v73;
    vint32m1_t v75 = v50;
    vint32m1_t v76 = __riscv_vwmacc_vx_i32m1(v75, v74, v61, v58);
    int8_t v77 = v5[v64];
    int16_t v78 = (int16_t) v77;
    vint32m1_t v79 = v51;
    vint32m1_t v80 = __riscv_vwmacc_vx_i32m1(v79, v78, v61, v58);
    int8_t v81 = v5[v65];
    int16_t v82 = (int16_t) v81;
    vint32m1_t v83 = v52;
    vint32m1_t v84 = __riscv_vwmacc_vx_i32m1(v83, v82, v61, v58);
    int8_t v85 = v5[v66];
    int16_t v86 = (int16_t) v85;
    vint32m1_t v87 = v53;
    vint32m1_t v88 = __riscv_vwmacc_vx_i32m1(v87, v86, v61, v58);
    int8_t v89 = v5[v67];
    int16_t v90 = (int16_t) v89;
    vint32m1_t v91 = v54;
    vint32m1_t v92 = __riscv_vwmacc_vx_i32m1(v91, v90, v61, v58);
    int8_t v93 = v5[v68];
    int16_t v94 = (int16_t) v93;
    vint32m1_t v95 = v55;
    vint32m1_t v96 = __riscv_vwmacc_vx_i32m1(v95, v94, v61, v58);
    v49 = v72;
    v50 = v76;
    v51 = v80;
    v52 = v84;
    v53 = v88;
    v54 = v92;
    v55 = v96;
  }
  vint32m1_t v97 = v49;
  vint32m1_t v98 = v50;
  vint32m1_t v99 = v51;
  vint32m1_t v100 = v52;
  vint32m1_t v101 = v53;
  vint32m1_t v102 = v54;
  vint32m1_t v103 = v55;
  int32_t* v104 = &v8[v10];
  __riscv_vse32_v_i32m1(v104, v97, v16);
  int32_t* v105 = &v8[v9];
  __riscv_vse32_v_i32m1(v105, v98, v16);
  size_t v106 = v12 * v9;
  int32_t* v107 = &v8[v106];
  __riscv_vse32_v_i32m1(v107, v99, v16);
  size_t v108 = v13 * v9;
  int32_t* v109 = &v8[v108];
  __riscv_vse32_v_i32m1(v109, v100, v16);
  size_t v110 = v14 * v9;
  int32_t* v111 = &v8[v110];
  __riscv_vse32_v_i32m1(v111, v101, v16);
  size_t v112 = v15 * v9;
  int32_t* v113 = &v8[v112];
  __riscv_vse32_v_i32m1(v113, v102, v16);
  size_t v114 = v16 * v9;
  int32_t* v115 = &v8[v114];
  __riscv_vse32_v_i32m1(v115, v103, v16);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_6x8_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 6
//   NR         : 8
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_6x8_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v22 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v25 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v26 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v27 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v28 = __riscv_vmv_v_x_i32m1(v20, v16);
  vint32m1_t v29;
  v29 = v21;
  vint32m1_t v30;
  v30 = v22;
  vint32m1_t v31;
  v31 = v23;
  vint32m1_t v32;
  v32 = v24;
  vint32m1_t v33;
  v33 = v25;
  vint32m1_t v34;
  v34 = v26;
  vint32m1_t v35;
  v35 = v27;
  vint32m1_t v36;
  v36 = v28;
  for (size_t i37 = v10; i37 < v1; i37 += v11) {
    size_t v38 = i37 * v4;
    size_t v39 = 6;
    int8_t* v40 = &v3[v38];
    vint8mf4_t v41 = __riscv_vle8_v_i8mf4(v40, v39);
    vint16mf2_t v42 = __riscv_vsext_vf2_i16mf2(v41, v39);
    size_t v43 = i37 * v6;
    size_t v44 = v43 + v11;
    size_t v45 = v43 + v12;
    size_t v46 = v43 + v13;
    size_t v47 = v43 + v14;
    size_t v48 = v43 + v15;
    size_t v49 = v43 + v16;
    size_t v50 = v43 + v17;
    int8_t v51 = v5[v43];
    int16_t v52 = (int16_t) v51;
    vint32m1_t v53 = v29;
    vint32m1_t v54 = __riscv_vwmacc_vx_i32m1(v53, v52, v42, v39);
    int8_t v55 = v5[v44];
    int16_t v56 = (int16_t) v55;
    vint32m1_t v57 = v30;
    vint32m1_t v58 = __riscv_vwmacc_vx_i32m1(v57, v56, v42, v39);
    int8_t v59 = v5[v45];
    int16_t v60 = (int16_t) v59;
    vint32m1_t v61 = v31;
    vint32m1_t v62 = __riscv_vwmacc_vx_i32m1(v61, v60, v42, v39);
    int8_t v63 = v5[v46];
    int16_t v64 = (int16_t) v63;
    vint32m1_t v65 = v32;
    vint32m1_t v66 = __riscv_vwmacc_vx_i32m1(v65, v64, v42, v39);
    int8_t v67 = v5[v47];
    int16_t v68 = (int16_t) v67;
    vint32m1_t v69 = v33;
    vint32m1_t v70 = __riscv_vwmacc_vx_i32m1(v69, v68, v42, v39);
    int8_t v71 = v5[v48];
    int16_t v72 = (int16_t) v71;
    vint32m1_t v73 = v34;
    vint32m1_t v74 = __riscv_vwmacc_vx_i32m1(v73, v72, v42, v39);
    int8_t v75 = v5[v49];
    int16_t v76 = (int16_t) v75;
    vint32m1_t v77 = v35;
    vint32m1_t v78 = __riscv_vwmacc_vx_i32m1(v77, v76, v42, v39);
    int8_t v79 = v5[v50];
    int16_t v80 = (int16_t) v79;
    vint32m1_t v81 = v36;
    vint32m1_t v82 = __riscv_vwmacc_vx_i32m1(v81, v80, v42, v39);
    v29 = v54;
    v30 = v58;
    v31 = v62;
    v32 = v66;
    v33 = v70;
    v34 = v74;
    v35 = v78;
    v36 = v82;
  }
  vint32m1_t v83 = v29;
  vint32m1_t v84 = v30;
  vint32m1_t v85 = v31;
  vint32m1_t v86 = v32;
  vint32m1_t v87 = v33;
  vint32m1_t v88 = v34;
  vint32m1_t v89 = v35;
  vint32m1_t v90 = v36;
  int32_t* v91 = &v8[v10];
  __riscv_vse32_v_i32m1(v91, v83, v16);
  int32_t* v92 = &v8[v9];
  __riscv_vse32_v_i32m1(v92, v84, v16);
  size_t v93 = v12 * v9;
  int32_t* v94 = &v8[v93];
  __riscv_vse32_v_i32m1(v94, v85, v16);
  size_t v95 = v13 * v9;
  int32_t* v96 = &v8[v95];
  __riscv_vse32_v_i32m1(v96, v86, v16);
  size_t v97 = v14 * v9;
  int32_t* v98 = &v8[v97];
  __riscv_vse32_v_i32m1(v98, v87, v16);
  size_t v99 = v15 * v9;
  int32_t* v100 = &v8[v99];
  __riscv_vse32_v_i32m1(v100, v88, v16);
  size_t v101 = v16 * v9;
  int32_t* v102 = &v8[v101];
  __riscv_vse32_v_i32m1(v102, v89, v16);
  size_t v103 = v17 * v9;
  int32_t* v104 = &v8[v103];
  __riscv_vse32_v_i32m1(v104, v90, v16);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_6x8_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 6
//   NR         : 8
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_6x8_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v16);
  size_t v25 = v11 * v9;
  size_t v26 = v10 + v25;
  int32_t* v27 = &v8[v26];
  vint32m1_t v28 = __riscv_vle32_v_i32m1(v27, v16);
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v16);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v16);
  size_t v37 = v14 * v9;
  size_t v38 = v10 + v37;
  int32_t* v39 = &v8[v38];
  vint32m1_t v40 = __riscv_vle32_v_i32m1(v39, v16);
  size_t v41 = v15 * v9;
  size_t v42 = v10 + v41;
  int32_t* v43 = &v8[v42];
  vint32m1_t v44 = __riscv_vle32_v_i32m1(v43, v16);
  size_t v45 = v16 * v9;
  size_t v46 = v10 + v45;
  int32_t* v47 = &v8[v46];
  vint32m1_t v48 = __riscv_vle32_v_i32m1(v47, v16);
  size_t v49 = v17 * v9;
  size_t v50 = v10 + v49;
  int32_t* v51 = &v8[v50];
  vint32m1_t v52 = __riscv_vle32_v_i32m1(v51, v16);
  vint32m1_t v53;
  v53 = v24;
  vint32m1_t v54;
  v54 = v28;
  vint32m1_t v55;
  v55 = v32;
  vint32m1_t v56;
  v56 = v36;
  vint32m1_t v57;
  v57 = v40;
  vint32m1_t v58;
  v58 = v44;
  vint32m1_t v59;
  v59 = v48;
  vint32m1_t v60;
  v60 = v52;
  for (size_t i61 = v10; i61 < v1; i61 += v11) {
    size_t v62 = i61 * v4;
    size_t v63 = 6;
    int8_t* v64 = &v3[v62];
    vint8mf4_t v65 = __riscv_vle8_v_i8mf4(v64, v63);
    vint16mf2_t v66 = __riscv_vsext_vf2_i16mf2(v65, v63);
    size_t v67 = i61 * v6;
    size_t v68 = v67 + v11;
    size_t v69 = v67 + v12;
    size_t v70 = v67 + v13;
    size_t v71 = v67 + v14;
    size_t v72 = v67 + v15;
    size_t v73 = v67 + v16;
    size_t v74 = v67 + v17;
    int8_t v75 = v5[v67];
    int16_t v76 = (int16_t) v75;
    vint32m1_t v77 = v53;
    vint32m1_t v78 = __riscv_vwmacc_vx_i32m1(v77, v76, v66, v63);
    int8_t v79 = v5[v68];
    int16_t v80 = (int16_t) v79;
    vint32m1_t v81 = v54;
    vint32m1_t v82 = __riscv_vwmacc_vx_i32m1(v81, v80, v66, v63);
    int8_t v83 = v5[v69];
    int16_t v84 = (int16_t) v83;
    vint32m1_t v85 = v55;
    vint32m1_t v86 = __riscv_vwmacc_vx_i32m1(v85, v84, v66, v63);
    int8_t v87 = v5[v70];
    int16_t v88 = (int16_t) v87;
    vint32m1_t v89 = v56;
    vint32m1_t v90 = __riscv_vwmacc_vx_i32m1(v89, v88, v66, v63);
    int8_t v91 = v5[v71];
    int16_t v92 = (int16_t) v91;
    vint32m1_t v93 = v57;
    vint32m1_t v94 = __riscv_vwmacc_vx_i32m1(v93, v92, v66, v63);
    int8_t v95 = v5[v72];
    int16_t v96 = (int16_t) v95;
    vint32m1_t v97 = v58;
    vint32m1_t v98 = __riscv_vwmacc_vx_i32m1(v97, v96, v66, v63);
    int8_t v99 = v5[v73];
    int16_t v100 = (int16_t) v99;
    vint32m1_t v101 = v59;
    vint32m1_t v102 = __riscv_vwmacc_vx_i32m1(v101, v100, v66, v63);
    int8_t v103 = v5[v74];
    int16_t v104 = (int16_t) v103;
    vint32m1_t v105 = v60;
    vint32m1_t v106 = __riscv_vwmacc_vx_i32m1(v105, v104, v66, v63);
    v53 = v78;
    v54 = v82;
    v55 = v86;
    v56 = v90;
    v57 = v94;
    v58 = v98;
    v59 = v102;
    v60 = v106;
  }
  vint32m1_t v107 = v53;
  vint32m1_t v108 = v54;
  vint32m1_t v109 = v55;
  vint32m1_t v110 = v56;
  vint32m1_t v111 = v57;
  vint32m1_t v112 = v58;
  vint32m1_t v113 = v59;
  vint32m1_t v114 = v60;
  int32_t* v115 = &v8[v10];
  __riscv_vse32_v_i32m1(v115, v107, v16);
  int32_t* v116 = &v8[v9];
  __riscv_vse32_v_i32m1(v116, v108, v16);
  size_t v117 = v12 * v9;
  int32_t* v118 = &v8[v117];
  __riscv_vse32_v_i32m1(v118, v109, v16);
  size_t v119 = v13 * v9;
  int32_t* v120 = &v8[v119];
  __riscv_vse32_v_i32m1(v120, v110, v16);
  size_t v121 = v14 * v9;
  int32_t* v122 = &v8[v121];
  __riscv_vse32_v_i32m1(v122, v111, v16);
  size_t v123 = v15 * v9;
  int32_t* v124 = &v8[v123];
  __riscv_vse32_v_i32m1(v124, v112, v16);
  size_t v125 = v16 * v9;
  int32_t* v126 = &v8[v125];
  __riscv_vse32_v_i32m1(v126, v113, v16);
  size_t v127 = v17 * v9;
  int32_t* v128 = &v8[v127];
  __riscv_vse32_v_i32m1(v128, v114, v16);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_7x1_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 7
//   NR         : 1
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_7x1_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v22;
  v22 = v21;
  for (size_t i23 = v10; i23 < v1; i23 += v11) {
    size_t v24 = i23 * v4;
    size_t v25 = 7;
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
  __riscv_vse32_v_i32m1(v35, v34, v17);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_7x1_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 7
//   NR         : 1
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_7x1_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v17);
  vint32m1_t v25;
  v25 = v24;
  for (size_t i26 = v10; i26 < v1; i26 += v11) {
    size_t v27 = i26 * v4;
    size_t v28 = 7;
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
  __riscv_vse32_v_i32m1(v38, v37, v17);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_7x2_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 7
//   NR         : 2
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_7x2_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v22 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v23;
  v23 = v21;
  vint32m1_t v24;
  v24 = v22;
  for (size_t i25 = v10; i25 < v1; i25 += v11) {
    size_t v26 = i25 * v4;
    size_t v27 = 7;
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
  __riscv_vse32_v_i32m1(v43, v41, v17);
  int32_t* v44 = &v8[v9];
  __riscv_vse32_v_i32m1(v44, v42, v17);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_7x2_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 7
//   NR         : 2
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_7x2_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v17);
  size_t v25 = v11 * v9;
  size_t v26 = v10 + v25;
  int32_t* v27 = &v8[v26];
  vint32m1_t v28 = __riscv_vle32_v_i32m1(v27, v17);
  vint32m1_t v29;
  v29 = v24;
  vint32m1_t v30;
  v30 = v28;
  for (size_t i31 = v10; i31 < v1; i31 += v11) {
    size_t v32 = i31 * v4;
    size_t v33 = 7;
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
  __riscv_vse32_v_i32m1(v49, v47, v17);
  int32_t* v50 = &v8[v9];
  __riscv_vse32_v_i32m1(v50, v48, v17);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_7x3_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 7
//   NR         : 3
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_7x3_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v22 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v24;
  v24 = v21;
  vint32m1_t v25;
  v25 = v22;
  vint32m1_t v26;
  v26 = v23;
  for (size_t i27 = v10; i27 < v1; i27 += v11) {
    size_t v28 = i27 * v4;
    size_t v29 = 7;
    int8_t* v30 = &v3[v28];
    vint8mf4_t v31 = __riscv_vle8_v_i8mf4(v30, v29);
    vint16mf2_t v32 = __riscv_vsext_vf2_i16mf2(v31, v29);
    size_t v33 = i27 * v6;
    size_t v34 = v33 + v11;
    size_t v35 = v33 + v12;
    int8_t v36 = v5[v33];
    int16_t v37 = (int16_t) v36;
    vint32m1_t v38 = v24;
    vint32m1_t v39 = __riscv_vwmacc_vx_i32m1(v38, v37, v32, v29);
    int8_t v40 = v5[v34];
    int16_t v41 = (int16_t) v40;
    vint32m1_t v42 = v25;
    vint32m1_t v43 = __riscv_vwmacc_vx_i32m1(v42, v41, v32, v29);
    int8_t v44 = v5[v35];
    int16_t v45 = (int16_t) v44;
    vint32m1_t v46 = v26;
    vint32m1_t v47 = __riscv_vwmacc_vx_i32m1(v46, v45, v32, v29);
    v24 = v39;
    v25 = v43;
    v26 = v47;
  }
  vint32m1_t v48 = v24;
  vint32m1_t v49 = v25;
  vint32m1_t v50 = v26;
  int32_t* v51 = &v8[v10];
  __riscv_vse32_v_i32m1(v51, v48, v17);
  int32_t* v52 = &v8[v9];
  __riscv_vse32_v_i32m1(v52, v49, v17);
  size_t v53 = v12 * v9;
  int32_t* v54 = &v8[v53];
  __riscv_vse32_v_i32m1(v54, v50, v17);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_7x3_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 7
//   NR         : 3
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_7x3_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v17);
  size_t v25 = v11 * v9;
  size_t v26 = v10 + v25;
  int32_t* v27 = &v8[v26];
  vint32m1_t v28 = __riscv_vle32_v_i32m1(v27, v17);
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v17);
  vint32m1_t v33;
  v33 = v24;
  vint32m1_t v34;
  v34 = v28;
  vint32m1_t v35;
  v35 = v32;
  for (size_t i36 = v10; i36 < v1; i36 += v11) {
    size_t v37 = i36 * v4;
    size_t v38 = 7;
    int8_t* v39 = &v3[v37];
    vint8mf4_t v40 = __riscv_vle8_v_i8mf4(v39, v38);
    vint16mf2_t v41 = __riscv_vsext_vf2_i16mf2(v40, v38);
    size_t v42 = i36 * v6;
    size_t v43 = v42 + v11;
    size_t v44 = v42 + v12;
    int8_t v45 = v5[v42];
    int16_t v46 = (int16_t) v45;
    vint32m1_t v47 = v33;
    vint32m1_t v48 = __riscv_vwmacc_vx_i32m1(v47, v46, v41, v38);
    int8_t v49 = v5[v43];
    int16_t v50 = (int16_t) v49;
    vint32m1_t v51 = v34;
    vint32m1_t v52 = __riscv_vwmacc_vx_i32m1(v51, v50, v41, v38);
    int8_t v53 = v5[v44];
    int16_t v54 = (int16_t) v53;
    vint32m1_t v55 = v35;
    vint32m1_t v56 = __riscv_vwmacc_vx_i32m1(v55, v54, v41, v38);
    v33 = v48;
    v34 = v52;
    v35 = v56;
  }
  vint32m1_t v57 = v33;
  vint32m1_t v58 = v34;
  vint32m1_t v59 = v35;
  int32_t* v60 = &v8[v10];
  __riscv_vse32_v_i32m1(v60, v57, v17);
  int32_t* v61 = &v8[v9];
  __riscv_vse32_v_i32m1(v61, v58, v17);
  size_t v62 = v12 * v9;
  int32_t* v63 = &v8[v62];
  __riscv_vse32_v_i32m1(v63, v59, v17);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_7x4_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 7
//   NR         : 4
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_7x4_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v22 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v25;
  v25 = v21;
  vint32m1_t v26;
  v26 = v22;
  vint32m1_t v27;
  v27 = v23;
  vint32m1_t v28;
  v28 = v24;
  for (size_t i29 = v10; i29 < v1; i29 += v11) {
    size_t v30 = i29 * v4;
    size_t v31 = 7;
    int8_t* v32 = &v3[v30];
    vint8mf4_t v33 = __riscv_vle8_v_i8mf4(v32, v31);
    vint16mf2_t v34 = __riscv_vsext_vf2_i16mf2(v33, v31);
    size_t v35 = i29 * v6;
    size_t v36 = v35 + v11;
    size_t v37 = v35 + v12;
    size_t v38 = v35 + v13;
    int8_t v39 = v5[v35];
    int16_t v40 = (int16_t) v39;
    vint32m1_t v41 = v25;
    vint32m1_t v42 = __riscv_vwmacc_vx_i32m1(v41, v40, v34, v31);
    int8_t v43 = v5[v36];
    int16_t v44 = (int16_t) v43;
    vint32m1_t v45 = v26;
    vint32m1_t v46 = __riscv_vwmacc_vx_i32m1(v45, v44, v34, v31);
    int8_t v47 = v5[v37];
    int16_t v48 = (int16_t) v47;
    vint32m1_t v49 = v27;
    vint32m1_t v50 = __riscv_vwmacc_vx_i32m1(v49, v48, v34, v31);
    int8_t v51 = v5[v38];
    int16_t v52 = (int16_t) v51;
    vint32m1_t v53 = v28;
    vint32m1_t v54 = __riscv_vwmacc_vx_i32m1(v53, v52, v34, v31);
    v25 = v42;
    v26 = v46;
    v27 = v50;
    v28 = v54;
  }
  vint32m1_t v55 = v25;
  vint32m1_t v56 = v26;
  vint32m1_t v57 = v27;
  vint32m1_t v58 = v28;
  int32_t* v59 = &v8[v10];
  __riscv_vse32_v_i32m1(v59, v55, v17);
  int32_t* v60 = &v8[v9];
  __riscv_vse32_v_i32m1(v60, v56, v17);
  size_t v61 = v12 * v9;
  int32_t* v62 = &v8[v61];
  __riscv_vse32_v_i32m1(v62, v57, v17);
  size_t v63 = v13 * v9;
  int32_t* v64 = &v8[v63];
  __riscv_vse32_v_i32m1(v64, v58, v17);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_7x4_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 7
//   NR         : 4
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_7x4_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v17);
  size_t v25 = v11 * v9;
  size_t v26 = v10 + v25;
  int32_t* v27 = &v8[v26];
  vint32m1_t v28 = __riscv_vle32_v_i32m1(v27, v17);
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v17);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v17);
  vint32m1_t v37;
  v37 = v24;
  vint32m1_t v38;
  v38 = v28;
  vint32m1_t v39;
  v39 = v32;
  vint32m1_t v40;
  v40 = v36;
  for (size_t i41 = v10; i41 < v1; i41 += v11) {
    size_t v42 = i41 * v4;
    size_t v43 = 7;
    int8_t* v44 = &v3[v42];
    vint8mf4_t v45 = __riscv_vle8_v_i8mf4(v44, v43);
    vint16mf2_t v46 = __riscv_vsext_vf2_i16mf2(v45, v43);
    size_t v47 = i41 * v6;
    size_t v48 = v47 + v11;
    size_t v49 = v47 + v12;
    size_t v50 = v47 + v13;
    int8_t v51 = v5[v47];
    int16_t v52 = (int16_t) v51;
    vint32m1_t v53 = v37;
    vint32m1_t v54 = __riscv_vwmacc_vx_i32m1(v53, v52, v46, v43);
    int8_t v55 = v5[v48];
    int16_t v56 = (int16_t) v55;
    vint32m1_t v57 = v38;
    vint32m1_t v58 = __riscv_vwmacc_vx_i32m1(v57, v56, v46, v43);
    int8_t v59 = v5[v49];
    int16_t v60 = (int16_t) v59;
    vint32m1_t v61 = v39;
    vint32m1_t v62 = __riscv_vwmacc_vx_i32m1(v61, v60, v46, v43);
    int8_t v63 = v5[v50];
    int16_t v64 = (int16_t) v63;
    vint32m1_t v65 = v40;
    vint32m1_t v66 = __riscv_vwmacc_vx_i32m1(v65, v64, v46, v43);
    v37 = v54;
    v38 = v58;
    v39 = v62;
    v40 = v66;
  }
  vint32m1_t v67 = v37;
  vint32m1_t v68 = v38;
  vint32m1_t v69 = v39;
  vint32m1_t v70 = v40;
  int32_t* v71 = &v8[v10];
  __riscv_vse32_v_i32m1(v71, v67, v17);
  int32_t* v72 = &v8[v9];
  __riscv_vse32_v_i32m1(v72, v68, v17);
  size_t v73 = v12 * v9;
  int32_t* v74 = &v8[v73];
  __riscv_vse32_v_i32m1(v74, v69, v17);
  size_t v75 = v13 * v9;
  int32_t* v76 = &v8[v75];
  __riscv_vse32_v_i32m1(v76, v70, v17);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_7x5_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 7
//   NR         : 5
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_7x5_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v22 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v25 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v26;
  v26 = v21;
  vint32m1_t v27;
  v27 = v22;
  vint32m1_t v28;
  v28 = v23;
  vint32m1_t v29;
  v29 = v24;
  vint32m1_t v30;
  v30 = v25;
  for (size_t i31 = v10; i31 < v1; i31 += v11) {
    size_t v32 = i31 * v4;
    size_t v33 = 7;
    int8_t* v34 = &v3[v32];
    vint8mf4_t v35 = __riscv_vle8_v_i8mf4(v34, v33);
    vint16mf2_t v36 = __riscv_vsext_vf2_i16mf2(v35, v33);
    size_t v37 = i31 * v6;
    size_t v38 = v37 + v11;
    size_t v39 = v37 + v12;
    size_t v40 = v37 + v13;
    size_t v41 = v37 + v14;
    int8_t v42 = v5[v37];
    int16_t v43 = (int16_t) v42;
    vint32m1_t v44 = v26;
    vint32m1_t v45 = __riscv_vwmacc_vx_i32m1(v44, v43, v36, v33);
    int8_t v46 = v5[v38];
    int16_t v47 = (int16_t) v46;
    vint32m1_t v48 = v27;
    vint32m1_t v49 = __riscv_vwmacc_vx_i32m1(v48, v47, v36, v33);
    int8_t v50 = v5[v39];
    int16_t v51 = (int16_t) v50;
    vint32m1_t v52 = v28;
    vint32m1_t v53 = __riscv_vwmacc_vx_i32m1(v52, v51, v36, v33);
    int8_t v54 = v5[v40];
    int16_t v55 = (int16_t) v54;
    vint32m1_t v56 = v29;
    vint32m1_t v57 = __riscv_vwmacc_vx_i32m1(v56, v55, v36, v33);
    int8_t v58 = v5[v41];
    int16_t v59 = (int16_t) v58;
    vint32m1_t v60 = v30;
    vint32m1_t v61 = __riscv_vwmacc_vx_i32m1(v60, v59, v36, v33);
    v26 = v45;
    v27 = v49;
    v28 = v53;
    v29 = v57;
    v30 = v61;
  }
  vint32m1_t v62 = v26;
  vint32m1_t v63 = v27;
  vint32m1_t v64 = v28;
  vint32m1_t v65 = v29;
  vint32m1_t v66 = v30;
  int32_t* v67 = &v8[v10];
  __riscv_vse32_v_i32m1(v67, v62, v17);
  int32_t* v68 = &v8[v9];
  __riscv_vse32_v_i32m1(v68, v63, v17);
  size_t v69 = v12 * v9;
  int32_t* v70 = &v8[v69];
  __riscv_vse32_v_i32m1(v70, v64, v17);
  size_t v71 = v13 * v9;
  int32_t* v72 = &v8[v71];
  __riscv_vse32_v_i32m1(v72, v65, v17);
  size_t v73 = v14 * v9;
  int32_t* v74 = &v8[v73];
  __riscv_vse32_v_i32m1(v74, v66, v17);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_7x5_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 7
//   NR         : 5
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_7x5_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v17);
  size_t v25 = v11 * v9;
  size_t v26 = v10 + v25;
  int32_t* v27 = &v8[v26];
  vint32m1_t v28 = __riscv_vle32_v_i32m1(v27, v17);
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v17);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v17);
  size_t v37 = v14 * v9;
  size_t v38 = v10 + v37;
  int32_t* v39 = &v8[v38];
  vint32m1_t v40 = __riscv_vle32_v_i32m1(v39, v17);
  vint32m1_t v41;
  v41 = v24;
  vint32m1_t v42;
  v42 = v28;
  vint32m1_t v43;
  v43 = v32;
  vint32m1_t v44;
  v44 = v36;
  vint32m1_t v45;
  v45 = v40;
  for (size_t i46 = v10; i46 < v1; i46 += v11) {
    size_t v47 = i46 * v4;
    size_t v48 = 7;
    int8_t* v49 = &v3[v47];
    vint8mf4_t v50 = __riscv_vle8_v_i8mf4(v49, v48);
    vint16mf2_t v51 = __riscv_vsext_vf2_i16mf2(v50, v48);
    size_t v52 = i46 * v6;
    size_t v53 = v52 + v11;
    size_t v54 = v52 + v12;
    size_t v55 = v52 + v13;
    size_t v56 = v52 + v14;
    int8_t v57 = v5[v52];
    int16_t v58 = (int16_t) v57;
    vint32m1_t v59 = v41;
    vint32m1_t v60 = __riscv_vwmacc_vx_i32m1(v59, v58, v51, v48);
    int8_t v61 = v5[v53];
    int16_t v62 = (int16_t) v61;
    vint32m1_t v63 = v42;
    vint32m1_t v64 = __riscv_vwmacc_vx_i32m1(v63, v62, v51, v48);
    int8_t v65 = v5[v54];
    int16_t v66 = (int16_t) v65;
    vint32m1_t v67 = v43;
    vint32m1_t v68 = __riscv_vwmacc_vx_i32m1(v67, v66, v51, v48);
    int8_t v69 = v5[v55];
    int16_t v70 = (int16_t) v69;
    vint32m1_t v71 = v44;
    vint32m1_t v72 = __riscv_vwmacc_vx_i32m1(v71, v70, v51, v48);
    int8_t v73 = v5[v56];
    int16_t v74 = (int16_t) v73;
    vint32m1_t v75 = v45;
    vint32m1_t v76 = __riscv_vwmacc_vx_i32m1(v75, v74, v51, v48);
    v41 = v60;
    v42 = v64;
    v43 = v68;
    v44 = v72;
    v45 = v76;
  }
  vint32m1_t v77 = v41;
  vint32m1_t v78 = v42;
  vint32m1_t v79 = v43;
  vint32m1_t v80 = v44;
  vint32m1_t v81 = v45;
  int32_t* v82 = &v8[v10];
  __riscv_vse32_v_i32m1(v82, v77, v17);
  int32_t* v83 = &v8[v9];
  __riscv_vse32_v_i32m1(v83, v78, v17);
  size_t v84 = v12 * v9;
  int32_t* v85 = &v8[v84];
  __riscv_vse32_v_i32m1(v85, v79, v17);
  size_t v86 = v13 * v9;
  int32_t* v87 = &v8[v86];
  __riscv_vse32_v_i32m1(v87, v80, v17);
  size_t v88 = v14 * v9;
  int32_t* v89 = &v8[v88];
  __riscv_vse32_v_i32m1(v89, v81, v17);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_7x6_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 7
//   NR         : 6
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_7x6_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v22 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v25 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v26 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v27;
  v27 = v21;
  vint32m1_t v28;
  v28 = v22;
  vint32m1_t v29;
  v29 = v23;
  vint32m1_t v30;
  v30 = v24;
  vint32m1_t v31;
  v31 = v25;
  vint32m1_t v32;
  v32 = v26;
  for (size_t i33 = v10; i33 < v1; i33 += v11) {
    size_t v34 = i33 * v4;
    size_t v35 = 7;
    int8_t* v36 = &v3[v34];
    vint8mf4_t v37 = __riscv_vle8_v_i8mf4(v36, v35);
    vint16mf2_t v38 = __riscv_vsext_vf2_i16mf2(v37, v35);
    size_t v39 = i33 * v6;
    size_t v40 = v39 + v11;
    size_t v41 = v39 + v12;
    size_t v42 = v39 + v13;
    size_t v43 = v39 + v14;
    size_t v44 = v39 + v15;
    int8_t v45 = v5[v39];
    int16_t v46 = (int16_t) v45;
    vint32m1_t v47 = v27;
    vint32m1_t v48 = __riscv_vwmacc_vx_i32m1(v47, v46, v38, v35);
    int8_t v49 = v5[v40];
    int16_t v50 = (int16_t) v49;
    vint32m1_t v51 = v28;
    vint32m1_t v52 = __riscv_vwmacc_vx_i32m1(v51, v50, v38, v35);
    int8_t v53 = v5[v41];
    int16_t v54 = (int16_t) v53;
    vint32m1_t v55 = v29;
    vint32m1_t v56 = __riscv_vwmacc_vx_i32m1(v55, v54, v38, v35);
    int8_t v57 = v5[v42];
    int16_t v58 = (int16_t) v57;
    vint32m1_t v59 = v30;
    vint32m1_t v60 = __riscv_vwmacc_vx_i32m1(v59, v58, v38, v35);
    int8_t v61 = v5[v43];
    int16_t v62 = (int16_t) v61;
    vint32m1_t v63 = v31;
    vint32m1_t v64 = __riscv_vwmacc_vx_i32m1(v63, v62, v38, v35);
    int8_t v65 = v5[v44];
    int16_t v66 = (int16_t) v65;
    vint32m1_t v67 = v32;
    vint32m1_t v68 = __riscv_vwmacc_vx_i32m1(v67, v66, v38, v35);
    v27 = v48;
    v28 = v52;
    v29 = v56;
    v30 = v60;
    v31 = v64;
    v32 = v68;
  }
  vint32m1_t v69 = v27;
  vint32m1_t v70 = v28;
  vint32m1_t v71 = v29;
  vint32m1_t v72 = v30;
  vint32m1_t v73 = v31;
  vint32m1_t v74 = v32;
  int32_t* v75 = &v8[v10];
  __riscv_vse32_v_i32m1(v75, v69, v17);
  int32_t* v76 = &v8[v9];
  __riscv_vse32_v_i32m1(v76, v70, v17);
  size_t v77 = v12 * v9;
  int32_t* v78 = &v8[v77];
  __riscv_vse32_v_i32m1(v78, v71, v17);
  size_t v79 = v13 * v9;
  int32_t* v80 = &v8[v79];
  __riscv_vse32_v_i32m1(v80, v72, v17);
  size_t v81 = v14 * v9;
  int32_t* v82 = &v8[v81];
  __riscv_vse32_v_i32m1(v82, v73, v17);
  size_t v83 = v15 * v9;
  int32_t* v84 = &v8[v83];
  __riscv_vse32_v_i32m1(v84, v74, v17);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_7x6_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 7
//   NR         : 6
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_7x6_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v17);
  size_t v25 = v11 * v9;
  size_t v26 = v10 + v25;
  int32_t* v27 = &v8[v26];
  vint32m1_t v28 = __riscv_vle32_v_i32m1(v27, v17);
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v17);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v17);
  size_t v37 = v14 * v9;
  size_t v38 = v10 + v37;
  int32_t* v39 = &v8[v38];
  vint32m1_t v40 = __riscv_vle32_v_i32m1(v39, v17);
  size_t v41 = v15 * v9;
  size_t v42 = v10 + v41;
  int32_t* v43 = &v8[v42];
  vint32m1_t v44 = __riscv_vle32_v_i32m1(v43, v17);
  vint32m1_t v45;
  v45 = v24;
  vint32m1_t v46;
  v46 = v28;
  vint32m1_t v47;
  v47 = v32;
  vint32m1_t v48;
  v48 = v36;
  vint32m1_t v49;
  v49 = v40;
  vint32m1_t v50;
  v50 = v44;
  for (size_t i51 = v10; i51 < v1; i51 += v11) {
    size_t v52 = i51 * v4;
    size_t v53 = 7;
    int8_t* v54 = &v3[v52];
    vint8mf4_t v55 = __riscv_vle8_v_i8mf4(v54, v53);
    vint16mf2_t v56 = __riscv_vsext_vf2_i16mf2(v55, v53);
    size_t v57 = i51 * v6;
    size_t v58 = v57 + v11;
    size_t v59 = v57 + v12;
    size_t v60 = v57 + v13;
    size_t v61 = v57 + v14;
    size_t v62 = v57 + v15;
    int8_t v63 = v5[v57];
    int16_t v64 = (int16_t) v63;
    vint32m1_t v65 = v45;
    vint32m1_t v66 = __riscv_vwmacc_vx_i32m1(v65, v64, v56, v53);
    int8_t v67 = v5[v58];
    int16_t v68 = (int16_t) v67;
    vint32m1_t v69 = v46;
    vint32m1_t v70 = __riscv_vwmacc_vx_i32m1(v69, v68, v56, v53);
    int8_t v71 = v5[v59];
    int16_t v72 = (int16_t) v71;
    vint32m1_t v73 = v47;
    vint32m1_t v74 = __riscv_vwmacc_vx_i32m1(v73, v72, v56, v53);
    int8_t v75 = v5[v60];
    int16_t v76 = (int16_t) v75;
    vint32m1_t v77 = v48;
    vint32m1_t v78 = __riscv_vwmacc_vx_i32m1(v77, v76, v56, v53);
    int8_t v79 = v5[v61];
    int16_t v80 = (int16_t) v79;
    vint32m1_t v81 = v49;
    vint32m1_t v82 = __riscv_vwmacc_vx_i32m1(v81, v80, v56, v53);
    int8_t v83 = v5[v62];
    int16_t v84 = (int16_t) v83;
    vint32m1_t v85 = v50;
    vint32m1_t v86 = __riscv_vwmacc_vx_i32m1(v85, v84, v56, v53);
    v45 = v66;
    v46 = v70;
    v47 = v74;
    v48 = v78;
    v49 = v82;
    v50 = v86;
  }
  vint32m1_t v87 = v45;
  vint32m1_t v88 = v46;
  vint32m1_t v89 = v47;
  vint32m1_t v90 = v48;
  vint32m1_t v91 = v49;
  vint32m1_t v92 = v50;
  int32_t* v93 = &v8[v10];
  __riscv_vse32_v_i32m1(v93, v87, v17);
  int32_t* v94 = &v8[v9];
  __riscv_vse32_v_i32m1(v94, v88, v17);
  size_t v95 = v12 * v9;
  int32_t* v96 = &v8[v95];
  __riscv_vse32_v_i32m1(v96, v89, v17);
  size_t v97 = v13 * v9;
  int32_t* v98 = &v8[v97];
  __riscv_vse32_v_i32m1(v98, v90, v17);
  size_t v99 = v14 * v9;
  int32_t* v100 = &v8[v99];
  __riscv_vse32_v_i32m1(v100, v91, v17);
  size_t v101 = v15 * v9;
  int32_t* v102 = &v8[v101];
  __riscv_vse32_v_i32m1(v102, v92, v17);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_7x7_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 7
//   NR         : 7
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_7x7_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v22 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v25 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v26 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v27 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v28;
  v28 = v21;
  vint32m1_t v29;
  v29 = v22;
  vint32m1_t v30;
  v30 = v23;
  vint32m1_t v31;
  v31 = v24;
  vint32m1_t v32;
  v32 = v25;
  vint32m1_t v33;
  v33 = v26;
  vint32m1_t v34;
  v34 = v27;
  for (size_t i35 = v10; i35 < v1; i35 += v11) {
    size_t v36 = i35 * v4;
    size_t v37 = 7;
    int8_t* v38 = &v3[v36];
    vint8mf4_t v39 = __riscv_vle8_v_i8mf4(v38, v37);
    vint16mf2_t v40 = __riscv_vsext_vf2_i16mf2(v39, v37);
    size_t v41 = i35 * v6;
    size_t v42 = v41 + v11;
    size_t v43 = v41 + v12;
    size_t v44 = v41 + v13;
    size_t v45 = v41 + v14;
    size_t v46 = v41 + v15;
    size_t v47 = v41 + v16;
    int8_t v48 = v5[v41];
    int16_t v49 = (int16_t) v48;
    vint32m1_t v50 = v28;
    vint32m1_t v51 = __riscv_vwmacc_vx_i32m1(v50, v49, v40, v37);
    int8_t v52 = v5[v42];
    int16_t v53 = (int16_t) v52;
    vint32m1_t v54 = v29;
    vint32m1_t v55 = __riscv_vwmacc_vx_i32m1(v54, v53, v40, v37);
    int8_t v56 = v5[v43];
    int16_t v57 = (int16_t) v56;
    vint32m1_t v58 = v30;
    vint32m1_t v59 = __riscv_vwmacc_vx_i32m1(v58, v57, v40, v37);
    int8_t v60 = v5[v44];
    int16_t v61 = (int16_t) v60;
    vint32m1_t v62 = v31;
    vint32m1_t v63 = __riscv_vwmacc_vx_i32m1(v62, v61, v40, v37);
    int8_t v64 = v5[v45];
    int16_t v65 = (int16_t) v64;
    vint32m1_t v66 = v32;
    vint32m1_t v67 = __riscv_vwmacc_vx_i32m1(v66, v65, v40, v37);
    int8_t v68 = v5[v46];
    int16_t v69 = (int16_t) v68;
    vint32m1_t v70 = v33;
    vint32m1_t v71 = __riscv_vwmacc_vx_i32m1(v70, v69, v40, v37);
    int8_t v72 = v5[v47];
    int16_t v73 = (int16_t) v72;
    vint32m1_t v74 = v34;
    vint32m1_t v75 = __riscv_vwmacc_vx_i32m1(v74, v73, v40, v37);
    v28 = v51;
    v29 = v55;
    v30 = v59;
    v31 = v63;
    v32 = v67;
    v33 = v71;
    v34 = v75;
  }
  vint32m1_t v76 = v28;
  vint32m1_t v77 = v29;
  vint32m1_t v78 = v30;
  vint32m1_t v79 = v31;
  vint32m1_t v80 = v32;
  vint32m1_t v81 = v33;
  vint32m1_t v82 = v34;
  int32_t* v83 = &v8[v10];
  __riscv_vse32_v_i32m1(v83, v76, v17);
  int32_t* v84 = &v8[v9];
  __riscv_vse32_v_i32m1(v84, v77, v17);
  size_t v85 = v12 * v9;
  int32_t* v86 = &v8[v85];
  __riscv_vse32_v_i32m1(v86, v78, v17);
  size_t v87 = v13 * v9;
  int32_t* v88 = &v8[v87];
  __riscv_vse32_v_i32m1(v88, v79, v17);
  size_t v89 = v14 * v9;
  int32_t* v90 = &v8[v89];
  __riscv_vse32_v_i32m1(v90, v80, v17);
  size_t v91 = v15 * v9;
  int32_t* v92 = &v8[v91];
  __riscv_vse32_v_i32m1(v92, v81, v17);
  size_t v93 = v16 * v9;
  int32_t* v94 = &v8[v93];
  __riscv_vse32_v_i32m1(v94, v82, v17);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_7x7_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 7
//   NR         : 7
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_7x7_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v17);
  size_t v25 = v11 * v9;
  size_t v26 = v10 + v25;
  int32_t* v27 = &v8[v26];
  vint32m1_t v28 = __riscv_vle32_v_i32m1(v27, v17);
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v17);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v17);
  size_t v37 = v14 * v9;
  size_t v38 = v10 + v37;
  int32_t* v39 = &v8[v38];
  vint32m1_t v40 = __riscv_vle32_v_i32m1(v39, v17);
  size_t v41 = v15 * v9;
  size_t v42 = v10 + v41;
  int32_t* v43 = &v8[v42];
  vint32m1_t v44 = __riscv_vle32_v_i32m1(v43, v17);
  size_t v45 = v16 * v9;
  size_t v46 = v10 + v45;
  int32_t* v47 = &v8[v46];
  vint32m1_t v48 = __riscv_vle32_v_i32m1(v47, v17);
  vint32m1_t v49;
  v49 = v24;
  vint32m1_t v50;
  v50 = v28;
  vint32m1_t v51;
  v51 = v32;
  vint32m1_t v52;
  v52 = v36;
  vint32m1_t v53;
  v53 = v40;
  vint32m1_t v54;
  v54 = v44;
  vint32m1_t v55;
  v55 = v48;
  for (size_t i56 = v10; i56 < v1; i56 += v11) {
    size_t v57 = i56 * v4;
    size_t v58 = 7;
    int8_t* v59 = &v3[v57];
    vint8mf4_t v60 = __riscv_vle8_v_i8mf4(v59, v58);
    vint16mf2_t v61 = __riscv_vsext_vf2_i16mf2(v60, v58);
    size_t v62 = i56 * v6;
    size_t v63 = v62 + v11;
    size_t v64 = v62 + v12;
    size_t v65 = v62 + v13;
    size_t v66 = v62 + v14;
    size_t v67 = v62 + v15;
    size_t v68 = v62 + v16;
    int8_t v69 = v5[v62];
    int16_t v70 = (int16_t) v69;
    vint32m1_t v71 = v49;
    vint32m1_t v72 = __riscv_vwmacc_vx_i32m1(v71, v70, v61, v58);
    int8_t v73 = v5[v63];
    int16_t v74 = (int16_t) v73;
    vint32m1_t v75 = v50;
    vint32m1_t v76 = __riscv_vwmacc_vx_i32m1(v75, v74, v61, v58);
    int8_t v77 = v5[v64];
    int16_t v78 = (int16_t) v77;
    vint32m1_t v79 = v51;
    vint32m1_t v80 = __riscv_vwmacc_vx_i32m1(v79, v78, v61, v58);
    int8_t v81 = v5[v65];
    int16_t v82 = (int16_t) v81;
    vint32m1_t v83 = v52;
    vint32m1_t v84 = __riscv_vwmacc_vx_i32m1(v83, v82, v61, v58);
    int8_t v85 = v5[v66];
    int16_t v86 = (int16_t) v85;
    vint32m1_t v87 = v53;
    vint32m1_t v88 = __riscv_vwmacc_vx_i32m1(v87, v86, v61, v58);
    int8_t v89 = v5[v67];
    int16_t v90 = (int16_t) v89;
    vint32m1_t v91 = v54;
    vint32m1_t v92 = __riscv_vwmacc_vx_i32m1(v91, v90, v61, v58);
    int8_t v93 = v5[v68];
    int16_t v94 = (int16_t) v93;
    vint32m1_t v95 = v55;
    vint32m1_t v96 = __riscv_vwmacc_vx_i32m1(v95, v94, v61, v58);
    v49 = v72;
    v50 = v76;
    v51 = v80;
    v52 = v84;
    v53 = v88;
    v54 = v92;
    v55 = v96;
  }
  vint32m1_t v97 = v49;
  vint32m1_t v98 = v50;
  vint32m1_t v99 = v51;
  vint32m1_t v100 = v52;
  vint32m1_t v101 = v53;
  vint32m1_t v102 = v54;
  vint32m1_t v103 = v55;
  int32_t* v104 = &v8[v10];
  __riscv_vse32_v_i32m1(v104, v97, v17);
  int32_t* v105 = &v8[v9];
  __riscv_vse32_v_i32m1(v105, v98, v17);
  size_t v106 = v12 * v9;
  int32_t* v107 = &v8[v106];
  __riscv_vse32_v_i32m1(v107, v99, v17);
  size_t v108 = v13 * v9;
  int32_t* v109 = &v8[v108];
  __riscv_vse32_v_i32m1(v109, v100, v17);
  size_t v110 = v14 * v9;
  int32_t* v111 = &v8[v110];
  __riscv_vse32_v_i32m1(v111, v101, v17);
  size_t v112 = v15 * v9;
  int32_t* v113 = &v8[v112];
  __riscv_vse32_v_i32m1(v113, v102, v17);
  size_t v114 = v16 * v9;
  int32_t* v115 = &v8[v114];
  __riscv_vse32_v_i32m1(v115, v103, v17);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_7x8_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 7
//   NR         : 8
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_7x8_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v22 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v25 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v26 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v27 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v28 = __riscv_vmv_v_x_i32m1(v20, v17);
  vint32m1_t v29;
  v29 = v21;
  vint32m1_t v30;
  v30 = v22;
  vint32m1_t v31;
  v31 = v23;
  vint32m1_t v32;
  v32 = v24;
  vint32m1_t v33;
  v33 = v25;
  vint32m1_t v34;
  v34 = v26;
  vint32m1_t v35;
  v35 = v27;
  vint32m1_t v36;
  v36 = v28;
  for (size_t i37 = v10; i37 < v1; i37 += v11) {
    size_t v38 = i37 * v4;
    size_t v39 = 7;
    int8_t* v40 = &v3[v38];
    vint8mf4_t v41 = __riscv_vle8_v_i8mf4(v40, v39);
    vint16mf2_t v42 = __riscv_vsext_vf2_i16mf2(v41, v39);
    size_t v43 = i37 * v6;
    size_t v44 = v43 + v11;
    size_t v45 = v43 + v12;
    size_t v46 = v43 + v13;
    size_t v47 = v43 + v14;
    size_t v48 = v43 + v15;
    size_t v49 = v43 + v16;
    size_t v50 = v43 + v17;
    int8_t v51 = v5[v43];
    int16_t v52 = (int16_t) v51;
    vint32m1_t v53 = v29;
    vint32m1_t v54 = __riscv_vwmacc_vx_i32m1(v53, v52, v42, v39);
    int8_t v55 = v5[v44];
    int16_t v56 = (int16_t) v55;
    vint32m1_t v57 = v30;
    vint32m1_t v58 = __riscv_vwmacc_vx_i32m1(v57, v56, v42, v39);
    int8_t v59 = v5[v45];
    int16_t v60 = (int16_t) v59;
    vint32m1_t v61 = v31;
    vint32m1_t v62 = __riscv_vwmacc_vx_i32m1(v61, v60, v42, v39);
    int8_t v63 = v5[v46];
    int16_t v64 = (int16_t) v63;
    vint32m1_t v65 = v32;
    vint32m1_t v66 = __riscv_vwmacc_vx_i32m1(v65, v64, v42, v39);
    int8_t v67 = v5[v47];
    int16_t v68 = (int16_t) v67;
    vint32m1_t v69 = v33;
    vint32m1_t v70 = __riscv_vwmacc_vx_i32m1(v69, v68, v42, v39);
    int8_t v71 = v5[v48];
    int16_t v72 = (int16_t) v71;
    vint32m1_t v73 = v34;
    vint32m1_t v74 = __riscv_vwmacc_vx_i32m1(v73, v72, v42, v39);
    int8_t v75 = v5[v49];
    int16_t v76 = (int16_t) v75;
    vint32m1_t v77 = v35;
    vint32m1_t v78 = __riscv_vwmacc_vx_i32m1(v77, v76, v42, v39);
    int8_t v79 = v5[v50];
    int16_t v80 = (int16_t) v79;
    vint32m1_t v81 = v36;
    vint32m1_t v82 = __riscv_vwmacc_vx_i32m1(v81, v80, v42, v39);
    v29 = v54;
    v30 = v58;
    v31 = v62;
    v32 = v66;
    v33 = v70;
    v34 = v74;
    v35 = v78;
    v36 = v82;
  }
  vint32m1_t v83 = v29;
  vint32m1_t v84 = v30;
  vint32m1_t v85 = v31;
  vint32m1_t v86 = v32;
  vint32m1_t v87 = v33;
  vint32m1_t v88 = v34;
  vint32m1_t v89 = v35;
  vint32m1_t v90 = v36;
  int32_t* v91 = &v8[v10];
  __riscv_vse32_v_i32m1(v91, v83, v17);
  int32_t* v92 = &v8[v9];
  __riscv_vse32_v_i32m1(v92, v84, v17);
  size_t v93 = v12 * v9;
  int32_t* v94 = &v8[v93];
  __riscv_vse32_v_i32m1(v94, v85, v17);
  size_t v95 = v13 * v9;
  int32_t* v96 = &v8[v95];
  __riscv_vse32_v_i32m1(v96, v86, v17);
  size_t v97 = v14 * v9;
  int32_t* v98 = &v8[v97];
  __riscv_vse32_v_i32m1(v98, v87, v17);
  size_t v99 = v15 * v9;
  int32_t* v100 = &v8[v99];
  __riscv_vse32_v_i32m1(v100, v88, v17);
  size_t v101 = v16 * v9;
  int32_t* v102 = &v8[v101];
  __riscv_vse32_v_i32m1(v102, v89, v17);
  size_t v103 = v17 * v9;
  int32_t* v104 = &v8[v103];
  __riscv_vse32_v_i32m1(v104, v90, v17);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_7x8_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 7
//   NR         : 8
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_7x8_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v17);
  size_t v25 = v11 * v9;
  size_t v26 = v10 + v25;
  int32_t* v27 = &v8[v26];
  vint32m1_t v28 = __riscv_vle32_v_i32m1(v27, v17);
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v17);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v17);
  size_t v37 = v14 * v9;
  size_t v38 = v10 + v37;
  int32_t* v39 = &v8[v38];
  vint32m1_t v40 = __riscv_vle32_v_i32m1(v39, v17);
  size_t v41 = v15 * v9;
  size_t v42 = v10 + v41;
  int32_t* v43 = &v8[v42];
  vint32m1_t v44 = __riscv_vle32_v_i32m1(v43, v17);
  size_t v45 = v16 * v9;
  size_t v46 = v10 + v45;
  int32_t* v47 = &v8[v46];
  vint32m1_t v48 = __riscv_vle32_v_i32m1(v47, v17);
  size_t v49 = v17 * v9;
  size_t v50 = v10 + v49;
  int32_t* v51 = &v8[v50];
  vint32m1_t v52 = __riscv_vle32_v_i32m1(v51, v17);
  vint32m1_t v53;
  v53 = v24;
  vint32m1_t v54;
  v54 = v28;
  vint32m1_t v55;
  v55 = v32;
  vint32m1_t v56;
  v56 = v36;
  vint32m1_t v57;
  v57 = v40;
  vint32m1_t v58;
  v58 = v44;
  vint32m1_t v59;
  v59 = v48;
  vint32m1_t v60;
  v60 = v52;
  for (size_t i61 = v10; i61 < v1; i61 += v11) {
    size_t v62 = i61 * v4;
    size_t v63 = 7;
    int8_t* v64 = &v3[v62];
    vint8mf4_t v65 = __riscv_vle8_v_i8mf4(v64, v63);
    vint16mf2_t v66 = __riscv_vsext_vf2_i16mf2(v65, v63);
    size_t v67 = i61 * v6;
    size_t v68 = v67 + v11;
    size_t v69 = v67 + v12;
    size_t v70 = v67 + v13;
    size_t v71 = v67 + v14;
    size_t v72 = v67 + v15;
    size_t v73 = v67 + v16;
    size_t v74 = v67 + v17;
    int8_t v75 = v5[v67];
    int16_t v76 = (int16_t) v75;
    vint32m1_t v77 = v53;
    vint32m1_t v78 = __riscv_vwmacc_vx_i32m1(v77, v76, v66, v63);
    int8_t v79 = v5[v68];
    int16_t v80 = (int16_t) v79;
    vint32m1_t v81 = v54;
    vint32m1_t v82 = __riscv_vwmacc_vx_i32m1(v81, v80, v66, v63);
    int8_t v83 = v5[v69];
    int16_t v84 = (int16_t) v83;
    vint32m1_t v85 = v55;
    vint32m1_t v86 = __riscv_vwmacc_vx_i32m1(v85, v84, v66, v63);
    int8_t v87 = v5[v70];
    int16_t v88 = (int16_t) v87;
    vint32m1_t v89 = v56;
    vint32m1_t v90 = __riscv_vwmacc_vx_i32m1(v89, v88, v66, v63);
    int8_t v91 = v5[v71];
    int16_t v92 = (int16_t) v91;
    vint32m1_t v93 = v57;
    vint32m1_t v94 = __riscv_vwmacc_vx_i32m1(v93, v92, v66, v63);
    int8_t v95 = v5[v72];
    int16_t v96 = (int16_t) v95;
    vint32m1_t v97 = v58;
    vint32m1_t v98 = __riscv_vwmacc_vx_i32m1(v97, v96, v66, v63);
    int8_t v99 = v5[v73];
    int16_t v100 = (int16_t) v99;
    vint32m1_t v101 = v59;
    vint32m1_t v102 = __riscv_vwmacc_vx_i32m1(v101, v100, v66, v63);
    int8_t v103 = v5[v74];
    int16_t v104 = (int16_t) v103;
    vint32m1_t v105 = v60;
    vint32m1_t v106 = __riscv_vwmacc_vx_i32m1(v105, v104, v66, v63);
    v53 = v78;
    v54 = v82;
    v55 = v86;
    v56 = v90;
    v57 = v94;
    v58 = v98;
    v59 = v102;
    v60 = v106;
  }
  vint32m1_t v107 = v53;
  vint32m1_t v108 = v54;
  vint32m1_t v109 = v55;
  vint32m1_t v110 = v56;
  vint32m1_t v111 = v57;
  vint32m1_t v112 = v58;
  vint32m1_t v113 = v59;
  vint32m1_t v114 = v60;
  int32_t* v115 = &v8[v10];
  __riscv_vse32_v_i32m1(v115, v107, v17);
  int32_t* v116 = &v8[v9];
  __riscv_vse32_v_i32m1(v116, v108, v17);
  size_t v117 = v12 * v9;
  int32_t* v118 = &v8[v117];
  __riscv_vse32_v_i32m1(v118, v109, v17);
  size_t v119 = v13 * v9;
  int32_t* v120 = &v8[v119];
  __riscv_vse32_v_i32m1(v120, v110, v17);
  size_t v121 = v14 * v9;
  int32_t* v122 = &v8[v121];
  __riscv_vse32_v_i32m1(v122, v111, v17);
  size_t v123 = v15 * v9;
  int32_t* v124 = &v8[v123];
  __riscv_vse32_v_i32m1(v124, v112, v17);
  size_t v125 = v16 * v9;
  int32_t* v126 = &v8[v125];
  __riscv_vse32_v_i32m1(v126, v113, v17);
  size_t v127 = v17 * v9;
  int32_t* v128 = &v8[v127];
  __riscv_vse32_v_i32m1(v128, v114, v17);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_8x1_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 8
//   NR         : 1
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_8x1_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v22;
  v22 = v21;
  for (size_t i23 = v10; i23 < v1; i23 += v11) {
    size_t v24 = i23 * v4;
    size_t v25 = 8;
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
  __riscv_vse32_v_i32m1(v35, v34, v18);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_8x1_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 8
//   NR         : 1
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_8x1_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v18);
  vint32m1_t v25;
  v25 = v24;
  for (size_t i26 = v10; i26 < v1; i26 += v11) {
    size_t v27 = i26 * v4;
    size_t v28 = 8;
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
  __riscv_vse32_v_i32m1(v38, v37, v18);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_8x2_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 8
//   NR         : 2
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_8x2_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v22 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v23;
  v23 = v21;
  vint32m1_t v24;
  v24 = v22;
  for (size_t i25 = v10; i25 < v1; i25 += v11) {
    size_t v26 = i25 * v4;
    size_t v27 = 8;
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
  __riscv_vse32_v_i32m1(v43, v41, v18);
  int32_t* v44 = &v8[v9];
  __riscv_vse32_v_i32m1(v44, v42, v18);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_8x2_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 8
//   NR         : 2
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_8x2_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v18);
  size_t v25 = v11 * v9;
  size_t v26 = v10 + v25;
  int32_t* v27 = &v8[v26];
  vint32m1_t v28 = __riscv_vle32_v_i32m1(v27, v18);
  vint32m1_t v29;
  v29 = v24;
  vint32m1_t v30;
  v30 = v28;
  for (size_t i31 = v10; i31 < v1; i31 += v11) {
    size_t v32 = i31 * v4;
    size_t v33 = 8;
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
  __riscv_vse32_v_i32m1(v49, v47, v18);
  int32_t* v50 = &v8[v9];
  __riscv_vse32_v_i32m1(v50, v48, v18);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_8x3_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 8
//   NR         : 3
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_8x3_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v22 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v24;
  v24 = v21;
  vint32m1_t v25;
  v25 = v22;
  vint32m1_t v26;
  v26 = v23;
  for (size_t i27 = v10; i27 < v1; i27 += v11) {
    size_t v28 = i27 * v4;
    size_t v29 = 8;
    int8_t* v30 = &v3[v28];
    vint8mf4_t v31 = __riscv_vle8_v_i8mf4(v30, v29);
    vint16mf2_t v32 = __riscv_vsext_vf2_i16mf2(v31, v29);
    size_t v33 = i27 * v6;
    size_t v34 = v33 + v11;
    size_t v35 = v33 + v12;
    int8_t v36 = v5[v33];
    int16_t v37 = (int16_t) v36;
    vint32m1_t v38 = v24;
    vint32m1_t v39 = __riscv_vwmacc_vx_i32m1(v38, v37, v32, v29);
    int8_t v40 = v5[v34];
    int16_t v41 = (int16_t) v40;
    vint32m1_t v42 = v25;
    vint32m1_t v43 = __riscv_vwmacc_vx_i32m1(v42, v41, v32, v29);
    int8_t v44 = v5[v35];
    int16_t v45 = (int16_t) v44;
    vint32m1_t v46 = v26;
    vint32m1_t v47 = __riscv_vwmacc_vx_i32m1(v46, v45, v32, v29);
    v24 = v39;
    v25 = v43;
    v26 = v47;
  }
  vint32m1_t v48 = v24;
  vint32m1_t v49 = v25;
  vint32m1_t v50 = v26;
  int32_t* v51 = &v8[v10];
  __riscv_vse32_v_i32m1(v51, v48, v18);
  int32_t* v52 = &v8[v9];
  __riscv_vse32_v_i32m1(v52, v49, v18);
  size_t v53 = v12 * v9;
  int32_t* v54 = &v8[v53];
  __riscv_vse32_v_i32m1(v54, v50, v18);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_8x3_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 8
//   NR         : 3
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_8x3_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v18);
  size_t v25 = v11 * v9;
  size_t v26 = v10 + v25;
  int32_t* v27 = &v8[v26];
  vint32m1_t v28 = __riscv_vle32_v_i32m1(v27, v18);
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v18);
  vint32m1_t v33;
  v33 = v24;
  vint32m1_t v34;
  v34 = v28;
  vint32m1_t v35;
  v35 = v32;
  for (size_t i36 = v10; i36 < v1; i36 += v11) {
    size_t v37 = i36 * v4;
    size_t v38 = 8;
    int8_t* v39 = &v3[v37];
    vint8mf4_t v40 = __riscv_vle8_v_i8mf4(v39, v38);
    vint16mf2_t v41 = __riscv_vsext_vf2_i16mf2(v40, v38);
    size_t v42 = i36 * v6;
    size_t v43 = v42 + v11;
    size_t v44 = v42 + v12;
    int8_t v45 = v5[v42];
    int16_t v46 = (int16_t) v45;
    vint32m1_t v47 = v33;
    vint32m1_t v48 = __riscv_vwmacc_vx_i32m1(v47, v46, v41, v38);
    int8_t v49 = v5[v43];
    int16_t v50 = (int16_t) v49;
    vint32m1_t v51 = v34;
    vint32m1_t v52 = __riscv_vwmacc_vx_i32m1(v51, v50, v41, v38);
    int8_t v53 = v5[v44];
    int16_t v54 = (int16_t) v53;
    vint32m1_t v55 = v35;
    vint32m1_t v56 = __riscv_vwmacc_vx_i32m1(v55, v54, v41, v38);
    v33 = v48;
    v34 = v52;
    v35 = v56;
  }
  vint32m1_t v57 = v33;
  vint32m1_t v58 = v34;
  vint32m1_t v59 = v35;
  int32_t* v60 = &v8[v10];
  __riscv_vse32_v_i32m1(v60, v57, v18);
  int32_t* v61 = &v8[v9];
  __riscv_vse32_v_i32m1(v61, v58, v18);
  size_t v62 = v12 * v9;
  int32_t* v63 = &v8[v62];
  __riscv_vse32_v_i32m1(v63, v59, v18);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_8x4_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 8
//   NR         : 4
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_8x4_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v22 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v25;
  v25 = v21;
  vint32m1_t v26;
  v26 = v22;
  vint32m1_t v27;
  v27 = v23;
  vint32m1_t v28;
  v28 = v24;
  for (size_t i29 = v10; i29 < v1; i29 += v11) {
    size_t v30 = i29 * v4;
    size_t v31 = 8;
    int8_t* v32 = &v3[v30];
    vint8mf4_t v33 = __riscv_vle8_v_i8mf4(v32, v31);
    vint16mf2_t v34 = __riscv_vsext_vf2_i16mf2(v33, v31);
    size_t v35 = i29 * v6;
    size_t v36 = v35 + v11;
    size_t v37 = v35 + v12;
    size_t v38 = v35 + v13;
    int8_t v39 = v5[v35];
    int16_t v40 = (int16_t) v39;
    vint32m1_t v41 = v25;
    vint32m1_t v42 = __riscv_vwmacc_vx_i32m1(v41, v40, v34, v31);
    int8_t v43 = v5[v36];
    int16_t v44 = (int16_t) v43;
    vint32m1_t v45 = v26;
    vint32m1_t v46 = __riscv_vwmacc_vx_i32m1(v45, v44, v34, v31);
    int8_t v47 = v5[v37];
    int16_t v48 = (int16_t) v47;
    vint32m1_t v49 = v27;
    vint32m1_t v50 = __riscv_vwmacc_vx_i32m1(v49, v48, v34, v31);
    int8_t v51 = v5[v38];
    int16_t v52 = (int16_t) v51;
    vint32m1_t v53 = v28;
    vint32m1_t v54 = __riscv_vwmacc_vx_i32m1(v53, v52, v34, v31);
    v25 = v42;
    v26 = v46;
    v27 = v50;
    v28 = v54;
  }
  vint32m1_t v55 = v25;
  vint32m1_t v56 = v26;
  vint32m1_t v57 = v27;
  vint32m1_t v58 = v28;
  int32_t* v59 = &v8[v10];
  __riscv_vse32_v_i32m1(v59, v55, v18);
  int32_t* v60 = &v8[v9];
  __riscv_vse32_v_i32m1(v60, v56, v18);
  size_t v61 = v12 * v9;
  int32_t* v62 = &v8[v61];
  __riscv_vse32_v_i32m1(v62, v57, v18);
  size_t v63 = v13 * v9;
  int32_t* v64 = &v8[v63];
  __riscv_vse32_v_i32m1(v64, v58, v18);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_8x4_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 8
//   NR         : 4
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_8x4_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v18);
  size_t v25 = v11 * v9;
  size_t v26 = v10 + v25;
  int32_t* v27 = &v8[v26];
  vint32m1_t v28 = __riscv_vle32_v_i32m1(v27, v18);
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v18);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v18);
  vint32m1_t v37;
  v37 = v24;
  vint32m1_t v38;
  v38 = v28;
  vint32m1_t v39;
  v39 = v32;
  vint32m1_t v40;
  v40 = v36;
  for (size_t i41 = v10; i41 < v1; i41 += v11) {
    size_t v42 = i41 * v4;
    size_t v43 = 8;
    int8_t* v44 = &v3[v42];
    vint8mf4_t v45 = __riscv_vle8_v_i8mf4(v44, v43);
    vint16mf2_t v46 = __riscv_vsext_vf2_i16mf2(v45, v43);
    size_t v47 = i41 * v6;
    size_t v48 = v47 + v11;
    size_t v49 = v47 + v12;
    size_t v50 = v47 + v13;
    int8_t v51 = v5[v47];
    int16_t v52 = (int16_t) v51;
    vint32m1_t v53 = v37;
    vint32m1_t v54 = __riscv_vwmacc_vx_i32m1(v53, v52, v46, v43);
    int8_t v55 = v5[v48];
    int16_t v56 = (int16_t) v55;
    vint32m1_t v57 = v38;
    vint32m1_t v58 = __riscv_vwmacc_vx_i32m1(v57, v56, v46, v43);
    int8_t v59 = v5[v49];
    int16_t v60 = (int16_t) v59;
    vint32m1_t v61 = v39;
    vint32m1_t v62 = __riscv_vwmacc_vx_i32m1(v61, v60, v46, v43);
    int8_t v63 = v5[v50];
    int16_t v64 = (int16_t) v63;
    vint32m1_t v65 = v40;
    vint32m1_t v66 = __riscv_vwmacc_vx_i32m1(v65, v64, v46, v43);
    v37 = v54;
    v38 = v58;
    v39 = v62;
    v40 = v66;
  }
  vint32m1_t v67 = v37;
  vint32m1_t v68 = v38;
  vint32m1_t v69 = v39;
  vint32m1_t v70 = v40;
  int32_t* v71 = &v8[v10];
  __riscv_vse32_v_i32m1(v71, v67, v18);
  int32_t* v72 = &v8[v9];
  __riscv_vse32_v_i32m1(v72, v68, v18);
  size_t v73 = v12 * v9;
  int32_t* v74 = &v8[v73];
  __riscv_vse32_v_i32m1(v74, v69, v18);
  size_t v75 = v13 * v9;
  int32_t* v76 = &v8[v75];
  __riscv_vse32_v_i32m1(v76, v70, v18);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_8x5_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 8
//   NR         : 5
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_8x5_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v22 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v25 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v26;
  v26 = v21;
  vint32m1_t v27;
  v27 = v22;
  vint32m1_t v28;
  v28 = v23;
  vint32m1_t v29;
  v29 = v24;
  vint32m1_t v30;
  v30 = v25;
  for (size_t i31 = v10; i31 < v1; i31 += v11) {
    size_t v32 = i31 * v4;
    size_t v33 = 8;
    int8_t* v34 = &v3[v32];
    vint8mf4_t v35 = __riscv_vle8_v_i8mf4(v34, v33);
    vint16mf2_t v36 = __riscv_vsext_vf2_i16mf2(v35, v33);
    size_t v37 = i31 * v6;
    size_t v38 = v37 + v11;
    size_t v39 = v37 + v12;
    size_t v40 = v37 + v13;
    size_t v41 = v37 + v14;
    int8_t v42 = v5[v37];
    int16_t v43 = (int16_t) v42;
    vint32m1_t v44 = v26;
    vint32m1_t v45 = __riscv_vwmacc_vx_i32m1(v44, v43, v36, v33);
    int8_t v46 = v5[v38];
    int16_t v47 = (int16_t) v46;
    vint32m1_t v48 = v27;
    vint32m1_t v49 = __riscv_vwmacc_vx_i32m1(v48, v47, v36, v33);
    int8_t v50 = v5[v39];
    int16_t v51 = (int16_t) v50;
    vint32m1_t v52 = v28;
    vint32m1_t v53 = __riscv_vwmacc_vx_i32m1(v52, v51, v36, v33);
    int8_t v54 = v5[v40];
    int16_t v55 = (int16_t) v54;
    vint32m1_t v56 = v29;
    vint32m1_t v57 = __riscv_vwmacc_vx_i32m1(v56, v55, v36, v33);
    int8_t v58 = v5[v41];
    int16_t v59 = (int16_t) v58;
    vint32m1_t v60 = v30;
    vint32m1_t v61 = __riscv_vwmacc_vx_i32m1(v60, v59, v36, v33);
    v26 = v45;
    v27 = v49;
    v28 = v53;
    v29 = v57;
    v30 = v61;
  }
  vint32m1_t v62 = v26;
  vint32m1_t v63 = v27;
  vint32m1_t v64 = v28;
  vint32m1_t v65 = v29;
  vint32m1_t v66 = v30;
  int32_t* v67 = &v8[v10];
  __riscv_vse32_v_i32m1(v67, v62, v18);
  int32_t* v68 = &v8[v9];
  __riscv_vse32_v_i32m1(v68, v63, v18);
  size_t v69 = v12 * v9;
  int32_t* v70 = &v8[v69];
  __riscv_vse32_v_i32m1(v70, v64, v18);
  size_t v71 = v13 * v9;
  int32_t* v72 = &v8[v71];
  __riscv_vse32_v_i32m1(v72, v65, v18);
  size_t v73 = v14 * v9;
  int32_t* v74 = &v8[v73];
  __riscv_vse32_v_i32m1(v74, v66, v18);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_8x5_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 8
//   NR         : 5
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_8x5_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v18);
  size_t v25 = v11 * v9;
  size_t v26 = v10 + v25;
  int32_t* v27 = &v8[v26];
  vint32m1_t v28 = __riscv_vle32_v_i32m1(v27, v18);
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v18);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v18);
  size_t v37 = v14 * v9;
  size_t v38 = v10 + v37;
  int32_t* v39 = &v8[v38];
  vint32m1_t v40 = __riscv_vle32_v_i32m1(v39, v18);
  vint32m1_t v41;
  v41 = v24;
  vint32m1_t v42;
  v42 = v28;
  vint32m1_t v43;
  v43 = v32;
  vint32m1_t v44;
  v44 = v36;
  vint32m1_t v45;
  v45 = v40;
  for (size_t i46 = v10; i46 < v1; i46 += v11) {
    size_t v47 = i46 * v4;
    size_t v48 = 8;
    int8_t* v49 = &v3[v47];
    vint8mf4_t v50 = __riscv_vle8_v_i8mf4(v49, v48);
    vint16mf2_t v51 = __riscv_vsext_vf2_i16mf2(v50, v48);
    size_t v52 = i46 * v6;
    size_t v53 = v52 + v11;
    size_t v54 = v52 + v12;
    size_t v55 = v52 + v13;
    size_t v56 = v52 + v14;
    int8_t v57 = v5[v52];
    int16_t v58 = (int16_t) v57;
    vint32m1_t v59 = v41;
    vint32m1_t v60 = __riscv_vwmacc_vx_i32m1(v59, v58, v51, v48);
    int8_t v61 = v5[v53];
    int16_t v62 = (int16_t) v61;
    vint32m1_t v63 = v42;
    vint32m1_t v64 = __riscv_vwmacc_vx_i32m1(v63, v62, v51, v48);
    int8_t v65 = v5[v54];
    int16_t v66 = (int16_t) v65;
    vint32m1_t v67 = v43;
    vint32m1_t v68 = __riscv_vwmacc_vx_i32m1(v67, v66, v51, v48);
    int8_t v69 = v5[v55];
    int16_t v70 = (int16_t) v69;
    vint32m1_t v71 = v44;
    vint32m1_t v72 = __riscv_vwmacc_vx_i32m1(v71, v70, v51, v48);
    int8_t v73 = v5[v56];
    int16_t v74 = (int16_t) v73;
    vint32m1_t v75 = v45;
    vint32m1_t v76 = __riscv_vwmacc_vx_i32m1(v75, v74, v51, v48);
    v41 = v60;
    v42 = v64;
    v43 = v68;
    v44 = v72;
    v45 = v76;
  }
  vint32m1_t v77 = v41;
  vint32m1_t v78 = v42;
  vint32m1_t v79 = v43;
  vint32m1_t v80 = v44;
  vint32m1_t v81 = v45;
  int32_t* v82 = &v8[v10];
  __riscv_vse32_v_i32m1(v82, v77, v18);
  int32_t* v83 = &v8[v9];
  __riscv_vse32_v_i32m1(v83, v78, v18);
  size_t v84 = v12 * v9;
  int32_t* v85 = &v8[v84];
  __riscv_vse32_v_i32m1(v85, v79, v18);
  size_t v86 = v13 * v9;
  int32_t* v87 = &v8[v86];
  __riscv_vse32_v_i32m1(v87, v80, v18);
  size_t v88 = v14 * v9;
  int32_t* v89 = &v8[v88];
  __riscv_vse32_v_i32m1(v89, v81, v18);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_8x6_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 8
//   NR         : 6
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_8x6_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v22 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v25 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v26 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v27;
  v27 = v21;
  vint32m1_t v28;
  v28 = v22;
  vint32m1_t v29;
  v29 = v23;
  vint32m1_t v30;
  v30 = v24;
  vint32m1_t v31;
  v31 = v25;
  vint32m1_t v32;
  v32 = v26;
  for (size_t i33 = v10; i33 < v1; i33 += v11) {
    size_t v34 = i33 * v4;
    size_t v35 = 8;
    int8_t* v36 = &v3[v34];
    vint8mf4_t v37 = __riscv_vle8_v_i8mf4(v36, v35);
    vint16mf2_t v38 = __riscv_vsext_vf2_i16mf2(v37, v35);
    size_t v39 = i33 * v6;
    size_t v40 = v39 + v11;
    size_t v41 = v39 + v12;
    size_t v42 = v39 + v13;
    size_t v43 = v39 + v14;
    size_t v44 = v39 + v15;
    int8_t v45 = v5[v39];
    int16_t v46 = (int16_t) v45;
    vint32m1_t v47 = v27;
    vint32m1_t v48 = __riscv_vwmacc_vx_i32m1(v47, v46, v38, v35);
    int8_t v49 = v5[v40];
    int16_t v50 = (int16_t) v49;
    vint32m1_t v51 = v28;
    vint32m1_t v52 = __riscv_vwmacc_vx_i32m1(v51, v50, v38, v35);
    int8_t v53 = v5[v41];
    int16_t v54 = (int16_t) v53;
    vint32m1_t v55 = v29;
    vint32m1_t v56 = __riscv_vwmacc_vx_i32m1(v55, v54, v38, v35);
    int8_t v57 = v5[v42];
    int16_t v58 = (int16_t) v57;
    vint32m1_t v59 = v30;
    vint32m1_t v60 = __riscv_vwmacc_vx_i32m1(v59, v58, v38, v35);
    int8_t v61 = v5[v43];
    int16_t v62 = (int16_t) v61;
    vint32m1_t v63 = v31;
    vint32m1_t v64 = __riscv_vwmacc_vx_i32m1(v63, v62, v38, v35);
    int8_t v65 = v5[v44];
    int16_t v66 = (int16_t) v65;
    vint32m1_t v67 = v32;
    vint32m1_t v68 = __riscv_vwmacc_vx_i32m1(v67, v66, v38, v35);
    v27 = v48;
    v28 = v52;
    v29 = v56;
    v30 = v60;
    v31 = v64;
    v32 = v68;
  }
  vint32m1_t v69 = v27;
  vint32m1_t v70 = v28;
  vint32m1_t v71 = v29;
  vint32m1_t v72 = v30;
  vint32m1_t v73 = v31;
  vint32m1_t v74 = v32;
  int32_t* v75 = &v8[v10];
  __riscv_vse32_v_i32m1(v75, v69, v18);
  int32_t* v76 = &v8[v9];
  __riscv_vse32_v_i32m1(v76, v70, v18);
  size_t v77 = v12 * v9;
  int32_t* v78 = &v8[v77];
  __riscv_vse32_v_i32m1(v78, v71, v18);
  size_t v79 = v13 * v9;
  int32_t* v80 = &v8[v79];
  __riscv_vse32_v_i32m1(v80, v72, v18);
  size_t v81 = v14 * v9;
  int32_t* v82 = &v8[v81];
  __riscv_vse32_v_i32m1(v82, v73, v18);
  size_t v83 = v15 * v9;
  int32_t* v84 = &v8[v83];
  __riscv_vse32_v_i32m1(v84, v74, v18);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_8x6_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 8
//   NR         : 6
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_8x6_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v18);
  size_t v25 = v11 * v9;
  size_t v26 = v10 + v25;
  int32_t* v27 = &v8[v26];
  vint32m1_t v28 = __riscv_vle32_v_i32m1(v27, v18);
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v18);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v18);
  size_t v37 = v14 * v9;
  size_t v38 = v10 + v37;
  int32_t* v39 = &v8[v38];
  vint32m1_t v40 = __riscv_vle32_v_i32m1(v39, v18);
  size_t v41 = v15 * v9;
  size_t v42 = v10 + v41;
  int32_t* v43 = &v8[v42];
  vint32m1_t v44 = __riscv_vle32_v_i32m1(v43, v18);
  vint32m1_t v45;
  v45 = v24;
  vint32m1_t v46;
  v46 = v28;
  vint32m1_t v47;
  v47 = v32;
  vint32m1_t v48;
  v48 = v36;
  vint32m1_t v49;
  v49 = v40;
  vint32m1_t v50;
  v50 = v44;
  for (size_t i51 = v10; i51 < v1; i51 += v11) {
    size_t v52 = i51 * v4;
    size_t v53 = 8;
    int8_t* v54 = &v3[v52];
    vint8mf4_t v55 = __riscv_vle8_v_i8mf4(v54, v53);
    vint16mf2_t v56 = __riscv_vsext_vf2_i16mf2(v55, v53);
    size_t v57 = i51 * v6;
    size_t v58 = v57 + v11;
    size_t v59 = v57 + v12;
    size_t v60 = v57 + v13;
    size_t v61 = v57 + v14;
    size_t v62 = v57 + v15;
    int8_t v63 = v5[v57];
    int16_t v64 = (int16_t) v63;
    vint32m1_t v65 = v45;
    vint32m1_t v66 = __riscv_vwmacc_vx_i32m1(v65, v64, v56, v53);
    int8_t v67 = v5[v58];
    int16_t v68 = (int16_t) v67;
    vint32m1_t v69 = v46;
    vint32m1_t v70 = __riscv_vwmacc_vx_i32m1(v69, v68, v56, v53);
    int8_t v71 = v5[v59];
    int16_t v72 = (int16_t) v71;
    vint32m1_t v73 = v47;
    vint32m1_t v74 = __riscv_vwmacc_vx_i32m1(v73, v72, v56, v53);
    int8_t v75 = v5[v60];
    int16_t v76 = (int16_t) v75;
    vint32m1_t v77 = v48;
    vint32m1_t v78 = __riscv_vwmacc_vx_i32m1(v77, v76, v56, v53);
    int8_t v79 = v5[v61];
    int16_t v80 = (int16_t) v79;
    vint32m1_t v81 = v49;
    vint32m1_t v82 = __riscv_vwmacc_vx_i32m1(v81, v80, v56, v53);
    int8_t v83 = v5[v62];
    int16_t v84 = (int16_t) v83;
    vint32m1_t v85 = v50;
    vint32m1_t v86 = __riscv_vwmacc_vx_i32m1(v85, v84, v56, v53);
    v45 = v66;
    v46 = v70;
    v47 = v74;
    v48 = v78;
    v49 = v82;
    v50 = v86;
  }
  vint32m1_t v87 = v45;
  vint32m1_t v88 = v46;
  vint32m1_t v89 = v47;
  vint32m1_t v90 = v48;
  vint32m1_t v91 = v49;
  vint32m1_t v92 = v50;
  int32_t* v93 = &v8[v10];
  __riscv_vse32_v_i32m1(v93, v87, v18);
  int32_t* v94 = &v8[v9];
  __riscv_vse32_v_i32m1(v94, v88, v18);
  size_t v95 = v12 * v9;
  int32_t* v96 = &v8[v95];
  __riscv_vse32_v_i32m1(v96, v89, v18);
  size_t v97 = v13 * v9;
  int32_t* v98 = &v8[v97];
  __riscv_vse32_v_i32m1(v98, v90, v18);
  size_t v99 = v14 * v9;
  int32_t* v100 = &v8[v99];
  __riscv_vse32_v_i32m1(v100, v91, v18);
  size_t v101 = v15 * v9;
  int32_t* v102 = &v8[v101];
  __riscv_vse32_v_i32m1(v102, v92, v18);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_8x7_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 8
//   NR         : 7
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_8x7_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v22 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v25 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v26 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v27 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v28;
  v28 = v21;
  vint32m1_t v29;
  v29 = v22;
  vint32m1_t v30;
  v30 = v23;
  vint32m1_t v31;
  v31 = v24;
  vint32m1_t v32;
  v32 = v25;
  vint32m1_t v33;
  v33 = v26;
  vint32m1_t v34;
  v34 = v27;
  for (size_t i35 = v10; i35 < v1; i35 += v11) {
    size_t v36 = i35 * v4;
    size_t v37 = 8;
    int8_t* v38 = &v3[v36];
    vint8mf4_t v39 = __riscv_vle8_v_i8mf4(v38, v37);
    vint16mf2_t v40 = __riscv_vsext_vf2_i16mf2(v39, v37);
    size_t v41 = i35 * v6;
    size_t v42 = v41 + v11;
    size_t v43 = v41 + v12;
    size_t v44 = v41 + v13;
    size_t v45 = v41 + v14;
    size_t v46 = v41 + v15;
    size_t v47 = v41 + v16;
    int8_t v48 = v5[v41];
    int16_t v49 = (int16_t) v48;
    vint32m1_t v50 = v28;
    vint32m1_t v51 = __riscv_vwmacc_vx_i32m1(v50, v49, v40, v37);
    int8_t v52 = v5[v42];
    int16_t v53 = (int16_t) v52;
    vint32m1_t v54 = v29;
    vint32m1_t v55 = __riscv_vwmacc_vx_i32m1(v54, v53, v40, v37);
    int8_t v56 = v5[v43];
    int16_t v57 = (int16_t) v56;
    vint32m1_t v58 = v30;
    vint32m1_t v59 = __riscv_vwmacc_vx_i32m1(v58, v57, v40, v37);
    int8_t v60 = v5[v44];
    int16_t v61 = (int16_t) v60;
    vint32m1_t v62 = v31;
    vint32m1_t v63 = __riscv_vwmacc_vx_i32m1(v62, v61, v40, v37);
    int8_t v64 = v5[v45];
    int16_t v65 = (int16_t) v64;
    vint32m1_t v66 = v32;
    vint32m1_t v67 = __riscv_vwmacc_vx_i32m1(v66, v65, v40, v37);
    int8_t v68 = v5[v46];
    int16_t v69 = (int16_t) v68;
    vint32m1_t v70 = v33;
    vint32m1_t v71 = __riscv_vwmacc_vx_i32m1(v70, v69, v40, v37);
    int8_t v72 = v5[v47];
    int16_t v73 = (int16_t) v72;
    vint32m1_t v74 = v34;
    vint32m1_t v75 = __riscv_vwmacc_vx_i32m1(v74, v73, v40, v37);
    v28 = v51;
    v29 = v55;
    v30 = v59;
    v31 = v63;
    v32 = v67;
    v33 = v71;
    v34 = v75;
  }
  vint32m1_t v76 = v28;
  vint32m1_t v77 = v29;
  vint32m1_t v78 = v30;
  vint32m1_t v79 = v31;
  vint32m1_t v80 = v32;
  vint32m1_t v81 = v33;
  vint32m1_t v82 = v34;
  int32_t* v83 = &v8[v10];
  __riscv_vse32_v_i32m1(v83, v76, v18);
  int32_t* v84 = &v8[v9];
  __riscv_vse32_v_i32m1(v84, v77, v18);
  size_t v85 = v12 * v9;
  int32_t* v86 = &v8[v85];
  __riscv_vse32_v_i32m1(v86, v78, v18);
  size_t v87 = v13 * v9;
  int32_t* v88 = &v8[v87];
  __riscv_vse32_v_i32m1(v88, v79, v18);
  size_t v89 = v14 * v9;
  int32_t* v90 = &v8[v89];
  __riscv_vse32_v_i32m1(v90, v80, v18);
  size_t v91 = v15 * v9;
  int32_t* v92 = &v8[v91];
  __riscv_vse32_v_i32m1(v92, v81, v18);
  size_t v93 = v16 * v9;
  int32_t* v94 = &v8[v93];
  __riscv_vse32_v_i32m1(v94, v82, v18);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_8x7_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 8
//   NR         : 7
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_8x7_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v18);
  size_t v25 = v11 * v9;
  size_t v26 = v10 + v25;
  int32_t* v27 = &v8[v26];
  vint32m1_t v28 = __riscv_vle32_v_i32m1(v27, v18);
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v18);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v18);
  size_t v37 = v14 * v9;
  size_t v38 = v10 + v37;
  int32_t* v39 = &v8[v38];
  vint32m1_t v40 = __riscv_vle32_v_i32m1(v39, v18);
  size_t v41 = v15 * v9;
  size_t v42 = v10 + v41;
  int32_t* v43 = &v8[v42];
  vint32m1_t v44 = __riscv_vle32_v_i32m1(v43, v18);
  size_t v45 = v16 * v9;
  size_t v46 = v10 + v45;
  int32_t* v47 = &v8[v46];
  vint32m1_t v48 = __riscv_vle32_v_i32m1(v47, v18);
  vint32m1_t v49;
  v49 = v24;
  vint32m1_t v50;
  v50 = v28;
  vint32m1_t v51;
  v51 = v32;
  vint32m1_t v52;
  v52 = v36;
  vint32m1_t v53;
  v53 = v40;
  vint32m1_t v54;
  v54 = v44;
  vint32m1_t v55;
  v55 = v48;
  for (size_t i56 = v10; i56 < v1; i56 += v11) {
    size_t v57 = i56 * v4;
    size_t v58 = 8;
    int8_t* v59 = &v3[v57];
    vint8mf4_t v60 = __riscv_vle8_v_i8mf4(v59, v58);
    vint16mf2_t v61 = __riscv_vsext_vf2_i16mf2(v60, v58);
    size_t v62 = i56 * v6;
    size_t v63 = v62 + v11;
    size_t v64 = v62 + v12;
    size_t v65 = v62 + v13;
    size_t v66 = v62 + v14;
    size_t v67 = v62 + v15;
    size_t v68 = v62 + v16;
    int8_t v69 = v5[v62];
    int16_t v70 = (int16_t) v69;
    vint32m1_t v71 = v49;
    vint32m1_t v72 = __riscv_vwmacc_vx_i32m1(v71, v70, v61, v58);
    int8_t v73 = v5[v63];
    int16_t v74 = (int16_t) v73;
    vint32m1_t v75 = v50;
    vint32m1_t v76 = __riscv_vwmacc_vx_i32m1(v75, v74, v61, v58);
    int8_t v77 = v5[v64];
    int16_t v78 = (int16_t) v77;
    vint32m1_t v79 = v51;
    vint32m1_t v80 = __riscv_vwmacc_vx_i32m1(v79, v78, v61, v58);
    int8_t v81 = v5[v65];
    int16_t v82 = (int16_t) v81;
    vint32m1_t v83 = v52;
    vint32m1_t v84 = __riscv_vwmacc_vx_i32m1(v83, v82, v61, v58);
    int8_t v85 = v5[v66];
    int16_t v86 = (int16_t) v85;
    vint32m1_t v87 = v53;
    vint32m1_t v88 = __riscv_vwmacc_vx_i32m1(v87, v86, v61, v58);
    int8_t v89 = v5[v67];
    int16_t v90 = (int16_t) v89;
    vint32m1_t v91 = v54;
    vint32m1_t v92 = __riscv_vwmacc_vx_i32m1(v91, v90, v61, v58);
    int8_t v93 = v5[v68];
    int16_t v94 = (int16_t) v93;
    vint32m1_t v95 = v55;
    vint32m1_t v96 = __riscv_vwmacc_vx_i32m1(v95, v94, v61, v58);
    v49 = v72;
    v50 = v76;
    v51 = v80;
    v52 = v84;
    v53 = v88;
    v54 = v92;
    v55 = v96;
  }
  vint32m1_t v97 = v49;
  vint32m1_t v98 = v50;
  vint32m1_t v99 = v51;
  vint32m1_t v100 = v52;
  vint32m1_t v101 = v53;
  vint32m1_t v102 = v54;
  vint32m1_t v103 = v55;
  int32_t* v104 = &v8[v10];
  __riscv_vse32_v_i32m1(v104, v97, v18);
  int32_t* v105 = &v8[v9];
  __riscv_vse32_v_i32m1(v105, v98, v18);
  size_t v106 = v12 * v9;
  int32_t* v107 = &v8[v106];
  __riscv_vse32_v_i32m1(v107, v99, v18);
  size_t v108 = v13 * v9;
  int32_t* v109 = &v8[v108];
  __riscv_vse32_v_i32m1(v109, v100, v18);
  size_t v110 = v14 * v9;
  int32_t* v111 = &v8[v110];
  __riscv_vse32_v_i32m1(v111, v101, v18);
  size_t v112 = v15 * v9;
  int32_t* v113 = &v8[v112];
  __riscv_vse32_v_i32m1(v113, v102, v18);
  size_t v114 = v16 * v9;
  int32_t* v115 = &v8[v114];
  __riscv_vse32_v_i32m1(v115, v103, v18);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_8x8_b0_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 8
//   NR         : 8
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_8x8_b0_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v22 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v24 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v25 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v26 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v27 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v28 = __riscv_vmv_v_x_i32m1(v20, v18);
  vint32m1_t v29;
  v29 = v21;
  vint32m1_t v30;
  v30 = v22;
  vint32m1_t v31;
  v31 = v23;
  vint32m1_t v32;
  v32 = v24;
  vint32m1_t v33;
  v33 = v25;
  vint32m1_t v34;
  v34 = v26;
  vint32m1_t v35;
  v35 = v27;
  vint32m1_t v36;
  v36 = v28;
  for (size_t i37 = v10; i37 < v1; i37 += v11) {
    size_t v38 = i37 * v4;
    size_t v39 = 8;
    int8_t* v40 = &v3[v38];
    vint8mf4_t v41 = __riscv_vle8_v_i8mf4(v40, v39);
    vint16mf2_t v42 = __riscv_vsext_vf2_i16mf2(v41, v39);
    size_t v43 = i37 * v6;
    size_t v44 = v43 + v11;
    size_t v45 = v43 + v12;
    size_t v46 = v43 + v13;
    size_t v47 = v43 + v14;
    size_t v48 = v43 + v15;
    size_t v49 = v43 + v16;
    size_t v50 = v43 + v17;
    int8_t v51 = v5[v43];
    int16_t v52 = (int16_t) v51;
    vint32m1_t v53 = v29;
    vint32m1_t v54 = __riscv_vwmacc_vx_i32m1(v53, v52, v42, v39);
    int8_t v55 = v5[v44];
    int16_t v56 = (int16_t) v55;
    vint32m1_t v57 = v30;
    vint32m1_t v58 = __riscv_vwmacc_vx_i32m1(v57, v56, v42, v39);
    int8_t v59 = v5[v45];
    int16_t v60 = (int16_t) v59;
    vint32m1_t v61 = v31;
    vint32m1_t v62 = __riscv_vwmacc_vx_i32m1(v61, v60, v42, v39);
    int8_t v63 = v5[v46];
    int16_t v64 = (int16_t) v63;
    vint32m1_t v65 = v32;
    vint32m1_t v66 = __riscv_vwmacc_vx_i32m1(v65, v64, v42, v39);
    int8_t v67 = v5[v47];
    int16_t v68 = (int16_t) v67;
    vint32m1_t v69 = v33;
    vint32m1_t v70 = __riscv_vwmacc_vx_i32m1(v69, v68, v42, v39);
    int8_t v71 = v5[v48];
    int16_t v72 = (int16_t) v71;
    vint32m1_t v73 = v34;
    vint32m1_t v74 = __riscv_vwmacc_vx_i32m1(v73, v72, v42, v39);
    int8_t v75 = v5[v49];
    int16_t v76 = (int16_t) v75;
    vint32m1_t v77 = v35;
    vint32m1_t v78 = __riscv_vwmacc_vx_i32m1(v77, v76, v42, v39);
    int8_t v79 = v5[v50];
    int16_t v80 = (int16_t) v79;
    vint32m1_t v81 = v36;
    vint32m1_t v82 = __riscv_vwmacc_vx_i32m1(v81, v80, v42, v39);
    v29 = v54;
    v30 = v58;
    v31 = v62;
    v32 = v66;
    v33 = v70;
    v34 = v74;
    v35 = v78;
    v36 = v82;
  }
  vint32m1_t v83 = v29;
  vint32m1_t v84 = v30;
  vint32m1_t v85 = v31;
  vint32m1_t v86 = v32;
  vint32m1_t v87 = v33;
  vint32m1_t v88 = v34;
  vint32m1_t v89 = v35;
  vint32m1_t v90 = v36;
  int32_t* v91 = &v8[v10];
  __riscv_vse32_v_i32m1(v91, v83, v18);
  int32_t* v92 = &v8[v9];
  __riscv_vse32_v_i32m1(v92, v84, v18);
  size_t v93 = v12 * v9;
  int32_t* v94 = &v8[v93];
  __riscv_vse32_v_i32m1(v94, v85, v18);
  size_t v95 = v13 * v9;
  int32_t* v96 = &v8[v95];
  __riscv_vse32_v_i32m1(v96, v86, v18);
  size_t v97 = v14 * v9;
  int32_t* v98 = &v8[v97];
  __riscv_vse32_v_i32m1(v98, v87, v18);
  size_t v99 = v15 * v9;
  int32_t* v100 = &v8[v99];
  __riscv_vse32_v_i32m1(v100, v88, v18);
  size_t v101 = v16 * v9;
  int32_t* v102 = &v8[v101];
  __riscv_vse32_v_i32m1(v102, v89, v18);
  size_t v103 = v17 * v9;
  int32_t* v104 = &v8[v103];
  __riscv_vse32_v_i32m1(v104, v90, v18);
  return;
}



// ------------------------------------------------------------
// Merged from xdsl_api_gemm_8x8_b1_i8i8i32.cpp
// ------------------------------------------------------------
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-06-29 11:41:11
//   MR         : 8
//   NR         : 8
//   VLEN       : 8  (i8i8i32 elements per vector register)
//   Precision  : i8i8i32
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_8x8_b1_col_i8i8i32(void* ctxt, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
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
  vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v18);
  size_t v25 = v11 * v9;
  size_t v26 = v10 + v25;
  int32_t* v27 = &v8[v26];
  vint32m1_t v28 = __riscv_vle32_v_i32m1(v27, v18);
  size_t v29 = v12 * v9;
  size_t v30 = v10 + v29;
  int32_t* v31 = &v8[v30];
  vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v18);
  size_t v33 = v13 * v9;
  size_t v34 = v10 + v33;
  int32_t* v35 = &v8[v34];
  vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v18);
  size_t v37 = v14 * v9;
  size_t v38 = v10 + v37;
  int32_t* v39 = &v8[v38];
  vint32m1_t v40 = __riscv_vle32_v_i32m1(v39, v18);
  size_t v41 = v15 * v9;
  size_t v42 = v10 + v41;
  int32_t* v43 = &v8[v42];
  vint32m1_t v44 = __riscv_vle32_v_i32m1(v43, v18);
  size_t v45 = v16 * v9;
  size_t v46 = v10 + v45;
  int32_t* v47 = &v8[v46];
  vint32m1_t v48 = __riscv_vle32_v_i32m1(v47, v18);
  size_t v49 = v17 * v9;
  size_t v50 = v10 + v49;
  int32_t* v51 = &v8[v50];
  vint32m1_t v52 = __riscv_vle32_v_i32m1(v51, v18);
  vint32m1_t v53;
  v53 = v24;
  vint32m1_t v54;
  v54 = v28;
  vint32m1_t v55;
  v55 = v32;
  vint32m1_t v56;
  v56 = v36;
  vint32m1_t v57;
  v57 = v40;
  vint32m1_t v58;
  v58 = v44;
  vint32m1_t v59;
  v59 = v48;
  vint32m1_t v60;
  v60 = v52;
  for (size_t i61 = v10; i61 < v1; i61 += v11) {
    size_t v62 = i61 * v4;
    size_t v63 = 8;
    int8_t* v64 = &v3[v62];
    vint8mf4_t v65 = __riscv_vle8_v_i8mf4(v64, v63);
    vint16mf2_t v66 = __riscv_vsext_vf2_i16mf2(v65, v63);
    size_t v67 = i61 * v6;
    size_t v68 = v67 + v11;
    size_t v69 = v67 + v12;
    size_t v70 = v67 + v13;
    size_t v71 = v67 + v14;
    size_t v72 = v67 + v15;
    size_t v73 = v67 + v16;
    size_t v74 = v67 + v17;
    int8_t v75 = v5[v67];
    int16_t v76 = (int16_t) v75;
    vint32m1_t v77 = v53;
    vint32m1_t v78 = __riscv_vwmacc_vx_i32m1(v77, v76, v66, v63);
    int8_t v79 = v5[v68];
    int16_t v80 = (int16_t) v79;
    vint32m1_t v81 = v54;
    vint32m1_t v82 = __riscv_vwmacc_vx_i32m1(v81, v80, v66, v63);
    int8_t v83 = v5[v69];
    int16_t v84 = (int16_t) v83;
    vint32m1_t v85 = v55;
    vint32m1_t v86 = __riscv_vwmacc_vx_i32m1(v85, v84, v66, v63);
    int8_t v87 = v5[v70];
    int16_t v88 = (int16_t) v87;
    vint32m1_t v89 = v56;
    vint32m1_t v90 = __riscv_vwmacc_vx_i32m1(v89, v88, v66, v63);
    int8_t v91 = v5[v71];
    int16_t v92 = (int16_t) v91;
    vint32m1_t v93 = v57;
    vint32m1_t v94 = __riscv_vwmacc_vx_i32m1(v93, v92, v66, v63);
    int8_t v95 = v5[v72];
    int16_t v96 = (int16_t) v95;
    vint32m1_t v97 = v58;
    vint32m1_t v98 = __riscv_vwmacc_vx_i32m1(v97, v96, v66, v63);
    int8_t v99 = v5[v73];
    int16_t v100 = (int16_t) v99;
    vint32m1_t v101 = v59;
    vint32m1_t v102 = __riscv_vwmacc_vx_i32m1(v101, v100, v66, v63);
    int8_t v103 = v5[v74];
    int16_t v104 = (int16_t) v103;
    vint32m1_t v105 = v60;
    vint32m1_t v106 = __riscv_vwmacc_vx_i32m1(v105, v104, v66, v63);
    v53 = v78;
    v54 = v82;
    v55 = v86;
    v56 = v90;
    v57 = v94;
    v58 = v98;
    v59 = v102;
    v60 = v106;
  }
  vint32m1_t v107 = v53;
  vint32m1_t v108 = v54;
  vint32m1_t v109 = v55;
  vint32m1_t v110 = v56;
  vint32m1_t v111 = v57;
  vint32m1_t v112 = v58;
  vint32m1_t v113 = v59;
  vint32m1_t v114 = v60;
  int32_t* v115 = &v8[v10];
  __riscv_vse32_v_i32m1(v115, v107, v18);
  int32_t* v116 = &v8[v9];
  __riscv_vse32_v_i32m1(v116, v108, v18);
  size_t v117 = v12 * v9;
  int32_t* v118 = &v8[v117];
  __riscv_vse32_v_i32m1(v118, v109, v18);
  size_t v119 = v13 * v9;
  int32_t* v120 = &v8[v119];
  __riscv_vse32_v_i32m1(v120, v110, v18);
  size_t v121 = v14 * v9;
  int32_t* v122 = &v8[v121];
  __riscv_vse32_v_i32m1(v122, v111, v18);
  size_t v123 = v15 * v9;
  int32_t* v124 = &v8[v123];
  __riscv_vse32_v_i32m1(v124, v112, v18);
  size_t v125 = v16 * v9;
  int32_t* v126 = &v8[v125];
  __riscv_vse32_v_i32m1(v126, v113, v18);
  size_t v127 = v17 * v9;
  int32_t* v128 = &v8[v127];
  __riscv_vse32_v_i32m1(v128, v114, v18);
  return;
}



