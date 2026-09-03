#include "kernels_RVV_20x6_fp32.h"



#include <stdio.h>
#include <stdlib.h>

#include <riscv_vector.h>


// gemm_RVV_10x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 10] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 10] @DRAM
// )
void gemm_RVV_10x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(2));
}

// gemm_RVV_10x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 10] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 10] @DRAM
// )
void gemm_RVV_10x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[8],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(2));
}

// gemm_RVV_10x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 10] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 10] @DRAM
// )
void gemm_RVV_10x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(2));
}

// gemm_RVV_10x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 10] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 10] @DRAM
// )
void gemm_RVV_10x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[8],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(2));
}

// gemm_RVV_10x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 10] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 10] @DRAM
// )
void gemm_RVV_10x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_regt_2,(2));
}

// gemm_RVV_10x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 10] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 10] @DRAM
// )
void gemm_RVV_10x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[8],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(2));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_regt_2,(2));
}

// gemm_RVV_10x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 10] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 10] @DRAM
// )
void gemm_RVV_10x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_regt_3,(2));
}

// gemm_RVV_10x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 10] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 10] @DRAM
// )
void gemm_RVV_10x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[8],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(2));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(2));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_regt_3,(2));
}

// gemm_RVV_10x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 10] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 10] @DRAM
// )
void gemm_RVV_10x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_regt_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_regt_4,(2));
}

// gemm_RVV_10x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 10] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 10] @DRAM
// )
void gemm_RVV_10x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[8],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(2));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(2));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(2));
C_regt_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_4_1 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_regt_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_regt_4,(2));
}

// gemm_RVV_10x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 10] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 10] @DRAM
// )
void gemm_RVV_10x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_5_1;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_5 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_5_1 = __riscv_vfmacc_vv_f32m1(C_reg_5_1, A_reg_1, B_reg_5,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(2));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4], C_reg_5_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_regt_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_regt_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8], C_regt_5,(2));
}

// gemm_RVV_10x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 10] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 10] @DRAM
// )
void gemm_RVV_10x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_5_1;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[8],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(2));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(2));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(2));
C_regt_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8],(2));
C_regt_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_4_1 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_5_1 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_5_1 = __riscv_vfmacc_vv_f32m1(C_reg_5_1, A_reg_1, B_reg_5,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(2));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4], C_reg_5_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_regt_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_regt_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8], C_regt_5,(2));
}

// gemm_RVV_11x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 11] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 11] @DRAM
// )
void gemm_RVV_11x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(3));
}

// gemm_RVV_11x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 11] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 11] @DRAM
// )
void gemm_RVV_11x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[8],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(3));
}

// gemm_RVV_11x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 11] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 11] @DRAM
// )
void gemm_RVV_11x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(3));
}

// gemm_RVV_11x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 11] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 11] @DRAM
// )
void gemm_RVV_11x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[8],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(3));
}

// gemm_RVV_11x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 11] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 11] @DRAM
// )
void gemm_RVV_11x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_regt_2,(3));
}

// gemm_RVV_11x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 11] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 11] @DRAM
// )
void gemm_RVV_11x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[8],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(3));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_regt_2,(3));
}

// gemm_RVV_11x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 11] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 11] @DRAM
// )
void gemm_RVV_11x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_regt_3,(3));
}

// gemm_RVV_11x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 11] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 11] @DRAM
// )
void gemm_RVV_11x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[8],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(3));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(3));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_regt_3,(3));
}

// gemm_RVV_11x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 11] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 11] @DRAM
// )
void gemm_RVV_11x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_regt_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_regt_4,(3));
}

// gemm_RVV_11x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 11] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 11] @DRAM
// )
void gemm_RVV_11x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[8],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(3));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(3));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(3));
C_regt_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_4_1 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_regt_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_regt_4,(3));
}

// gemm_RVV_11x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 11] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 11] @DRAM
// )
void gemm_RVV_11x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_5_1;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_5 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_5_1 = __riscv_vfmacc_vv_f32m1(C_reg_5_1, A_reg_1, B_reg_5,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(3));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4], C_reg_5_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_regt_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_regt_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8], C_regt_5,(3));
}

// gemm_RVV_11x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 11] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 11] @DRAM
// )
void gemm_RVV_11x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_5_1;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[8],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(3));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(3));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(3));
C_regt_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8],(3));
C_regt_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_4_1 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_5_1 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_5_1 = __riscv_vfmacc_vv_f32m1(C_reg_5_1, A_reg_1, B_reg_5,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(3));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4], C_reg_5_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_regt_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_regt_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8], C_regt_5,(3));
}

// gemm_RVV_12x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 12] @DRAM
// )
void gemm_RVV_12x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
}

// gemm_RVV_12x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 12] @DRAM
// )
void gemm_RVV_12x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
}

// gemm_RVV_12x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 12] @DRAM
// )
void gemm_RVV_12x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
}

// gemm_RVV_12x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 12] @DRAM
// )
void gemm_RVV_12x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
}

// gemm_RVV_12x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 12] @DRAM
// )
void gemm_RVV_12x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
}

// gemm_RVV_12x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 12] @DRAM
// )
void gemm_RVV_12x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
}

// gemm_RVV_12x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 12] @DRAM
// )
void gemm_RVV_12x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
}

// gemm_RVV_12x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 12] @DRAM
// )
void gemm_RVV_12x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
}

// gemm_RVV_12x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 12] @DRAM
// )
void gemm_RVV_12x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
}

// gemm_RVV_12x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 12] @DRAM
// )
void gemm_RVV_12x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_4_1 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4],(4));
C_reg_4_2 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
}

// gemm_RVV_12x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 12] @DRAM
// )
void gemm_RVV_12x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_5_1;
vfloat32m1_t C_reg_5_2;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_5_1 = __riscv_vfmacc_vv_f32m1(C_reg_5_1, A_reg_1, B_reg_5,(4));
  C_reg_5_2 = __riscv_vfmacc_vv_f32m1(C_reg_5_2, A_reg_2, B_reg_5,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4], C_reg_5_1,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8], C_reg_5_2,(4));
}

// gemm_RVV_12x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 12] @DRAM
// )
void gemm_RVV_12x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_5_1;
vfloat32m1_t C_reg_5_2;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_4_1 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4],(4));
C_reg_4_2 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_5_1 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4],(4));
C_reg_5_2 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_5_1 = __riscv_vfmacc_vv_f32m1(C_reg_5_1, A_reg_1, B_reg_5,(4));
  C_reg_5_2 = __riscv_vfmacc_vv_f32m1(C_reg_5_2, A_reg_2, B_reg_5,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4], C_reg_5_1,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8], C_reg_5_2,(4));
}

// gemm_RVV_13x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 13] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 13] @DRAM
// )
void gemm_RVV_13x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(1));
}

// gemm_RVV_13x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 13] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 13] @DRAM
// )
void gemm_RVV_13x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[12],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(1));
}

// gemm_RVV_13x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 13] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 13] @DRAM
// )
void gemm_RVV_13x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_regt_1,(1));
}

// gemm_RVV_13x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 13] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 13] @DRAM
// )
void gemm_RVV_13x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[12],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_regt_1,(1));
}

// gemm_RVV_13x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 13] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 13] @DRAM
// )
void gemm_RVV_13x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_regt_2,(1));
}

// gemm_RVV_13x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 13] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 13] @DRAM
// )
void gemm_RVV_13x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[12],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(1));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_regt_2,(1));
}

// gemm_RVV_13x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 13] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 13] @DRAM
// )
void gemm_RVV_13x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_regt_3,(1));
}

// gemm_RVV_13x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 13] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 13] @DRAM
// )
void gemm_RVV_13x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[12],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(1));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12],(1));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_regt_3,(1));
}

// gemm_RVV_13x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 13] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 13] @DRAM
// )
void gemm_RVV_13x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_regt_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12], C_regt_4,(1));
}

// gemm_RVV_13x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 13] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 13] @DRAM
// )
void gemm_RVV_13x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[12],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(1));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12],(1));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12],(1));
C_regt_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_4_1 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4],(4));
C_reg_4_2 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_regt_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12], C_regt_4,(1));
}

// gemm_RVV_13x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 13] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 13] @DRAM
// )
void gemm_RVV_13x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_5_1;
vfloat32m1_t C_reg_5_2;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_5 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_5_1 = __riscv_vfmacc_vv_f32m1(C_reg_5_1, A_reg_1, B_reg_5,(4));
  C_reg_5_2 = __riscv_vfmacc_vv_f32m1(C_reg_5_2, A_reg_2, B_reg_5,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(1));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4], C_reg_5_1,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8], C_reg_5_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_regt_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12], C_regt_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 12], C_regt_5,(1));
}

// gemm_RVV_13x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 13] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 13] @DRAM
// )
void gemm_RVV_13x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_5_1;
vfloat32m1_t C_reg_5_2;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[12],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(1));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12],(1));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12],(1));
C_regt_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12],(1));
C_regt_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 12],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_4_1 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4],(4));
C_reg_4_2 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_5_1 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4],(4));
C_reg_5_2 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_5_1 = __riscv_vfmacc_vv_f32m1(C_reg_5_1, A_reg_1, B_reg_5,(4));
  C_reg_5_2 = __riscv_vfmacc_vv_f32m1(C_reg_5_2, A_reg_2, B_reg_5,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(1));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4], C_reg_5_1,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8], C_reg_5_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_regt_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12], C_regt_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 12], C_regt_5,(1));
}

// gemm_RVV_14x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 14] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 14] @DRAM
// )
void gemm_RVV_14x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(2));
}

// gemm_RVV_14x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 14] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 14] @DRAM
// )
void gemm_RVV_14x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[12],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(2));
}

// gemm_RVV_14x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 14] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 14] @DRAM
// )
void gemm_RVV_14x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_regt_1,(2));
}

// gemm_RVV_14x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 14] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 14] @DRAM
// )
void gemm_RVV_14x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[12],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_regt_1,(2));
}

// gemm_RVV_14x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 14] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 14] @DRAM
// )
void gemm_RVV_14x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_regt_2,(2));
}

// gemm_RVV_14x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 14] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 14] @DRAM
// )
void gemm_RVV_14x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[12],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(2));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_regt_2,(2));
}

// gemm_RVV_14x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 14] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 14] @DRAM
// )
void gemm_RVV_14x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_regt_3,(2));
}

// gemm_RVV_14x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 14] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 14] @DRAM
// )
void gemm_RVV_14x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[12],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(2));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12],(2));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_regt_3,(2));
}

// gemm_RVV_14x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 14] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 14] @DRAM
// )
void gemm_RVV_14x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_regt_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12], C_regt_4,(2));
}

// gemm_RVV_14x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 14] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 14] @DRAM
// )
void gemm_RVV_14x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[12],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(2));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12],(2));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12],(2));
C_regt_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_4_1 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4],(4));
C_reg_4_2 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_regt_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12], C_regt_4,(2));
}

// gemm_RVV_14x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 14] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 14] @DRAM
// )
void gemm_RVV_14x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_5_1;
vfloat32m1_t C_reg_5_2;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_5 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_5_1 = __riscv_vfmacc_vv_f32m1(C_reg_5_1, A_reg_1, B_reg_5,(4));
  C_reg_5_2 = __riscv_vfmacc_vv_f32m1(C_reg_5_2, A_reg_2, B_reg_5,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(2));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4], C_reg_5_1,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8], C_reg_5_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_regt_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12], C_regt_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 12], C_regt_5,(2));
}

// gemm_RVV_14x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 14] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 14] @DRAM
// )
void gemm_RVV_14x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_5_1;
vfloat32m1_t C_reg_5_2;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[12],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(2));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12],(2));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12],(2));
C_regt_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12],(2));
C_regt_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 12],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_4_1 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4],(4));
C_reg_4_2 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_5_1 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4],(4));
C_reg_5_2 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_5_1 = __riscv_vfmacc_vv_f32m1(C_reg_5_1, A_reg_1, B_reg_5,(4));
  C_reg_5_2 = __riscv_vfmacc_vv_f32m1(C_reg_5_2, A_reg_2, B_reg_5,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(2));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4], C_reg_5_1,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8], C_reg_5_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_regt_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12], C_regt_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 12], C_regt_5,(2));
}

// gemm_RVV_15x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 15] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 15] @DRAM
// )
void gemm_RVV_15x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(3));
}

// gemm_RVV_15x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 15] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 15] @DRAM
// )
void gemm_RVV_15x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[12],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(3));
}

// gemm_RVV_15x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 15] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 15] @DRAM
// )
void gemm_RVV_15x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_regt_1,(3));
}

// gemm_RVV_15x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 15] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 15] @DRAM
// )
void gemm_RVV_15x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[12],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_regt_1,(3));
}

// gemm_RVV_15x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 15] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 15] @DRAM
// )
void gemm_RVV_15x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_regt_2,(3));
}

// gemm_RVV_15x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 15] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 15] @DRAM
// )
void gemm_RVV_15x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[12],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(3));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_regt_2,(3));
}

// gemm_RVV_15x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 15] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 15] @DRAM
// )
void gemm_RVV_15x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_regt_3,(3));
}

// gemm_RVV_15x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 15] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 15] @DRAM
// )
void gemm_RVV_15x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[12],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(3));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12],(3));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_regt_3,(3));
}

// gemm_RVV_15x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 15] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 15] @DRAM
// )
void gemm_RVV_15x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_regt_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12], C_regt_4,(3));
}

// gemm_RVV_15x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 15] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 15] @DRAM
// )
void gemm_RVV_15x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[12],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(3));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12],(3));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12],(3));
C_regt_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_4_1 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4],(4));
C_reg_4_2 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_regt_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12], C_regt_4,(3));
}

// gemm_RVV_15x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 15] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 15] @DRAM
// )
void gemm_RVV_15x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_5_1;
vfloat32m1_t C_reg_5_2;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_5 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_5_1 = __riscv_vfmacc_vv_f32m1(C_reg_5_1, A_reg_1, B_reg_5,(4));
  C_reg_5_2 = __riscv_vfmacc_vv_f32m1(C_reg_5_2, A_reg_2, B_reg_5,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(3));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4], C_reg_5_1,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8], C_reg_5_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_regt_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12], C_regt_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 12], C_regt_5,(3));
}

// gemm_RVV_15x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 15] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 15] @DRAM
// )
void gemm_RVV_15x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_5_1;
vfloat32m1_t C_reg_5_2;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[12],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(3));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12],(3));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12],(3));
C_regt_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12],(3));
C_regt_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 12],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_4_1 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4],(4));
C_reg_4_2 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_5_1 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4],(4));
C_reg_5_2 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_5_1 = __riscv_vfmacc_vv_f32m1(C_reg_5_1, A_reg_1, B_reg_5,(4));
  C_reg_5_2 = __riscv_vfmacc_vv_f32m1(C_reg_5_2, A_reg_2, B_reg_5,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(3));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4], C_reg_5_1,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8], C_reg_5_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_regt_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12], C_regt_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 12], C_regt_5,(3));
}

// gemm_RVV_16x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 16] @DRAM
// )
void gemm_RVV_16x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
}

// gemm_RVV_16x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 16] @DRAM
// )
void gemm_RVV_16x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C.data[12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
}

// gemm_RVV_16x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 16] @DRAM
// )
void gemm_RVV_16x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
}

// gemm_RVV_16x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 16] @DRAM
// )
void gemm_RVV_16x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C.data[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
}

// gemm_RVV_16x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 16] @DRAM
// )
void gemm_RVV_16x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
}

// gemm_RVV_16x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 16] @DRAM
// )
void gemm_RVV_16x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C.data[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
}

// gemm_RVV_16x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 16] @DRAM
// )
void gemm_RVV_16x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_reg_3_3,(4));
}

// gemm_RVV_16x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 16] @DRAM
// )
void gemm_RVV_16x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C.data[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(4));
C_reg_3_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_reg_3_3,(4));
}

// gemm_RVV_16x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 16] @DRAM
// )
void gemm_RVV_16x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
vfloat32m1_t C_reg_4_3;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
  C_reg_4_3 = __riscv_vfmacc_vv_f32m1(C_reg_4_3, A_reg_3, B_reg_4,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12], C_reg_4_3,(4));
}

// gemm_RVV_16x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 16] @DRAM
// )
void gemm_RVV_16x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
vfloat32m1_t C_reg_4_3;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C.data[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(4));
C_reg_3_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_4_1 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4],(4));
C_reg_4_2 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8],(4));
C_reg_4_3 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
  C_reg_4_3 = __riscv_vfmacc_vv_f32m1(C_reg_4_3, A_reg_3, B_reg_4,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12], C_reg_4_3,(4));
}

// gemm_RVV_16x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 16] @DRAM
// )
void gemm_RVV_16x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
vfloat32m1_t C_reg_4_3;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_5_1;
vfloat32m1_t C_reg_5_2;
vfloat32m1_t C_reg_5_3;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
  C_reg_4_3 = __riscv_vfmacc_vv_f32m1(C_reg_4_3, A_reg_3, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_5_1 = __riscv_vfmacc_vv_f32m1(C_reg_5_1, A_reg_1, B_reg_5,(4));
  C_reg_5_2 = __riscv_vfmacc_vv_f32m1(C_reg_5_2, A_reg_2, B_reg_5,(4));
  C_reg_5_3 = __riscv_vfmacc_vv_f32m1(C_reg_5_3, A_reg_3, B_reg_5,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12], C_reg_4_3,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4], C_reg_5_1,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8], C_reg_5_2,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 12], C_reg_5_3,(4));
}

// gemm_RVV_16x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 16] @DRAM
// )
void gemm_RVV_16x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
vfloat32m1_t C_reg_4_3;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_5_1;
vfloat32m1_t C_reg_5_2;
vfloat32m1_t C_reg_5_3;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C.data[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(4));
C_reg_3_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_4_1 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4],(4));
C_reg_4_2 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8],(4));
C_reg_4_3 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_5_1 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4],(4));
C_reg_5_2 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8],(4));
C_reg_5_3 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
  C_reg_4_3 = __riscv_vfmacc_vv_f32m1(C_reg_4_3, A_reg_3, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_5_1 = __riscv_vfmacc_vv_f32m1(C_reg_5_1, A_reg_1, B_reg_5,(4));
  C_reg_5_2 = __riscv_vfmacc_vv_f32m1(C_reg_5_2, A_reg_2, B_reg_5,(4));
  C_reg_5_3 = __riscv_vfmacc_vv_f32m1(C_reg_5_3, A_reg_3, B_reg_5,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12], C_reg_4_3,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4], C_reg_5_1,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8], C_reg_5_2,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 12], C_reg_5_3,(4));
}

// gemm_RVV_17x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 17] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 17] @DRAM
// )
void gemm_RVV_17x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(1));
}

// gemm_RVV_17x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 17] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 17] @DRAM
// )
void gemm_RVV_17x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[16],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C.data[12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(1));
}

// gemm_RVV_17x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 17] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 17] @DRAM
// )
void gemm_RVV_17x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_regt_1,(1));
}

// gemm_RVV_17x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 17] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 17] @DRAM
// )
void gemm_RVV_17x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[16],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 16],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C.data[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_regt_1,(1));
}

// gemm_RVV_17x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 17] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 17] @DRAM
// )
void gemm_RVV_17x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16], C_regt_2,(1));
}

// gemm_RVV_17x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 17] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 17] @DRAM
// )
void gemm_RVV_17x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[16],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 16],(1));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C.data[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16], C_regt_2,(1));
}

// gemm_RVV_17x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 17] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 17] @DRAM
// )
void gemm_RVV_17x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16], C_regt_3,(1));
}

// gemm_RVV_17x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 17] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 17] @DRAM
// )
void gemm_RVV_17x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[16],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 16],(1));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16],(1));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C.data[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(4));
C_reg_3_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16], C_regt_3,(1));
}

// gemm_RVV_17x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 17] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 17] @DRAM
// )
void gemm_RVV_17x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
vfloat32m1_t C_reg_4_3;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
  C_reg_4_3 = __riscv_vfmacc_vv_f32m1(C_reg_4_3, A_reg_3, B_reg_4,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12], C_reg_4_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16], C_regt_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 16], C_regt_4,(1));
}

// gemm_RVV_17x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 17] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 17] @DRAM
// )
void gemm_RVV_17x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
vfloat32m1_t C_reg_4_3;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[16],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 16],(1));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16],(1));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16],(1));
C_regt_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 16],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C.data[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(4));
C_reg_3_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_4_1 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4],(4));
C_reg_4_2 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8],(4));
C_reg_4_3 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
  C_reg_4_3 = __riscv_vfmacc_vv_f32m1(C_reg_4_3, A_reg_3, B_reg_4,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12], C_reg_4_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16], C_regt_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 16], C_regt_4,(1));
}

// gemm_RVV_17x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 17] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 17] @DRAM
// )
void gemm_RVV_17x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
vfloat32m1_t C_reg_4_3;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_5_1;
vfloat32m1_t C_reg_5_2;
vfloat32m1_t C_reg_5_3;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_5 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
  C_reg_4_3 = __riscv_vfmacc_vv_f32m1(C_reg_4_3, A_reg_3, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_5_1 = __riscv_vfmacc_vv_f32m1(C_reg_5_1, A_reg_1, B_reg_5,(4));
  C_reg_5_2 = __riscv_vfmacc_vv_f32m1(C_reg_5_2, A_reg_2, B_reg_5,(4));
  C_reg_5_3 = __riscv_vfmacc_vv_f32m1(C_reg_5_3, A_reg_3, B_reg_5,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(1));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12], C_reg_4_3,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4], C_reg_5_1,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8], C_reg_5_2,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 12], C_reg_5_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16], C_regt_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 16], C_regt_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 16], C_regt_5,(1));
}

// gemm_RVV_17x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 17] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 17] @DRAM
// )
void gemm_RVV_17x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
vfloat32m1_t C_reg_4_3;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_5_1;
vfloat32m1_t C_reg_5_2;
vfloat32m1_t C_reg_5_3;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[16],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 16],(1));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16],(1));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16],(1));
C_regt_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 16],(1));
C_regt_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 16],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C.data[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(4));
C_reg_3_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_4_1 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4],(4));
C_reg_4_2 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8],(4));
C_reg_4_3 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_5_1 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4],(4));
C_reg_5_2 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8],(4));
C_reg_5_3 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
  C_reg_4_3 = __riscv_vfmacc_vv_f32m1(C_reg_4_3, A_reg_3, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_5_1 = __riscv_vfmacc_vv_f32m1(C_reg_5_1, A_reg_1, B_reg_5,(4));
  C_reg_5_2 = __riscv_vfmacc_vv_f32m1(C_reg_5_2, A_reg_2, B_reg_5,(4));
  C_reg_5_3 = __riscv_vfmacc_vv_f32m1(C_reg_5_3, A_reg_3, B_reg_5,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(1));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12], C_reg_4_3,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4], C_reg_5_1,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8], C_reg_5_2,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 12], C_reg_5_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16], C_regt_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 16], C_regt_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 16], C_regt_5,(1));
}

// gemm_RVV_18x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 18] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 18] @DRAM
// )
void gemm_RVV_18x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(2));
}

// gemm_RVV_18x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 18] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 18] @DRAM
// )
void gemm_RVV_18x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[16],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C.data[12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(2));
}

// gemm_RVV_18x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 18] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 18] @DRAM
// )
void gemm_RVV_18x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_regt_1,(2));
}

// gemm_RVV_18x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 18] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 18] @DRAM
// )
void gemm_RVV_18x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[16],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 16],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C.data[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_regt_1,(2));
}

// gemm_RVV_18x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 18] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 18] @DRAM
// )
void gemm_RVV_18x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16], C_regt_2,(2));
}

// gemm_RVV_18x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 18] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 18] @DRAM
// )
void gemm_RVV_18x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[16],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 16],(2));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C.data[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16], C_regt_2,(2));
}

// gemm_RVV_18x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 18] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 18] @DRAM
// )
void gemm_RVV_18x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16], C_regt_3,(2));
}

// gemm_RVV_18x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 18] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 18] @DRAM
// )
void gemm_RVV_18x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[16],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 16],(2));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16],(2));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C.data[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(4));
C_reg_3_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16], C_regt_3,(2));
}

// gemm_RVV_18x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 18] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 18] @DRAM
// )
void gemm_RVV_18x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
vfloat32m1_t C_reg_4_3;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
  C_reg_4_3 = __riscv_vfmacc_vv_f32m1(C_reg_4_3, A_reg_3, B_reg_4,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12], C_reg_4_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16], C_regt_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 16], C_regt_4,(2));
}

// gemm_RVV_18x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 18] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 18] @DRAM
// )
void gemm_RVV_18x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
vfloat32m1_t C_reg_4_3;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[16],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 16],(2));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16],(2));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16],(2));
C_regt_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 16],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C.data[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(4));
C_reg_3_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_4_1 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4],(4));
C_reg_4_2 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8],(4));
C_reg_4_3 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
  C_reg_4_3 = __riscv_vfmacc_vv_f32m1(C_reg_4_3, A_reg_3, B_reg_4,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12], C_reg_4_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16], C_regt_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 16], C_regt_4,(2));
}

// gemm_RVV_18x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 18] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 18] @DRAM
// )
void gemm_RVV_18x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
vfloat32m1_t C_reg_4_3;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_5_1;
vfloat32m1_t C_reg_5_2;
vfloat32m1_t C_reg_5_3;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_5 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
  C_reg_4_3 = __riscv_vfmacc_vv_f32m1(C_reg_4_3, A_reg_3, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_5_1 = __riscv_vfmacc_vv_f32m1(C_reg_5_1, A_reg_1, B_reg_5,(4));
  C_reg_5_2 = __riscv_vfmacc_vv_f32m1(C_reg_5_2, A_reg_2, B_reg_5,(4));
  C_reg_5_3 = __riscv_vfmacc_vv_f32m1(C_reg_5_3, A_reg_3, B_reg_5,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(2));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12], C_reg_4_3,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4], C_reg_5_1,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8], C_reg_5_2,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 12], C_reg_5_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16], C_regt_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 16], C_regt_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 16], C_regt_5,(2));
}

// gemm_RVV_18x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 18] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 18] @DRAM
// )
void gemm_RVV_18x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
vfloat32m1_t C_reg_4_3;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_5_1;
vfloat32m1_t C_reg_5_2;
vfloat32m1_t C_reg_5_3;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[16],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 16],(2));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16],(2));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16],(2));
C_regt_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 16],(2));
C_regt_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 16],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C.data[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(4));
C_reg_3_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_4_1 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4],(4));
C_reg_4_2 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8],(4));
C_reg_4_3 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_5_1 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4],(4));
C_reg_5_2 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8],(4));
C_reg_5_3 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
  C_reg_4_3 = __riscv_vfmacc_vv_f32m1(C_reg_4_3, A_reg_3, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_5_1 = __riscv_vfmacc_vv_f32m1(C_reg_5_1, A_reg_1, B_reg_5,(4));
  C_reg_5_2 = __riscv_vfmacc_vv_f32m1(C_reg_5_2, A_reg_2, B_reg_5,(4));
  C_reg_5_3 = __riscv_vfmacc_vv_f32m1(C_reg_5_3, A_reg_3, B_reg_5,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(2));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12], C_reg_4_3,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4], C_reg_5_1,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8], C_reg_5_2,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 12], C_reg_5_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16], C_regt_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 16], C_regt_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 16], C_regt_5,(2));
}

// gemm_RVV_19x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 19] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 19] @DRAM
// )
void gemm_RVV_19x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(3));
}

// gemm_RVV_19x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 19] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 19] @DRAM
// )
void gemm_RVV_19x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[16],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C.data[12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(3));
}

// gemm_RVV_19x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 19] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 19] @DRAM
// )
void gemm_RVV_19x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_regt_1,(3));
}

// gemm_RVV_19x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 19] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 19] @DRAM
// )
void gemm_RVV_19x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[16],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 16],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C.data[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_regt_1,(3));
}

// gemm_RVV_19x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 19] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 19] @DRAM
// )
void gemm_RVV_19x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16], C_regt_2,(3));
}

// gemm_RVV_19x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 19] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 19] @DRAM
// )
void gemm_RVV_19x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[16],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 16],(3));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C.data[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16], C_regt_2,(3));
}

// gemm_RVV_19x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 19] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 19] @DRAM
// )
void gemm_RVV_19x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16], C_regt_3,(3));
}

// gemm_RVV_19x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 19] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 19] @DRAM
// )
void gemm_RVV_19x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[16],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 16],(3));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16],(3));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C.data[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(4));
C_reg_3_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16], C_regt_3,(3));
}

// gemm_RVV_19x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 19] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 19] @DRAM
// )
void gemm_RVV_19x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
vfloat32m1_t C_reg_4_3;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
  C_reg_4_3 = __riscv_vfmacc_vv_f32m1(C_reg_4_3, A_reg_3, B_reg_4,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12], C_reg_4_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16], C_regt_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 16], C_regt_4,(3));
}

// gemm_RVV_19x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 19] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 19] @DRAM
// )
void gemm_RVV_19x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
vfloat32m1_t C_reg_4_3;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[16],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 16],(3));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16],(3));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16],(3));
C_regt_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 16],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C.data[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(4));
C_reg_3_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_4_1 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4],(4));
C_reg_4_2 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8],(4));
C_reg_4_3 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
  C_reg_4_3 = __riscv_vfmacc_vv_f32m1(C_reg_4_3, A_reg_3, B_reg_4,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12], C_reg_4_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16], C_regt_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 16], C_regt_4,(3));
}

// gemm_RVV_19x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 19] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 19] @DRAM
// )
void gemm_RVV_19x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
vfloat32m1_t C_reg_4_3;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_5_1;
vfloat32m1_t C_reg_5_2;
vfloat32m1_t C_reg_5_3;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_5 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
  C_reg_4_3 = __riscv_vfmacc_vv_f32m1(C_reg_4_3, A_reg_3, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_5_1 = __riscv_vfmacc_vv_f32m1(C_reg_5_1, A_reg_1, B_reg_5,(4));
  C_reg_5_2 = __riscv_vfmacc_vv_f32m1(C_reg_5_2, A_reg_2, B_reg_5,(4));
  C_reg_5_3 = __riscv_vfmacc_vv_f32m1(C_reg_5_3, A_reg_3, B_reg_5,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(3));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12], C_reg_4_3,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4], C_reg_5_1,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8], C_reg_5_2,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 12], C_reg_5_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16], C_regt_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 16], C_regt_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 16], C_regt_5,(3));
}

// gemm_RVV_19x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 19] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 19] @DRAM
// )
void gemm_RVV_19x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
vfloat32m1_t C_reg_4_3;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_5_1;
vfloat32m1_t C_reg_5_2;
vfloat32m1_t C_reg_5_3;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[16],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 16],(3));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16],(3));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16],(3));
C_regt_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 16],(3));
C_regt_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 16],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C.data[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(4));
C_reg_3_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_4_1 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4],(4));
C_reg_4_2 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8],(4));
C_reg_4_3 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_5_1 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4],(4));
C_reg_5_2 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8],(4));
C_reg_5_3 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
  C_reg_4_3 = __riscv_vfmacc_vv_f32m1(C_reg_4_3, A_reg_3, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_5_1 = __riscv_vfmacc_vv_f32m1(C_reg_5_1, A_reg_1, B_reg_5,(4));
  C_reg_5_2 = __riscv_vfmacc_vv_f32m1(C_reg_5_2, A_reg_2, B_reg_5,(4));
  C_reg_5_3 = __riscv_vfmacc_vv_f32m1(C_reg_5_3, A_reg_3, B_reg_5,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(3));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12], C_reg_4_3,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4], C_reg_5_1,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8], C_reg_5_2,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 12], C_reg_5_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16], C_regt_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 16], C_regt_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 16], C_regt_5,(3));
}

// gemm_RVV_1x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 1] @DRAM
// )
void gemm_RVV_1x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
}

// gemm_RVV_1x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 1] @DRAM
// )
void gemm_RVV_1x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
}

// gemm_RVV_1x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 1] @DRAM
// )
void gemm_RVV_1x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
}

// gemm_RVV_1x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 1] @DRAM
// )
void gemm_RVV_1x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
}

// gemm_RVV_1x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 1] @DRAM
// )
void gemm_RVV_1x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
}

// gemm_RVV_1x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 1] @DRAM
// )
void gemm_RVV_1x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
}

// gemm_RVV_1x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 1] @DRAM
// )
void gemm_RVV_1x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
}

// gemm_RVV_1x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 1] @DRAM
// )
void gemm_RVV_1x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
}

// gemm_RVV_1x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 1] @DRAM
// )
void gemm_RVV_1x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
}

// gemm_RVV_1x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 1] @DRAM
// )
void gemm_RVV_1x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
}

// gemm_RVV_1x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 1] @DRAM
// )
void gemm_RVV_1x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
}

// gemm_RVV_1x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 1] @DRAM
// )
void gemm_RVV_1x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
}

// gemm_RVV_20x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 20] @DRAM
// )
void gemm_RVV_20x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_reg_0_4,(4));
}

// gemm_RVV_20x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 20] @DRAM
// )
void gemm_RVV_20x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C.data[12],(4));
C_reg_0_4 = __riscv_vle32_v_f32m1(&C.data[16],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_reg_0_4,(4));
}

// gemm_RVV_20x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 20] @DRAM
// )
void gemm_RVV_20x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_reg_1_4,(4));
}

// gemm_RVV_20x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 20] @DRAM
// )
void gemm_RVV_20x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C.data[12],(4));
C_reg_0_4 = __riscv_vle32_v_f32m1(&C.data[16],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(4));
C_reg_1_4 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 16],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_reg_1_4,(4));
}

// gemm_RVV_20x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 20] @DRAM
// )
void gemm_RVV_20x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_2_4;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_2_4 = __riscv_vfmacc_vv_f32m1(C_reg_2_4, A_reg_4, B_reg_2,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16], C_reg_2_4,(4));
}

// gemm_RVV_20x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 20] @DRAM
// )
void gemm_RVV_20x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_2_4;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C.data[12],(4));
C_reg_0_4 = __riscv_vle32_v_f32m1(&C.data[16],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(4));
C_reg_1_4 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 16],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12],(4));
C_reg_2_4 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_2_4 = __riscv_vfmacc_vv_f32m1(C_reg_2_4, A_reg_4, B_reg_2,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16], C_reg_2_4,(4));
}

// gemm_RVV_20x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 20] @DRAM
// )
void gemm_RVV_20x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_2_4;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
vfloat32m1_t C_reg_3_4;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_2_4 = __riscv_vfmacc_vv_f32m1(C_reg_2_4, A_reg_4, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_reg_3_4 = __riscv_vfmacc_vv_f32m1(C_reg_3_4, A_reg_4, B_reg_3,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16], C_reg_2_4,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16], C_reg_3_4,(4));
}

// gemm_RVV_20x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 20] @DRAM
// )
void gemm_RVV_20x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_2_4;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
vfloat32m1_t C_reg_3_4;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C.data[12],(4));
C_reg_0_4 = __riscv_vle32_v_f32m1(&C.data[16],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(4));
C_reg_1_4 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 16],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12],(4));
C_reg_2_4 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(4));
C_reg_3_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12],(4));
C_reg_3_4 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_2_4 = __riscv_vfmacc_vv_f32m1(C_reg_2_4, A_reg_4, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_reg_3_4 = __riscv_vfmacc_vv_f32m1(C_reg_3_4, A_reg_4, B_reg_3,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16], C_reg_2_4,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16], C_reg_3_4,(4));
}

// gemm_RVV_20x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 20] @DRAM
// )
void gemm_RVV_20x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_2_4;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
vfloat32m1_t C_reg_3_4;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
vfloat32m1_t C_reg_4_3;
vfloat32m1_t C_reg_4_4;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_2_4 = __riscv_vfmacc_vv_f32m1(C_reg_2_4, A_reg_4, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_reg_3_4 = __riscv_vfmacc_vv_f32m1(C_reg_3_4, A_reg_4, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
  C_reg_4_3 = __riscv_vfmacc_vv_f32m1(C_reg_4_3, A_reg_3, B_reg_4,(4));
  C_reg_4_4 = __riscv_vfmacc_vv_f32m1(C_reg_4_4, A_reg_4, B_reg_4,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16], C_reg_2_4,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16], C_reg_3_4,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12], C_reg_4_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 16], C_reg_4_4,(4));
}

// gemm_RVV_20x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 20] @DRAM
// )
void gemm_RVV_20x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_2_4;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
vfloat32m1_t C_reg_3_4;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
vfloat32m1_t C_reg_4_3;
vfloat32m1_t C_reg_4_4;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C.data[12],(4));
C_reg_0_4 = __riscv_vle32_v_f32m1(&C.data[16],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(4));
C_reg_1_4 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 16],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12],(4));
C_reg_2_4 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(4));
C_reg_3_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12],(4));
C_reg_3_4 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_4_1 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4],(4));
C_reg_4_2 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8],(4));
C_reg_4_3 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12],(4));
C_reg_4_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 16],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_2_4 = __riscv_vfmacc_vv_f32m1(C_reg_2_4, A_reg_4, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_reg_3_4 = __riscv_vfmacc_vv_f32m1(C_reg_3_4, A_reg_4, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
  C_reg_4_3 = __riscv_vfmacc_vv_f32m1(C_reg_4_3, A_reg_3, B_reg_4,(4));
  C_reg_4_4 = __riscv_vfmacc_vv_f32m1(C_reg_4_4, A_reg_4, B_reg_4,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16], C_reg_2_4,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16], C_reg_3_4,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12], C_reg_4_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 16], C_reg_4_4,(4));
}

// gemm_RVV_20x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 20] @DRAM
// )
void gemm_RVV_20x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_2_4;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
vfloat32m1_t C_reg_3_4;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
vfloat32m1_t C_reg_4_3;
vfloat32m1_t C_reg_4_4;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_5_1;
vfloat32m1_t C_reg_5_2;
vfloat32m1_t C_reg_5_3;
vfloat32m1_t C_reg_5_4;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_2_4 = __riscv_vfmacc_vv_f32m1(C_reg_2_4, A_reg_4, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_reg_3_4 = __riscv_vfmacc_vv_f32m1(C_reg_3_4, A_reg_4, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
  C_reg_4_3 = __riscv_vfmacc_vv_f32m1(C_reg_4_3, A_reg_3, B_reg_4,(4));
  C_reg_4_4 = __riscv_vfmacc_vv_f32m1(C_reg_4_4, A_reg_4, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_5_1 = __riscv_vfmacc_vv_f32m1(C_reg_5_1, A_reg_1, B_reg_5,(4));
  C_reg_5_2 = __riscv_vfmacc_vv_f32m1(C_reg_5_2, A_reg_2, B_reg_5,(4));
  C_reg_5_3 = __riscv_vfmacc_vv_f32m1(C_reg_5_3, A_reg_3, B_reg_5,(4));
  C_reg_5_4 = __riscv_vfmacc_vv_f32m1(C_reg_5_4, A_reg_4, B_reg_5,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16], C_reg_2_4,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16], C_reg_3_4,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12], C_reg_4_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 16], C_reg_4_4,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4], C_reg_5_1,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8], C_reg_5_2,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 12], C_reg_5_3,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 16], C_reg_5_4,(4));
}

// gemm_RVV_20x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 20] @DRAM
// )
void gemm_RVV_20x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_2_4;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
vfloat32m1_t C_reg_3_4;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_4_2;
vfloat32m1_t C_reg_4_3;
vfloat32m1_t C_reg_4_4;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_5_1;
vfloat32m1_t C_reg_5_2;
vfloat32m1_t C_reg_5_3;
vfloat32m1_t C_reg_5_4;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C.data[12],(4));
C_reg_0_4 = __riscv_vle32_v_f32m1(&C.data[16],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 12],(4));
C_reg_1_4 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 16],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12],(4));
C_reg_2_4 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(4));
C_reg_3_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12],(4));
C_reg_3_4 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_4_1 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4],(4));
C_reg_4_2 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8],(4));
C_reg_4_3 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12],(4));
C_reg_4_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 16],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_5_1 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4],(4));
C_reg_5_2 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8],(4));
C_reg_5_3 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 12],(4));
C_reg_5_4 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 16],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 16],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_2_4 = __riscv_vfmacc_vv_f32m1(C_reg_2_4, A_reg_4, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_reg_3_4 = __riscv_vfmacc_vv_f32m1(C_reg_3_4, A_reg_4, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_4_2 = __riscv_vfmacc_vv_f32m1(C_reg_4_2, A_reg_2, B_reg_4,(4));
  C_reg_4_3 = __riscv_vfmacc_vv_f32m1(C_reg_4_3, A_reg_3, B_reg_4,(4));
  C_reg_4_4 = __riscv_vfmacc_vv_f32m1(C_reg_4_4, A_reg_4, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_5_1 = __riscv_vfmacc_vv_f32m1(C_reg_5_1, A_reg_1, B_reg_5,(4));
  C_reg_5_2 = __riscv_vfmacc_vv_f32m1(C_reg_5_2, A_reg_2, B_reg_5,(4));
  C_reg_5_3 = __riscv_vfmacc_vv_f32m1(C_reg_5_3, A_reg_3, B_reg_5,(4));
  C_reg_5_4 = __riscv_vfmacc_vv_f32m1(C_reg_5_4, A_reg_4, B_reg_5,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C.data[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C.data[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 16], C_reg_2_4,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 16], C_reg_3_4,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_reg_4_2,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 12], C_reg_4_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 16], C_reg_4_4,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4], C_reg_5_1,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8], C_reg_5_2,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 12], C_reg_5_3,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 16], C_reg_5_4,(4));
}

// gemm_RVV_2x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 2] @DRAM
// )
void gemm_RVV_2x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
}

// gemm_RVV_2x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 2] @DRAM
// )
void gemm_RVV_2x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
}

// gemm_RVV_2x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 2] @DRAM
// )
void gemm_RVV_2x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
}

// gemm_RVV_2x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 2] @DRAM
// )
void gemm_RVV_2x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
}

// gemm_RVV_2x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 2] @DRAM
// )
void gemm_RVV_2x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
}

// gemm_RVV_2x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 2] @DRAM
// )
void gemm_RVV_2x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
}

// gemm_RVV_2x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 2] @DRAM
// )
void gemm_RVV_2x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
}

// gemm_RVV_2x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 2] @DRAM
// )
void gemm_RVV_2x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
}

// gemm_RVV_2x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 2] @DRAM
// )
void gemm_RVV_2x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
}

// gemm_RVV_2x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 2] @DRAM
// )
void gemm_RVV_2x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
}

// gemm_RVV_2x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 2] @DRAM
// )
void gemm_RVV_2x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
}

// gemm_RVV_2x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 2] @DRAM
// )
void gemm_RVV_2x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
}

// gemm_RVV_3x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 3] @DRAM
// )
void gemm_RVV_3x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
}

// gemm_RVV_3x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 3] @DRAM
// )
void gemm_RVV_3x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
}

// gemm_RVV_3x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 3] @DRAM
// )
void gemm_RVV_3x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
}

// gemm_RVV_3x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 3] @DRAM
// )
void gemm_RVV_3x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
}

// gemm_RVV_3x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 3] @DRAM
// )
void gemm_RVV_3x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
}

// gemm_RVV_3x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 3] @DRAM
// )
void gemm_RVV_3x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
}

// gemm_RVV_3x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 3] @DRAM
// )
void gemm_RVV_3x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
}

// gemm_RVV_3x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 3] @DRAM
// )
void gemm_RVV_3x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
}

// gemm_RVV_3x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 3] @DRAM
// )
void gemm_RVV_3x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
}

// gemm_RVV_3x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 3] @DRAM
// )
void gemm_RVV_3x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
}

// gemm_RVV_3x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 3] @DRAM
// )
void gemm_RVV_3x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
}

// gemm_RVV_3x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 3] @DRAM
// )
void gemm_RVV_3x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
}

// gemm_RVV_4x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 4] @DRAM
// )
void gemm_RVV_4x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
}

// gemm_RVV_4x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 4] @DRAM
// )
void gemm_RVV_4x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
}

// gemm_RVV_4x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 4] @DRAM
// )
void gemm_RVV_4x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
}

// gemm_RVV_4x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 4] @DRAM
// )
void gemm_RVV_4x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
}

// gemm_RVV_4x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 4] @DRAM
// )
void gemm_RVV_4x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
}

// gemm_RVV_4x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 4] @DRAM
// )
void gemm_RVV_4x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
}

// gemm_RVV_4x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 4] @DRAM
// )
void gemm_RVV_4x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
}

// gemm_RVV_4x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 4] @DRAM
// )
void gemm_RVV_4x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
}

// gemm_RVV_4x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 4] @DRAM
// )
void gemm_RVV_4x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
}

// gemm_RVV_4x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 4] @DRAM
// )
void gemm_RVV_4x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
}

// gemm_RVV_4x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 4] @DRAM
// )
void gemm_RVV_4x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
}

// gemm_RVV_4x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 4] @DRAM
// )
void gemm_RVV_4x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
}

// gemm_RVV_5x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 5] @DRAM
// )
void gemm_RVV_5x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(1));
}

// gemm_RVV_5x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 5] @DRAM
// )
void gemm_RVV_5x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[4],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(1));
}

// gemm_RVV_5x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 5] @DRAM
// )
void gemm_RVV_5x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_regt_1,(1));
}

// gemm_RVV_5x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 5] @DRAM
// )
void gemm_RVV_5x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[4],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_regt_1,(1));
}

// gemm_RVV_5x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 5] @DRAM
// )
void gemm_RVV_5x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_regt_2,(1));
}

// gemm_RVV_5x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 5] @DRAM
// )
void gemm_RVV_5x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[4],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(1));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_regt_2,(1));
}

// gemm_RVV_5x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 5] @DRAM
// )
void gemm_RVV_5x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_regt_3,(1));
}

// gemm_RVV_5x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 5] @DRAM
// )
void gemm_RVV_5x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[4],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(1));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(1));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_regt_3,(1));
}

// gemm_RVV_5x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 5] @DRAM
// )
void gemm_RVV_5x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_regt_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_regt_4,(1));
}

// gemm_RVV_5x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 5] @DRAM
// )
void gemm_RVV_5x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[4],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(1));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(1));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(1));
C_regt_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_regt_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_regt_4,(1));
}

// gemm_RVV_5x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 5] @DRAM
// )
void gemm_RVV_5x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_5_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_5 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(1));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_regt_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_regt_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4], C_regt_5,(1));
}

// gemm_RVV_5x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 5] @DRAM
// )
void gemm_RVV_5x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_5_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[4],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(1));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(1));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(1));
C_regt_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4],(1));
C_regt_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(1));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_regt_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_regt_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4], C_regt_5,(1));
}

// gemm_RVV_6x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 6] @DRAM
// )
void gemm_RVV_6x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(2));
}

// gemm_RVV_6x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 6] @DRAM
// )
void gemm_RVV_6x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[4],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(2));
}

// gemm_RVV_6x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 6] @DRAM
// )
void gemm_RVV_6x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_regt_1,(2));
}

// gemm_RVV_6x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 6] @DRAM
// )
void gemm_RVV_6x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[4],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_regt_1,(2));
}

// gemm_RVV_6x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 6] @DRAM
// )
void gemm_RVV_6x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_regt_2,(2));
}

// gemm_RVV_6x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 6] @DRAM
// )
void gemm_RVV_6x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[4],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(2));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_regt_2,(2));
}

// gemm_RVV_6x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 6] @DRAM
// )
void gemm_RVV_6x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_regt_3,(2));
}

// gemm_RVV_6x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 6] @DRAM
// )
void gemm_RVV_6x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[4],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(2));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(2));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_regt_3,(2));
}

// gemm_RVV_6x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 6] @DRAM
// )
void gemm_RVV_6x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_regt_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_regt_4,(2));
}

// gemm_RVV_6x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 6] @DRAM
// )
void gemm_RVV_6x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[4],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(2));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(2));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(2));
C_regt_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_regt_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_regt_4,(2));
}

// gemm_RVV_6x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 6] @DRAM
// )
void gemm_RVV_6x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_5_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_5 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(2));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_regt_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_regt_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4], C_regt_5,(2));
}

// gemm_RVV_6x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 6] @DRAM
// )
void gemm_RVV_6x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_5_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[4],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(2));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(2));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(2));
C_regt_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4],(2));
C_regt_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(2));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_regt_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_regt_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4], C_regt_5,(2));
}

// gemm_RVV_7x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 7] @DRAM
// )
void gemm_RVV_7x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(3));
}

// gemm_RVV_7x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 7] @DRAM
// )
void gemm_RVV_7x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[4],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(3));
}

// gemm_RVV_7x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 7] @DRAM
// )
void gemm_RVV_7x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_regt_1,(3));
}

// gemm_RVV_7x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 7] @DRAM
// )
void gemm_RVV_7x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[4],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_regt_1,(3));
}

// gemm_RVV_7x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 7] @DRAM
// )
void gemm_RVV_7x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_regt_2,(3));
}

// gemm_RVV_7x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 7] @DRAM
// )
void gemm_RVV_7x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[4],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(3));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_regt_2,(3));
}

// gemm_RVV_7x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 7] @DRAM
// )
void gemm_RVV_7x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_regt_3,(3));
}

// gemm_RVV_7x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 7] @DRAM
// )
void gemm_RVV_7x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[4],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(3));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(3));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_regt_3,(3));
}

// gemm_RVV_7x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 7] @DRAM
// )
void gemm_RVV_7x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_regt_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_regt_4,(3));
}

// gemm_RVV_7x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 7] @DRAM
// )
void gemm_RVV_7x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[4],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(3));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(3));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(3));
C_regt_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_regt_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_regt_4,(3));
}

// gemm_RVV_7x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 7] @DRAM
// )
void gemm_RVV_7x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_5_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_5 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(3));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_regt_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_regt_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4], C_regt_5,(3));
}

// gemm_RVV_7x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 7] @DRAM
// )
void gemm_RVV_7x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_5_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[4],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(3));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(3));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(3));
C_regt_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4],(3));
C_regt_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(3));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_regt_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_regt_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4], C_regt_5,(3));
}

// gemm_RVV_8x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 8] @DRAM
// )
void gemm_RVV_8x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
}

// gemm_RVV_8x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 8] @DRAM
// )
void gemm_RVV_8x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
}

// gemm_RVV_8x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 8] @DRAM
// )
void gemm_RVV_8x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
}

// gemm_RVV_8x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 8] @DRAM
// )
void gemm_RVV_8x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
}

// gemm_RVV_8x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 8] @DRAM
// )
void gemm_RVV_8x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
}

// gemm_RVV_8x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 8] @DRAM
// )
void gemm_RVV_8x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
}

// gemm_RVV_8x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 8] @DRAM
// )
void gemm_RVV_8x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
}

// gemm_RVV_8x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 8] @DRAM
// )
void gemm_RVV_8x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
}

// gemm_RVV_8x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 8] @DRAM
// )
void gemm_RVV_8x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
}

// gemm_RVV_8x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 8] @DRAM
// )
void gemm_RVV_8x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_4_1 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
}

// gemm_RVV_8x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 8] @DRAM
// )
void gemm_RVV_8x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_5_1;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_5_1 = __riscv_vfmacc_vv_f32m1(C_reg_5_1, A_reg_1, B_reg_5,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4], C_reg_5_1,(4));
}

// gemm_RVV_8x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 8] @DRAM
// )
void gemm_RVV_8x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_5_1;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_4_1 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_5_1 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_5_1 = __riscv_vfmacc_vv_f32m1(C_reg_5_1, A_reg_1, B_reg_5,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4], C_reg_5_1,(4));
}

// gemm_RVV_9x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 9] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 9] @DRAM
// )
void gemm_RVV_9x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(1));
}

// gemm_RVV_9x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 9] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 9] @DRAM
// )
void gemm_RVV_9x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[8],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(1));
}

// gemm_RVV_9x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 9] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 9] @DRAM
// )
void gemm_RVV_9x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(1));
}

// gemm_RVV_9x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 9] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 9] @DRAM
// )
void gemm_RVV_9x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[8],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(1));
}

// gemm_RVV_9x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 9] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 9] @DRAM
// )
void gemm_RVV_9x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_regt_2,(1));
}

// gemm_RVV_9x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 9] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 9] @DRAM
// )
void gemm_RVV_9x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[8],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(1));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_regt_2,(1));
}

// gemm_RVV_9x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 9] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 9] @DRAM
// )
void gemm_RVV_9x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_regt_3,(1));
}

// gemm_RVV_9x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 9] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 9] @DRAM
// )
void gemm_RVV_9x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[8],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(1));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(1));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_regt_3,(1));
}

// gemm_RVV_9x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 9] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 9] @DRAM
// )
void gemm_RVV_9x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_regt_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_regt_4,(1));
}

// gemm_RVV_9x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 9] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 9] @DRAM
// )
void gemm_RVV_9x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[8],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(1));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(1));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(1));
C_regt_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_4_1 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_regt_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_regt_4,(1));
}

// gemm_RVV_9x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 9] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 9] @DRAM
// )
void gemm_RVV_9x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_5_1;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_5 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_5_1 = __riscv_vfmacc_vv_f32m1(C_reg_5_1, A_reg_1, B_reg_5,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(1));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4], C_reg_5_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_regt_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_regt_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8], C_regt_5,(1));
}

// gemm_RVV_9x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 9] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 9] @DRAM
// )
void gemm_RVV_9x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_4_1;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_5_1;
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[8],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(1));
C_regt_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8],(1));
C_regt_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8],(1));
C_regt_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8],(1));
C_regt_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 4],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_4_1 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_5_1 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_4_1 = __riscv_vfmacc_vv_f32m1(C_reg_4_1, A_reg_1, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_5_1 = __riscv_vfmacc_vv_f32m1(C_reg_5_1, A_reg_1, B_reg_5,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(1));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C.data[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 4], C_reg_5_1,(4));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0]) + 8], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0]) + 8], C_regt_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0]) + 8], C_regt_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0]) + 8], C_regt_5,(1));
}


/* relying on the following instruction..."
rvv_broadcast_4xf32(dst,src,vl)
{dst_data} = __riscv_vfmv_v_f_f32m1({src_data},{vl});
*/

/* relying on the following instruction..."
rvv_broadcast_4xf32_0(dst,vl)
{dst_data} = __riscv_vfmv_v_f_f32m1(0.0f,{vl});
*/

/* relying on the following instruction..."
rvv_vfmacc_4xf32_4xf32(dst,lhs,rhs,vl)
{dst_data} = __riscv_vfmacc_vv_f32m1({dst_data}, {lhs_data}, {rhs_data},{vl});
*/

/* relying on the following instruction..."
rvv_vld_4xf32(dst,src,vl)
{dst_data} = __riscv_vle32_v_f32m1(&{src_data},{vl});
*/

/* relying on the following instruction..."
rvv_vst_4xf32(dst,src,vl)
__riscv_vse32_v_f32m1(&{dst_data}, {src_data},{vl});
*/
