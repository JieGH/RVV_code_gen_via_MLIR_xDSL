#include "kernels_RVV_48x2_fp16.h"



#include <stdio.h>
#include <stdlib.h>

#include <riscv_vector.h>


// gemm_RVV_10x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 10] @DRAM
// )
void gemm_RVV_10x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_regt_0,(2));
}

// gemm_RVV_10x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 10] @DRAM
// )
void gemm_RVV_10x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[8],(2));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_regt_0,(2));
}

// gemm_RVV_10x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 10] @DRAM
// )
void gemm_RVV_10x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_regt_1,(2));
}

// gemm_RVV_10x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 10] @DRAM
// )
void gemm_RVV_10x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[8],(2));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(2));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_regt_1,(2));
}

// gemm_RVV_11x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 11] @DRAM
// )
void gemm_RVV_11x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_regt_0,(3));
}

// gemm_RVV_11x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 11] @DRAM
// )
void gemm_RVV_11x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[8],(3));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_regt_0,(3));
}

// gemm_RVV_11x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 11] @DRAM
// )
void gemm_RVV_11x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_regt_1,(3));
}

// gemm_RVV_11x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 11] @DRAM
// )
void gemm_RVV_11x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[8],(3));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(3));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_regt_1,(3));
}

// gemm_RVV_12x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 12] @DRAM
// )
void gemm_RVV_12x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_regt_0,(4));
}

// gemm_RVV_12x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 12] @DRAM
// )
void gemm_RVV_12x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[8],(4));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_regt_0,(4));
}

// gemm_RVV_12x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 12] @DRAM
// )
void gemm_RVV_12x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_regt_1,(4));
}

// gemm_RVV_12x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 12] @DRAM
// )
void gemm_RVV_12x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[8],(4));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(4));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_regt_1,(4));
}

// gemm_RVV_13x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 13] @DRAM
// )
void gemm_RVV_13x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_regt_0,(5));
}

// gemm_RVV_13x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 13] @DRAM
// )
void gemm_RVV_13x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[8],(5));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_regt_0,(5));
}

// gemm_RVV_13x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 13] @DRAM
// )
void gemm_RVV_13x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_regt_1,(5));
}

// gemm_RVV_13x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 13] @DRAM
// )
void gemm_RVV_13x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[8],(5));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(5));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_regt_1,(5));
}

// gemm_RVV_14x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 14] @DRAM
// )
void gemm_RVV_14x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_regt_0,(6));
}

// gemm_RVV_14x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 14] @DRAM
// )
void gemm_RVV_14x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[8],(6));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_regt_0,(6));
}

// gemm_RVV_14x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 14] @DRAM
// )
void gemm_RVV_14x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_regt_1,(6));
}

// gemm_RVV_14x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 14] @DRAM
// )
void gemm_RVV_14x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[8],(6));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(6));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_regt_1,(6));
}

// gemm_RVV_15x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 15] @DRAM
// )
void gemm_RVV_15x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_regt_0,(7));
}

// gemm_RVV_15x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 15] @DRAM
// )
void gemm_RVV_15x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[8],(7));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_regt_0,(7));
}

// gemm_RVV_15x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 15] @DRAM
// )
void gemm_RVV_15x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_regt_1,(7));
}

// gemm_RVV_15x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 15] @DRAM
// )
void gemm_RVV_15x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[8],(7));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(7));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_regt_1,(7));
}

// gemm_RVV_16x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 16] @DRAM
// )
void gemm_RVV_16x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
}

// gemm_RVV_16x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 16] @DRAM
// )
void gemm_RVV_16x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
}

// gemm_RVV_16x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 16] @DRAM
// )
void gemm_RVV_16x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
}

// gemm_RVV_16x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 16] @DRAM
// )
void gemm_RVV_16x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
}

// gemm_RVV_17x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 17] @DRAM
// )
void gemm_RVV_17x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(1));
}

// gemm_RVV_17x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 17] @DRAM
// )
void gemm_RVV_17x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(1));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(1));
}

// gemm_RVV_17x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 17] @DRAM
// )
void gemm_RVV_17x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(1));
}

// gemm_RVV_17x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 17] @DRAM
// )
void gemm_RVV_17x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(1));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(1));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(1));
}

// gemm_RVV_18x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 18] @DRAM
// )
void gemm_RVV_18x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(2));
}

// gemm_RVV_18x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 18] @DRAM
// )
void gemm_RVV_18x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(2));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(2));
}

// gemm_RVV_18x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 18] @DRAM
// )
void gemm_RVV_18x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(2));
}

// gemm_RVV_18x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 18] @DRAM
// )
void gemm_RVV_18x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(2));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(2));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(2));
}

// gemm_RVV_19x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 19] @DRAM
// )
void gemm_RVV_19x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(3));
}

// gemm_RVV_19x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 19] @DRAM
// )
void gemm_RVV_19x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(3));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(3));
}

// gemm_RVV_19x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 19] @DRAM
// )
void gemm_RVV_19x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(3));
}

// gemm_RVV_19x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 19] @DRAM
// )
void gemm_RVV_19x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(3));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(3));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(3));
}

// gemm_RVV_1x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 1] @DRAM
// )
void gemm_RVV_1x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
}

// gemm_RVV_1x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 1] @DRAM
// )
void gemm_RVV_1x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
}

// gemm_RVV_1x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 1] @DRAM
// )
void gemm_RVV_1x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
}

// gemm_RVV_1x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 1] @DRAM
// )
void gemm_RVV_1x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
}

// gemm_RVV_20x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 20] @DRAM
// )
void gemm_RVV_20x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(4));
}

// gemm_RVV_20x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 20] @DRAM
// )
void gemm_RVV_20x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(4));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(4));
}

// gemm_RVV_20x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 20] @DRAM
// )
void gemm_RVV_20x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(4));
}

// gemm_RVV_20x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 20] @DRAM
// )
void gemm_RVV_20x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(4));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(4));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(4));
}

// gemm_RVV_21x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 21] @DRAM
// )
void gemm_RVV_21x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(5));
}

// gemm_RVV_21x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 21] @DRAM
// )
void gemm_RVV_21x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(5));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(5));
}

// gemm_RVV_21x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 21] @DRAM
// )
void gemm_RVV_21x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(5));
}

// gemm_RVV_21x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 21] @DRAM
// )
void gemm_RVV_21x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(5));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(5));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(5));
}

// gemm_RVV_22x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 22] @DRAM
// )
void gemm_RVV_22x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(6));
}

// gemm_RVV_22x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 22] @DRAM
// )
void gemm_RVV_22x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(6));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(6));
}

// gemm_RVV_22x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 22] @DRAM
// )
void gemm_RVV_22x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(6));
}

// gemm_RVV_22x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 22] @DRAM
// )
void gemm_RVV_22x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(6));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(6));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(6));
}

// gemm_RVV_23x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 23] @DRAM
// )
void gemm_RVV_23x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(7));
}

// gemm_RVV_23x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 23] @DRAM
// )
void gemm_RVV_23x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(7));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(7));
}

// gemm_RVV_23x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 23] @DRAM
// )
void gemm_RVV_23x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(7));
}

// gemm_RVV_23x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 23] @DRAM
// )
void gemm_RVV_23x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[16],(7));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(7));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_regt_1,(7));
}

// gemm_RVV_24x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 24] @DRAM
// )
void gemm_RVV_24x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
}

// gemm_RVV_24x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 24] @DRAM
// )
void gemm_RVV_24x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
}

// gemm_RVV_24x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 24] @DRAM
// )
void gemm_RVV_24x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
}

// gemm_RVV_24x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 24] @DRAM
// )
void gemm_RVV_24x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
}

// gemm_RVV_25x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 25] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 25] @DRAM
// )
void gemm_RVV_25x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_regt_0,(1));
}

// gemm_RVV_25x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 25] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 25] @DRAM
// )
void gemm_RVV_25x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[24],(1));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_regt_0,(1));
}

// gemm_RVV_25x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 25] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 25] @DRAM
// )
void gemm_RVV_25x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_regt_1,(1));
}

// gemm_RVV_25x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 25] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 25] @DRAM
// )
void gemm_RVV_25x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[24],(1));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 24],(1));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_regt_1,(1));
}

// gemm_RVV_26x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 26] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 26] @DRAM
// )
void gemm_RVV_26x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_regt_0,(2));
}

// gemm_RVV_26x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 26] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 26] @DRAM
// )
void gemm_RVV_26x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[24],(2));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_regt_0,(2));
}

// gemm_RVV_26x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 26] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 26] @DRAM
// )
void gemm_RVV_26x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_regt_1,(2));
}

// gemm_RVV_26x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 26] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 26] @DRAM
// )
void gemm_RVV_26x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[24],(2));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 24],(2));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_regt_1,(2));
}

// gemm_RVV_27x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 27] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 27] @DRAM
// )
void gemm_RVV_27x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_regt_0,(3));
}

// gemm_RVV_27x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 27] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 27] @DRAM
// )
void gemm_RVV_27x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[24],(3));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_regt_0,(3));
}

// gemm_RVV_27x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 27] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 27] @DRAM
// )
void gemm_RVV_27x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_regt_1,(3));
}

// gemm_RVV_27x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 27] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 27] @DRAM
// )
void gemm_RVV_27x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[24],(3));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 24],(3));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_regt_1,(3));
}

// gemm_RVV_28x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 28] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 28] @DRAM
// )
void gemm_RVV_28x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_regt_0,(4));
}

// gemm_RVV_28x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 28] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 28] @DRAM
// )
void gemm_RVV_28x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[24],(4));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_regt_0,(4));
}

// gemm_RVV_28x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 28] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 28] @DRAM
// )
void gemm_RVV_28x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_regt_1,(4));
}

// gemm_RVV_28x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 28] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 28] @DRAM
// )
void gemm_RVV_28x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[24],(4));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 24],(4));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_regt_1,(4));
}

// gemm_RVV_29x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 29] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 29] @DRAM
// )
void gemm_RVV_29x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_regt_0,(5));
}

// gemm_RVV_29x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 29] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 29] @DRAM
// )
void gemm_RVV_29x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[24],(5));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_regt_0,(5));
}

// gemm_RVV_29x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 29] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 29] @DRAM
// )
void gemm_RVV_29x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_regt_1,(5));
}

// gemm_RVV_29x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 29] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 29] @DRAM
// )
void gemm_RVV_29x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[24],(5));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 24],(5));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_regt_1,(5));
}

// gemm_RVV_2x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 2] @DRAM
// )
void gemm_RVV_2x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
}

// gemm_RVV_2x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 2] @DRAM
// )
void gemm_RVV_2x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
}

// gemm_RVV_2x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 2] @DRAM
// )
void gemm_RVV_2x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
}

// gemm_RVV_2x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 2] @DRAM
// )
void gemm_RVV_2x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
}

// gemm_RVV_30x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 30] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 30] @DRAM
// )
void gemm_RVV_30x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_regt_0,(6));
}

// gemm_RVV_30x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 30] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 30] @DRAM
// )
void gemm_RVV_30x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[24],(6));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_regt_0,(6));
}

// gemm_RVV_30x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 30] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 30] @DRAM
// )
void gemm_RVV_30x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_regt_1,(6));
}

// gemm_RVV_30x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 30] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 30] @DRAM
// )
void gemm_RVV_30x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[24],(6));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 24],(6));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_regt_1,(6));
}

// gemm_RVV_31x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 31] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 31] @DRAM
// )
void gemm_RVV_31x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_regt_0,(7));
}

// gemm_RVV_31x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 31] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 31] @DRAM
// )
void gemm_RVV_31x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[24],(7));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_regt_0,(7));
}

// gemm_RVV_31x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 31] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 31] @DRAM
// )
void gemm_RVV_31x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_regt_1,(7));
}

// gemm_RVV_31x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 31] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 31] @DRAM
// )
void gemm_RVV_31x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[24],(7));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 24],(7));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_regt_1,(7));
}

// gemm_RVV_32x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 32] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 32] @DRAM
// )
void gemm_RVV_32x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
}

// gemm_RVV_32x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 32] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 32] @DRAM
// )
void gemm_RVV_32x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[24],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
}

// gemm_RVV_32x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 32] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 32] @DRAM
// )
void gemm_RVV_32x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_reg_1_3,(8));
}

// gemm_RVV_32x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 32] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 32] @DRAM
// )
void gemm_RVV_32x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[24],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(8));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 24],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_reg_1_3,(8));
}

// gemm_RVV_33x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 33] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 33] @DRAM
// )
void gemm_RVV_33x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(1));
}

// gemm_RVV_33x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 33] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 33] @DRAM
// )
void gemm_RVV_33x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(1));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[24],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(1));
}

// gemm_RVV_33x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 33] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 33] @DRAM
// )
void gemm_RVV_33x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(1));
}

// gemm_RVV_33x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 33] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 33] @DRAM
// )
void gemm_RVV_33x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(1));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(1));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[24],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(8));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 24],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(1));
}

// gemm_RVV_34x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 34] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 34] @DRAM
// )
void gemm_RVV_34x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(2));
}

// gemm_RVV_34x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 34] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 34] @DRAM
// )
void gemm_RVV_34x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(2));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[24],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(2));
}

// gemm_RVV_34x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 34] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 34] @DRAM
// )
void gemm_RVV_34x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(2));
}

// gemm_RVV_34x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 34] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 34] @DRAM
// )
void gemm_RVV_34x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(2));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(2));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[24],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(8));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 24],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(2));
}

// gemm_RVV_35x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 35] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 35] @DRAM
// )
void gemm_RVV_35x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(3));
}

// gemm_RVV_35x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 35] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 35] @DRAM
// )
void gemm_RVV_35x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(3));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[24],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(3));
}

// gemm_RVV_35x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 35] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 35] @DRAM
// )
void gemm_RVV_35x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(3));
}

// gemm_RVV_35x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 35] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 35] @DRAM
// )
void gemm_RVV_35x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(3));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(3));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[24],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(8));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 24],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(3));
}

// gemm_RVV_36x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 36] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 36] @DRAM
// )
void gemm_RVV_36x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(4));
}

// gemm_RVV_36x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 36] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 36] @DRAM
// )
void gemm_RVV_36x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(4));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[24],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(4));
}

// gemm_RVV_36x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 36] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 36] @DRAM
// )
void gemm_RVV_36x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(4));
}

// gemm_RVV_36x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 36] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 36] @DRAM
// )
void gemm_RVV_36x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(4));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(4));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[24],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(8));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 24],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(4));
}

// gemm_RVV_37x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 37] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 37] @DRAM
// )
void gemm_RVV_37x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(5));
}

// gemm_RVV_37x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 37] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 37] @DRAM
// )
void gemm_RVV_37x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(5));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[24],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(5));
}

// gemm_RVV_37x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 37] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 37] @DRAM
// )
void gemm_RVV_37x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(5));
}

// gemm_RVV_37x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 37] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 37] @DRAM
// )
void gemm_RVV_37x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(5));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(5));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[24],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(8));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 24],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(5));
}

// gemm_RVV_38x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 38] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 38] @DRAM
// )
void gemm_RVV_38x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(6));
}

// gemm_RVV_38x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 38] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 38] @DRAM
// )
void gemm_RVV_38x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(6));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[24],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(6));
}

// gemm_RVV_38x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 38] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 38] @DRAM
// )
void gemm_RVV_38x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(6));
}

// gemm_RVV_38x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 38] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 38] @DRAM
// )
void gemm_RVV_38x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(6));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(6));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[24],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(8));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 24],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(6));
}

// gemm_RVV_39x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 39] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 39] @DRAM
// )
void gemm_RVV_39x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(7));
}

// gemm_RVV_39x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 39] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 39] @DRAM
// )
void gemm_RVV_39x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(7));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[24],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(7));
}

// gemm_RVV_39x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 39] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 39] @DRAM
// )
void gemm_RVV_39x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(7));
}

// gemm_RVV_39x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 39] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 39] @DRAM
// )
void gemm_RVV_39x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[32],(7));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(7));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[24],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(8));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 24],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_regt_1,(7));
}

// gemm_RVV_3x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 3] @DRAM
// )
void gemm_RVV_3x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
}

// gemm_RVV_3x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 3] @DRAM
// )
void gemm_RVV_3x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
}

// gemm_RVV_3x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 3] @DRAM
// )
void gemm_RVV_3x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
}

// gemm_RVV_3x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 3] @DRAM
// )
void gemm_RVV_3x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
}

// gemm_RVV_40x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 40] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 40] @DRAM
// )
void gemm_RVV_40x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
}

// gemm_RVV_40x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 40] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 40] @DRAM
// )
void gemm_RVV_40x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[24],(8));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[32],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
}

// gemm_RVV_40x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 40] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 40] @DRAM
// )
void gemm_RVV_40x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(8));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_4,(8));
}

// gemm_RVV_40x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 40] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 40] @DRAM
// )
void gemm_RVV_40x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[24],(8));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[32],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(8));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 24],(8));
C_reg_1_4 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(8));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_4,(8));
}

// gemm_RVV_41x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 41] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 41] @DRAM
// )
void gemm_RVV_41x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 40],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
__riscv_vse16_v_f16m1(&C.data[40], C_regt_0,(1));
}

// gemm_RVV_41x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 41] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 41] @DRAM
// )
void gemm_RVV_41x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[40],(1));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[24],(8));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[32],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 40],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
__riscv_vse16_v_f16m1(&C.data[40], C_regt_0,(1));
}

// gemm_RVV_41x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 41] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 41] @DRAM
// )
void gemm_RVV_41x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 40],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(8));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_4,(8));
__riscv_vse16_v_f16m1(&C.data[40], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 40], C_regt_1,(1));
}

// gemm_RVV_41x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 41] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 41] @DRAM
// )
void gemm_RVV_41x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[40],(1));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 40],(1));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[24],(8));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[32],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(8));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 24],(8));
C_reg_1_4 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 40],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(8));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_4,(8));
__riscv_vse16_v_f16m1(&C.data[40], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 40], C_regt_1,(1));
}

// gemm_RVV_42x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 42] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 42] @DRAM
// )
void gemm_RVV_42x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 40],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
__riscv_vse16_v_f16m1(&C.data[40], C_regt_0,(2));
}

// gemm_RVV_42x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 42] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 42] @DRAM
// )
void gemm_RVV_42x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[40],(2));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[24],(8));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[32],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 40],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
__riscv_vse16_v_f16m1(&C.data[40], C_regt_0,(2));
}

// gemm_RVV_42x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 42] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 42] @DRAM
// )
void gemm_RVV_42x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 40],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(8));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_4,(8));
__riscv_vse16_v_f16m1(&C.data[40], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 40], C_regt_1,(2));
}

// gemm_RVV_42x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 42] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 42] @DRAM
// )
void gemm_RVV_42x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[40],(2));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 40],(2));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[24],(8));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[32],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(8));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 24],(8));
C_reg_1_4 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 40],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(8));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_4,(8));
__riscv_vse16_v_f16m1(&C.data[40], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 40], C_regt_1,(2));
}

// gemm_RVV_43x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 43] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 43] @DRAM
// )
void gemm_RVV_43x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 40],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
__riscv_vse16_v_f16m1(&C.data[40], C_regt_0,(3));
}

// gemm_RVV_43x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 43] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 43] @DRAM
// )
void gemm_RVV_43x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[40],(3));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[24],(8));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[32],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 40],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
__riscv_vse16_v_f16m1(&C.data[40], C_regt_0,(3));
}

// gemm_RVV_43x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 43] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 43] @DRAM
// )
void gemm_RVV_43x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 40],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(8));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_4,(8));
__riscv_vse16_v_f16m1(&C.data[40], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 40], C_regt_1,(3));
}

// gemm_RVV_43x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 43] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 43] @DRAM
// )
void gemm_RVV_43x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[40],(3));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 40],(3));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[24],(8));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[32],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(8));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 24],(8));
C_reg_1_4 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 40],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(8));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_4,(8));
__riscv_vse16_v_f16m1(&C.data[40], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 40], C_regt_1,(3));
}

// gemm_RVV_44x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 44] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 44] @DRAM
// )
void gemm_RVV_44x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 40],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
__riscv_vse16_v_f16m1(&C.data[40], C_regt_0,(4));
}

// gemm_RVV_44x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 44] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 44] @DRAM
// )
void gemm_RVV_44x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[40],(4));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[24],(8));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[32],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 40],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
__riscv_vse16_v_f16m1(&C.data[40], C_regt_0,(4));
}

// gemm_RVV_44x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 44] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 44] @DRAM
// )
void gemm_RVV_44x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 40],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(8));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_4,(8));
__riscv_vse16_v_f16m1(&C.data[40], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 40], C_regt_1,(4));
}

// gemm_RVV_44x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 44] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 44] @DRAM
// )
void gemm_RVV_44x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[40],(4));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 40],(4));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[24],(8));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[32],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(8));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 24],(8));
C_reg_1_4 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 40],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(8));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_4,(8));
__riscv_vse16_v_f16m1(&C.data[40], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 40], C_regt_1,(4));
}

// gemm_RVV_45x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 45] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 45] @DRAM
// )
void gemm_RVV_45x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 40],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
__riscv_vse16_v_f16m1(&C.data[40], C_regt_0,(5));
}

// gemm_RVV_45x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 45] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 45] @DRAM
// )
void gemm_RVV_45x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[40],(5));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[24],(8));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[32],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 40],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
__riscv_vse16_v_f16m1(&C.data[40], C_regt_0,(5));
}

// gemm_RVV_45x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 45] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 45] @DRAM
// )
void gemm_RVV_45x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 40],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(8));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_4,(8));
__riscv_vse16_v_f16m1(&C.data[40], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 40], C_regt_1,(5));
}

// gemm_RVV_45x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 45] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 45] @DRAM
// )
void gemm_RVV_45x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[40],(5));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 40],(5));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[24],(8));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[32],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(8));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 24],(8));
C_reg_1_4 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 40],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(8));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_4,(8));
__riscv_vse16_v_f16m1(&C.data[40], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 40], C_regt_1,(5));
}

// gemm_RVV_46x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 46] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 46] @DRAM
// )
void gemm_RVV_46x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 40],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
__riscv_vse16_v_f16m1(&C.data[40], C_regt_0,(6));
}

// gemm_RVV_46x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 46] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 46] @DRAM
// )
void gemm_RVV_46x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[40],(6));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[24],(8));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[32],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 40],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
__riscv_vse16_v_f16m1(&C.data[40], C_regt_0,(6));
}

// gemm_RVV_46x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 46] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 46] @DRAM
// )
void gemm_RVV_46x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 40],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(8));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_4,(8));
__riscv_vse16_v_f16m1(&C.data[40], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 40], C_regt_1,(6));
}

// gemm_RVV_46x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 46] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 46] @DRAM
// )
void gemm_RVV_46x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[40],(6));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 40],(6));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[24],(8));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[32],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(8));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 24],(8));
C_reg_1_4 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 40],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(8));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_4,(8));
__riscv_vse16_v_f16m1(&C.data[40], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 40], C_regt_1,(6));
}

// gemm_RVV_47x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 47] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 47] @DRAM
// )
void gemm_RVV_47x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 40],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
__riscv_vse16_v_f16m1(&C.data[40], C_regt_0,(7));
}

// gemm_RVV_47x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 47] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 47] @DRAM
// )
void gemm_RVV_47x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[40],(7));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[24],(8));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[32],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 40],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
__riscv_vse16_v_f16m1(&C.data[40], C_regt_0,(7));
}

// gemm_RVV_47x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 47] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 47] @DRAM
// )
void gemm_RVV_47x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 40],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(8));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_4,(8));
__riscv_vse16_v_f16m1(&C.data[40], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 40], C_regt_1,(7));
}

// gemm_RVV_47x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 47] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 47] @DRAM
// )
void gemm_RVV_47x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[40],(7));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 40],(7));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[24],(8));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[32],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(8));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 24],(8));
C_reg_1_4 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 40],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(8));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_4,(8));
__riscv_vse16_v_f16m1(&C.data[40], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 40], C_regt_1,(7));
}

// gemm_RVV_48x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 48] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 48] @DRAM
// )
void gemm_RVV_48x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_reg_5;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_0_5;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_5 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  A_reg_5 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 40],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
  C_reg_0_5 = __riscv_vfmacc_vv_f16m1(C_reg_0_5, A_reg_5, B_reg_0,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
__riscv_vse16_v_f16m1(&C.data[40], C_reg_0_5,(8));
}

// gemm_RVV_48x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 48] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 48] @DRAM
// )
void gemm_RVV_48x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_reg_5;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_0_5;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[24],(8));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[32],(8));
C_reg_0_5 = __riscv_vle16_v_f16m1(&C.data[40],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  A_reg_5 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 40],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
  C_reg_0_5 = __riscv_vfmacc_vv_f16m1(C_reg_0_5, A_reg_5, B_reg_0,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
__riscv_vse16_v_f16m1(&C.data[40], C_reg_0_5,(8));
}

// gemm_RVV_48x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 48] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 48] @DRAM
// )
void gemm_RVV_48x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_reg_5;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_0_5;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
vfloat16m1_t C_reg_1_5;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_5 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_5 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  A_reg_5 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 40],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
  C_reg_0_5 = __riscv_vfmacc_vv_f16m1(C_reg_0_5, A_reg_5, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(8));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(8));
  C_reg_1_5 = __riscv_vfmacc_vv_f16m1(C_reg_1_5, A_reg_5, B_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
__riscv_vse16_v_f16m1(&C.data[40], C_reg_0_5,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_4,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 40], C_reg_1_5,(8));
}

// gemm_RVV_48x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 48] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 48] @DRAM
// )
void gemm_RVV_48x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_reg_5;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_0_4;
vfloat16m1_t C_reg_0_5;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_1_4;
vfloat16m1_t C_reg_1_5;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[24],(8));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C.data[32],(8));
C_reg_0_5 = __riscv_vle16_v_f16m1(&C.data[40],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(8));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 24],(8));
C_reg_1_4 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(8));
C_reg_1_5 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 40],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 24],(8));
  A_reg_4 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(8));
  A_reg_5 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 40],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(8));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(8));
  C_reg_0_3 = __riscv_vfmacc_vv_f16m1(C_reg_0_3, A_reg_3, B_reg_0,(8));
  C_reg_0_4 = __riscv_vfmacc_vv_f16m1(C_reg_0_4, A_reg_4, B_reg_0,(8));
  C_reg_0_5 = __riscv_vfmacc_vv_f16m1(C_reg_0_5, A_reg_5, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(8));
  C_reg_1_3 = __riscv_vfmacc_vv_f16m1(C_reg_1_3, A_reg_3, B_reg_1,(8));
  C_reg_1_4 = __riscv_vfmacc_vv_f16m1(C_reg_1_4, A_reg_4, B_reg_1,(8));
  C_reg_1_5 = __riscv_vfmacc_vv_f16m1(C_reg_1_5, A_reg_5, B_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C.data[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_4,(8));
__riscv_vse16_v_f16m1(&C.data[40], C_reg_0_5,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_4,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 40], C_reg_1_5,(8));
}

// gemm_RVV_4x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 4] @DRAM
// )
void gemm_RVV_4x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
}

// gemm_RVV_4x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 4] @DRAM
// )
void gemm_RVV_4x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
}

// gemm_RVV_4x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 4] @DRAM
// )
void gemm_RVV_4x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
}

// gemm_RVV_4x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 4] @DRAM
// )
void gemm_RVV_4x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
}

// gemm_RVV_5x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 5] @DRAM
// )
void gemm_RVV_5x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
}

// gemm_RVV_5x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 5] @DRAM
// )
void gemm_RVV_5x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
}

// gemm_RVV_5x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 5] @DRAM
// )
void gemm_RVV_5x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
}

// gemm_RVV_5x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 5] @DRAM
// )
void gemm_RVV_5x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
}

// gemm_RVV_6x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 6] @DRAM
// )
void gemm_RVV_6x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
}

// gemm_RVV_6x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 6] @DRAM
// )
void gemm_RVV_6x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
}

// gemm_RVV_6x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 6] @DRAM
// )
void gemm_RVV_6x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
}

// gemm_RVV_6x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 6] @DRAM
// )
void gemm_RVV_6x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
}

// gemm_RVV_7x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 7] @DRAM
// )
void gemm_RVV_7x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
}

// gemm_RVV_7x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 7] @DRAM
// )
void gemm_RVV_7x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
}

// gemm_RVV_7x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 7] @DRAM
// )
void gemm_RVV_7x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
}

// gemm_RVV_7x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 7] @DRAM
// )
void gemm_RVV_7x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
}

// gemm_RVV_8x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 8] @DRAM
// )
void gemm_RVV_8x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
}

// gemm_RVV_8x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 8] @DRAM
// )
void gemm_RVV_8x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
}

// gemm_RVV_8x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 8] @DRAM
// )
void gemm_RVV_8x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
}

// gemm_RVV_8x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 8] @DRAM
// )
void gemm_RVV_8x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
}

// gemm_RVV_9x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 9] @DRAM
// )
void gemm_RVV_9x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_regt_0,(1));
}

// gemm_RVV_9x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 9] @DRAM
// )
void gemm_RVV_9x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[8],(1));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_regt_0,(1));
}

// gemm_RVV_9x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 9] @DRAM
// )
void gemm_RVV_9x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_regt_1,(1));
}

// gemm_RVV_9x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 9] @DRAM
// )
void gemm_RVV_9x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C.data[8],(1));
C_regt_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 8],(1));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 8],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0])],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B.data[(k) * (B.strides[0]) + 1],(8));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C.data[8], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 8], C_regt_1,(1));
}


/* relying on the following instruction..."
rvv_broadcast_8xf16(dst,src,vl)
{dst_data} = __riscv_vfmv_v_f_f16m1({src_data},{vl});
*/

/* relying on the following instruction..."
rvv_broadcast_8xf16_0(dst,vl)
{dst_data} = __riscv_vfmv_v_f_f16m1(0.0f,{vl});
*/

/* relying on the following instruction..."
rvv_vfmacc_8xf16_8xf16(dst,lhs,rhs,vl)
{dst_data} = __riscv_vfmacc_vv_f16m1({dst_data}, {lhs_data}, {rhs_data},{vl});
*/

/* relying on the following instruction..."
rvv_vld_8xf16(dst,src,vl)
{dst_data} = __riscv_vle16_v_f16m1(&{src_data},{vl});
*/

/* relying on the following instruction..."
rvv_vst_8xf16(dst,src,vl)
__riscv_vse16_v_f16m1(&{dst_data}, {src_data},{vl});
*/
