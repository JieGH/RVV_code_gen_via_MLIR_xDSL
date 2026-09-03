#include "kernels_RVV_16x6_fp16.h"



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
void gemm_RVV_10x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(2));
}

// gemm_RVV_10x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 10] @DRAM
// )
void gemm_RVV_10x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(2));
}

// gemm_RVV_10x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 10] @DRAM
// )
void gemm_RVV_10x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(2));
}

// gemm_RVV_10x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 10] @DRAM
// )
void gemm_RVV_10x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(2));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(2));
}

// gemm_RVV_10x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 10] @DRAM
// )
void gemm_RVV_10x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(2));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(2));
}

// gemm_RVV_10x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 10] @DRAM
// )
void gemm_RVV_10x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(2));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(2));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(2));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(2));
}

// gemm_RVV_10x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 10] @DRAM
// )
void gemm_RVV_10x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(2));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(2));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(2));
}

// gemm_RVV_10x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 10] @DRAM
// )
void gemm_RVV_10x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(2));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(2));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(2));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 8],(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(2));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(2));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(2));
}

// gemm_RVV_10x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 10] @DRAM
// )
void gemm_RVV_10x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_regt_4;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_4_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_4 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (6) + 4], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(2));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(2));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(2));
  C_regt_4 = __riscv_vfmacc_vf_f16m1(C_regt_4, B[(k) * (6) + 4], A_regt,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 8], C_regt_4,(2));
}

// gemm_RVV_10x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 10] @DRAM
// )
void gemm_RVV_10x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_regt_4;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_4_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_4_0 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(2));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(2));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(2));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 8],(2));
C_regt_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc) + 8],(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (6) + 4], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(2));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(2));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(2));
  C_regt_4 = __riscv_vfmacc_vf_f16m1(C_regt_4, B[(k) * (6) + 4], A_regt,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 8], C_regt_4,(2));
}

// gemm_RVV_10x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 10] @DRAM
// )
void gemm_RVV_10x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_regt_4;
vfloat16m1_t C_regt_5;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_4_0;
vfloat16m1_t C_reg_5_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_5_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_4 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_5 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (6) + 4], A_reg_0,(8));
  C_reg_5_0 = __riscv_vfmacc_vf_f16m1(C_reg_5_0, B[(k) * (6) + 5], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(2));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(2));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(2));
  C_regt_4 = __riscv_vfmacc_vf_f16m1(C_regt_4, B[(k) * (6) + 4], A_regt,(2));
  C_regt_5 = __riscv_vfmacc_vf_f16m1(C_regt_5, B[(k) * (6) + 5], A_regt,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 8], C_regt_4,(2));
__riscv_vse16_v_f16m1(&C[(5) * (ldc) + 8], C_regt_5,(2));
}

// gemm_RVV_10x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 10] @DRAM
// )
void gemm_RVV_10x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_regt_4;
vfloat16m1_t C_regt_5;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_4_0;
vfloat16m1_t C_reg_5_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_4_0 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(8));
C_reg_5_0 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(2));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(2));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(2));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 8],(2));
C_regt_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc) + 8],(2));
C_regt_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc) + 8],(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (6) + 4], A_reg_0,(8));
  C_reg_5_0 = __riscv_vfmacc_vf_f16m1(C_reg_5_0, B[(k) * (6) + 5], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(2));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(2));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(2));
  C_regt_4 = __riscv_vfmacc_vf_f16m1(C_regt_4, B[(k) * (6) + 4], A_regt,(2));
  C_regt_5 = __riscv_vfmacc_vf_f16m1(C_regt_5, B[(k) * (6) + 5], A_regt,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 8], C_regt_4,(2));
__riscv_vse16_v_f16m1(&C[(5) * (ldc) + 8], C_regt_5,(2));
}

// gemm_RVV_11x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 11] @DRAM
// )
void gemm_RVV_11x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(3));
}

// gemm_RVV_11x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 11] @DRAM
// )
void gemm_RVV_11x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(3));
}

// gemm_RVV_11x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 11] @DRAM
// )
void gemm_RVV_11x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(3));
}

// gemm_RVV_11x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 11] @DRAM
// )
void gemm_RVV_11x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(3));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(3));
}

// gemm_RVV_11x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 11] @DRAM
// )
void gemm_RVV_11x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(3));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(3));
}

// gemm_RVV_11x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 11] @DRAM
// )
void gemm_RVV_11x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(3));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(3));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(3));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(3));
}

// gemm_RVV_11x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 11] @DRAM
// )
void gemm_RVV_11x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(3));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(3));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(3));
}

// gemm_RVV_11x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 11] @DRAM
// )
void gemm_RVV_11x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(3));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(3));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(3));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 8],(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(3));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(3));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(3));
}

// gemm_RVV_11x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 11] @DRAM
// )
void gemm_RVV_11x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_regt_4;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_4_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_4 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (6) + 4], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(3));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(3));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(3));
  C_regt_4 = __riscv_vfmacc_vf_f16m1(C_regt_4, B[(k) * (6) + 4], A_regt,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 8], C_regt_4,(3));
}

// gemm_RVV_11x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 11] @DRAM
// )
void gemm_RVV_11x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_regt_4;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_4_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_4_0 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(3));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(3));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(3));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 8],(3));
C_regt_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc) + 8],(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (6) + 4], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(3));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(3));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(3));
  C_regt_4 = __riscv_vfmacc_vf_f16m1(C_regt_4, B[(k) * (6) + 4], A_regt,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 8], C_regt_4,(3));
}

// gemm_RVV_11x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 11] @DRAM
// )
void gemm_RVV_11x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_regt_4;
vfloat16m1_t C_regt_5;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_4_0;
vfloat16m1_t C_reg_5_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_5_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_4 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_5 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (6) + 4], A_reg_0,(8));
  C_reg_5_0 = __riscv_vfmacc_vf_f16m1(C_reg_5_0, B[(k) * (6) + 5], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(3));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(3));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(3));
  C_regt_4 = __riscv_vfmacc_vf_f16m1(C_regt_4, B[(k) * (6) + 4], A_regt,(3));
  C_regt_5 = __riscv_vfmacc_vf_f16m1(C_regt_5, B[(k) * (6) + 5], A_regt,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 8], C_regt_4,(3));
__riscv_vse16_v_f16m1(&C[(5) * (ldc) + 8], C_regt_5,(3));
}

// gemm_RVV_11x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 11] @DRAM
// )
void gemm_RVV_11x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_regt_4;
vfloat16m1_t C_regt_5;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_4_0;
vfloat16m1_t C_reg_5_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_4_0 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(8));
C_reg_5_0 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(3));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(3));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(3));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 8],(3));
C_regt_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc) + 8],(3));
C_regt_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc) + 8],(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (6) + 4], A_reg_0,(8));
  C_reg_5_0 = __riscv_vfmacc_vf_f16m1(C_reg_5_0, B[(k) * (6) + 5], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(3));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(3));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(3));
  C_regt_4 = __riscv_vfmacc_vf_f16m1(C_regt_4, B[(k) * (6) + 4], A_regt,(3));
  C_regt_5 = __riscv_vfmacc_vf_f16m1(C_regt_5, B[(k) * (6) + 5], A_regt,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 8], C_regt_4,(3));
__riscv_vse16_v_f16m1(&C[(5) * (ldc) + 8], C_regt_5,(3));
}

// gemm_RVV_12x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 12] @DRAM
// )
void gemm_RVV_12x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(4));
}

// gemm_RVV_12x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 12] @DRAM
// )
void gemm_RVV_12x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(4));
}

// gemm_RVV_12x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 12] @DRAM
// )
void gemm_RVV_12x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(4));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(4));
}

// gemm_RVV_12x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 12] @DRAM
// )
void gemm_RVV_12x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(4));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(4));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(4));
}

// gemm_RVV_12x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 12] @DRAM
// )
void gemm_RVV_12x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(4));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(4));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(4));
}

// gemm_RVV_12x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 12] @DRAM
// )
void gemm_RVV_12x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(4));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(4));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(4));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(4));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(4));
}

// gemm_RVV_12x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 12] @DRAM
// )
void gemm_RVV_12x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(4));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(4));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(4));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(4));
}

// gemm_RVV_12x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 12] @DRAM
// )
void gemm_RVV_12x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(4));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(4));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(4));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(4));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(4));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(4));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(4));
}

// gemm_RVV_12x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 12] @DRAM
// )
void gemm_RVV_12x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_regt_4;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_4_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_4 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (6) + 4], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(4));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(4));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(4));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(4));
  C_regt_4 = __riscv_vfmacc_vf_f16m1(C_regt_4, B[(k) * (6) + 4], A_regt,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 8], C_regt_4,(4));
}

// gemm_RVV_12x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 12] @DRAM
// )
void gemm_RVV_12x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_regt_4;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_4_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_4_0 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(4));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(4));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(4));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 8],(4));
C_regt_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc) + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (6) + 4], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(4));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(4));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(4));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(4));
  C_regt_4 = __riscv_vfmacc_vf_f16m1(C_regt_4, B[(k) * (6) + 4], A_regt,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 8], C_regt_4,(4));
}

// gemm_RVV_12x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 12] @DRAM
// )
void gemm_RVV_12x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_regt_4;
vfloat16m1_t C_regt_5;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_4_0;
vfloat16m1_t C_reg_5_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_5_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_4 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_5 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (6) + 4], A_reg_0,(8));
  C_reg_5_0 = __riscv_vfmacc_vf_f16m1(C_reg_5_0, B[(k) * (6) + 5], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(4));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(4));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(4));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(4));
  C_regt_4 = __riscv_vfmacc_vf_f16m1(C_regt_4, B[(k) * (6) + 4], A_regt,(4));
  C_regt_5 = __riscv_vfmacc_vf_f16m1(C_regt_5, B[(k) * (6) + 5], A_regt,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 8], C_regt_4,(4));
__riscv_vse16_v_f16m1(&C[(5) * (ldc) + 8], C_regt_5,(4));
}

// gemm_RVV_12x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 12] @DRAM
// )
void gemm_RVV_12x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_regt_4;
vfloat16m1_t C_regt_5;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_4_0;
vfloat16m1_t C_reg_5_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_4_0 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(8));
C_reg_5_0 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(4));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(4));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(4));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 8],(4));
C_regt_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc) + 8],(4));
C_regt_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc) + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (6) + 4], A_reg_0,(8));
  C_reg_5_0 = __riscv_vfmacc_vf_f16m1(C_reg_5_0, B[(k) * (6) + 5], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(4));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(4));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(4));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(4));
  C_regt_4 = __riscv_vfmacc_vf_f16m1(C_regt_4, B[(k) * (6) + 4], A_regt,(4));
  C_regt_5 = __riscv_vfmacc_vf_f16m1(C_regt_5, B[(k) * (6) + 5], A_regt,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 8], C_regt_4,(4));
__riscv_vse16_v_f16m1(&C[(5) * (ldc) + 8], C_regt_5,(4));
}

// gemm_RVV_13x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 13] @DRAM
// )
void gemm_RVV_13x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(5));
}

// gemm_RVV_13x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 13] @DRAM
// )
void gemm_RVV_13x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(5));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(5));
}

// gemm_RVV_13x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 13] @DRAM
// )
void gemm_RVV_13x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(5));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(5));
}

// gemm_RVV_13x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 13] @DRAM
// )
void gemm_RVV_13x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(5));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(5));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(5));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(5));
}

// gemm_RVV_13x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 13] @DRAM
// )
void gemm_RVV_13x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(5));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(5));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(5));
}

// gemm_RVV_13x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 13] @DRAM
// )
void gemm_RVV_13x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(5));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(5));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(5));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(5));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(5));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(5));
}

// gemm_RVV_13x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 13] @DRAM
// )
void gemm_RVV_13x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(5));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(5));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(5));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(5));
}

// gemm_RVV_13x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 13] @DRAM
// )
void gemm_RVV_13x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(5));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(5));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(5));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 8],(5));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(5));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(5));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(5));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(5));
}

// gemm_RVV_13x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 13] @DRAM
// )
void gemm_RVV_13x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_regt_4;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_4_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_4 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (6) + 4], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(5));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(5));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(5));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(5));
  C_regt_4 = __riscv_vfmacc_vf_f16m1(C_regt_4, B[(k) * (6) + 4], A_regt,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 8], C_regt_4,(5));
}

// gemm_RVV_13x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 13] @DRAM
// )
void gemm_RVV_13x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_regt_4;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_4_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_4_0 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(5));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(5));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(5));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 8],(5));
C_regt_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc) + 8],(5));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (6) + 4], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(5));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(5));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(5));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(5));
  C_regt_4 = __riscv_vfmacc_vf_f16m1(C_regt_4, B[(k) * (6) + 4], A_regt,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 8], C_regt_4,(5));
}

// gemm_RVV_13x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 13] @DRAM
// )
void gemm_RVV_13x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_regt_4;
vfloat16m1_t C_regt_5;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_4_0;
vfloat16m1_t C_reg_5_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_5_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_4 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_5 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (6) + 4], A_reg_0,(8));
  C_reg_5_0 = __riscv_vfmacc_vf_f16m1(C_reg_5_0, B[(k) * (6) + 5], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(5));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(5));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(5));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(5));
  C_regt_4 = __riscv_vfmacc_vf_f16m1(C_regt_4, B[(k) * (6) + 4], A_regt,(5));
  C_regt_5 = __riscv_vfmacc_vf_f16m1(C_regt_5, B[(k) * (6) + 5], A_regt,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 8], C_regt_4,(5));
__riscv_vse16_v_f16m1(&C[(5) * (ldc) + 8], C_regt_5,(5));
}

// gemm_RVV_13x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 13] @DRAM
// )
void gemm_RVV_13x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_regt_4;
vfloat16m1_t C_regt_5;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_4_0;
vfloat16m1_t C_reg_5_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_4_0 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(8));
C_reg_5_0 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(5));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(5));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(5));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 8],(5));
C_regt_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc) + 8],(5));
C_regt_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc) + 8],(5));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (6) + 4], A_reg_0,(8));
  C_reg_5_0 = __riscv_vfmacc_vf_f16m1(C_reg_5_0, B[(k) * (6) + 5], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(5));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(5));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(5));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(5));
  C_regt_4 = __riscv_vfmacc_vf_f16m1(C_regt_4, B[(k) * (6) + 4], A_regt,(5));
  C_regt_5 = __riscv_vfmacc_vf_f16m1(C_regt_5, B[(k) * (6) + 5], A_regt,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 8], C_regt_4,(5));
__riscv_vse16_v_f16m1(&C[(5) * (ldc) + 8], C_regt_5,(5));
}

// gemm_RVV_14x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 14] @DRAM
// )
void gemm_RVV_14x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(6));
}

// gemm_RVV_14x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 14] @DRAM
// )
void gemm_RVV_14x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(6));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(6));
}

// gemm_RVV_14x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 14] @DRAM
// )
void gemm_RVV_14x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(6));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(6));
}

// gemm_RVV_14x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 14] @DRAM
// )
void gemm_RVV_14x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(6));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(6));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(6));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(6));
}

// gemm_RVV_14x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 14] @DRAM
// )
void gemm_RVV_14x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(6));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(6));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(6));
}

// gemm_RVV_14x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 14] @DRAM
// )
void gemm_RVV_14x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(6));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(6));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(6));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(6));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(6));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(6));
}

// gemm_RVV_14x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 14] @DRAM
// )
void gemm_RVV_14x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(6));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(6));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(6));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(6));
}

// gemm_RVV_14x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 14] @DRAM
// )
void gemm_RVV_14x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(6));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(6));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(6));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 8],(6));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(6));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(6));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(6));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(6));
}

// gemm_RVV_14x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 14] @DRAM
// )
void gemm_RVV_14x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_regt_4;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_4_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_4 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (6) + 4], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(6));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(6));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(6));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(6));
  C_regt_4 = __riscv_vfmacc_vf_f16m1(C_regt_4, B[(k) * (6) + 4], A_regt,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 8], C_regt_4,(6));
}

// gemm_RVV_14x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 14] @DRAM
// )
void gemm_RVV_14x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_regt_4;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_4_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_4_0 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(6));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(6));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(6));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 8],(6));
C_regt_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc) + 8],(6));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (6) + 4], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(6));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(6));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(6));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(6));
  C_regt_4 = __riscv_vfmacc_vf_f16m1(C_regt_4, B[(k) * (6) + 4], A_regt,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 8], C_regt_4,(6));
}

// gemm_RVV_14x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 14] @DRAM
// )
void gemm_RVV_14x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_regt_4;
vfloat16m1_t C_regt_5;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_4_0;
vfloat16m1_t C_reg_5_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_5_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_4 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_5 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (6) + 4], A_reg_0,(8));
  C_reg_5_0 = __riscv_vfmacc_vf_f16m1(C_reg_5_0, B[(k) * (6) + 5], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(6));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(6));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(6));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(6));
  C_regt_4 = __riscv_vfmacc_vf_f16m1(C_regt_4, B[(k) * (6) + 4], A_regt,(6));
  C_regt_5 = __riscv_vfmacc_vf_f16m1(C_regt_5, B[(k) * (6) + 5], A_regt,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 8], C_regt_4,(6));
__riscv_vse16_v_f16m1(&C[(5) * (ldc) + 8], C_regt_5,(6));
}

// gemm_RVV_14x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 14] @DRAM
// )
void gemm_RVV_14x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_regt_4;
vfloat16m1_t C_regt_5;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_4_0;
vfloat16m1_t C_reg_5_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_4_0 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(8));
C_reg_5_0 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(6));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(6));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(6));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 8],(6));
C_regt_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc) + 8],(6));
C_regt_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc) + 8],(6));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (6) + 4], A_reg_0,(8));
  C_reg_5_0 = __riscv_vfmacc_vf_f16m1(C_reg_5_0, B[(k) * (6) + 5], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(6));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(6));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(6));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(6));
  C_regt_4 = __riscv_vfmacc_vf_f16m1(C_regt_4, B[(k) * (6) + 4], A_regt,(6));
  C_regt_5 = __riscv_vfmacc_vf_f16m1(C_regt_5, B[(k) * (6) + 5], A_regt,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 8], C_regt_4,(6));
__riscv_vse16_v_f16m1(&C[(5) * (ldc) + 8], C_regt_5,(6));
}

// gemm_RVV_15x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 15] @DRAM
// )
void gemm_RVV_15x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(7));
}

// gemm_RVV_15x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 15] @DRAM
// )
void gemm_RVV_15x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(7));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(7));
}

// gemm_RVV_15x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 15] @DRAM
// )
void gemm_RVV_15x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(7));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(7));
}

// gemm_RVV_15x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 15] @DRAM
// )
void gemm_RVV_15x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(7));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(7));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(7));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(7));
}

// gemm_RVV_15x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 15] @DRAM
// )
void gemm_RVV_15x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(7));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(7));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(7));
}

// gemm_RVV_15x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 15] @DRAM
// )
void gemm_RVV_15x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(7));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(7));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(7));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(7));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(7));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(7));
}

// gemm_RVV_15x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 15] @DRAM
// )
void gemm_RVV_15x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(7));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(7));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(7));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(7));
}

// gemm_RVV_15x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 15] @DRAM
// )
void gemm_RVV_15x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(7));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(7));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(7));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 8],(7));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(7));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(7));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(7));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(7));
}

// gemm_RVV_15x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 15] @DRAM
// )
void gemm_RVV_15x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_regt_4;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_4_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_4 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (6) + 4], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(7));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(7));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(7));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(7));
  C_regt_4 = __riscv_vfmacc_vf_f16m1(C_regt_4, B[(k) * (6) + 4], A_regt,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 8], C_regt_4,(7));
}

// gemm_RVV_15x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 15] @DRAM
// )
void gemm_RVV_15x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_regt_4;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_4_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_4_0 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(7));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(7));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(7));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 8],(7));
C_regt_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc) + 8],(7));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (6) + 4], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(7));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(7));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(7));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(7));
  C_regt_4 = __riscv_vfmacc_vf_f16m1(C_regt_4, B[(k) * (6) + 4], A_regt,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 8], C_regt_4,(7));
}

// gemm_RVV_15x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 15] @DRAM
// )
void gemm_RVV_15x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_regt_4;
vfloat16m1_t C_regt_5;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_4_0;
vfloat16m1_t C_reg_5_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_5_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_4 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_5 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (6) + 4], A_reg_0,(8));
  C_reg_5_0 = __riscv_vfmacc_vf_f16m1(C_reg_5_0, B[(k) * (6) + 5], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(7));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(7));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(7));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(7));
  C_regt_4 = __riscv_vfmacc_vf_f16m1(C_regt_4, B[(k) * (6) + 4], A_regt,(7));
  C_regt_5 = __riscv_vfmacc_vf_f16m1(C_regt_5, B[(k) * (6) + 5], A_regt,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 8], C_regt_4,(7));
__riscv_vse16_v_f16m1(&C[(5) * (ldc) + 8], C_regt_5,(7));
}

// gemm_RVV_15x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 15] @DRAM
// )
void gemm_RVV_15x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_regt_4;
vfloat16m1_t C_regt_5;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_4_0;
vfloat16m1_t C_reg_5_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_4_0 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(8));
C_reg_5_0 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(7));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(7));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(7));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 8],(7));
C_regt_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc) + 8],(7));
C_regt_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc) + 8],(7));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (6) + 4], A_reg_0,(8));
  C_reg_5_0 = __riscv_vfmacc_vf_f16m1(C_reg_5_0, B[(k) * (6) + 5], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(7));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(7));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(7));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(7));
  C_regt_4 = __riscv_vfmacc_vf_f16m1(C_regt_4, B[(k) * (6) + 4], A_regt,(7));
  C_regt_5 = __riscv_vfmacc_vf_f16m1(C_regt_5, B[(k) * (6) + 5], A_regt,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 8], C_regt_4,(7));
__riscv_vse16_v_f16m1(&C[(5) * (ldc) + 8], C_regt_5,(7));
}

// gemm_RVV_16x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 16] @DRAM
// )
void gemm_RVV_16x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (6)], A_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
}

// gemm_RVV_16x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 16] @DRAM
// )
void gemm_RVV_16x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (6)], A_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
}

// gemm_RVV_16x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 16] @DRAM
// )
void gemm_RVV_16x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (6)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (6) + 1], A_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
}

// gemm_RVV_16x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 16] @DRAM
// )
void gemm_RVV_16x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (6)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (6) + 1], A_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
}

// gemm_RVV_16x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 16] @DRAM
// )
void gemm_RVV_16x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (6)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (6) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (6) + 2], A_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
}

// gemm_RVV_16x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 16] @DRAM
// )
void gemm_RVV_16x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (6)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (6) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (6) + 2], A_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
}

// gemm_RVV_16x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 16] @DRAM
// )
void gemm_RVV_16x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_3_1;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (6)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (6) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (6) + 2], A_reg_1,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (6) + 3], A_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
}

// gemm_RVV_16x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 16] @DRAM
// )
void gemm_RVV_16x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_3_1;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(8));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_3_1 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 8],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (6)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (6) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (6) + 2], A_reg_1,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (6) + 3], A_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
}

// gemm_RVV_16x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 16] @DRAM
// )
void gemm_RVV_16x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_3_1;
vfloat16m1_t C_reg_4_0;
vfloat16m1_t C_reg_4_1;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (6)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (6) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (6) + 2], A_reg_1,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (6) + 3], A_reg_1,(8));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (6) + 4], A_reg_0,(8));
  C_reg_4_1 = __riscv_vfmacc_vf_f16m1(C_reg_4_1, B[(k) * (6) + 4], A_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 8], C_reg_4_1,(8));
}

// gemm_RVV_16x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 16] @DRAM
// )
void gemm_RVV_16x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_3_1;
vfloat16m1_t C_reg_4_0;
vfloat16m1_t C_reg_4_1;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(8));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_3_1 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 8],(8));
C_reg_4_0 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(8));
C_reg_4_1 = __riscv_vle16_v_f16m1(&C[(4) * (ldc) + 8],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (6)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (6) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (6) + 2], A_reg_1,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (6) + 3], A_reg_1,(8));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (6) + 4], A_reg_0,(8));
  C_reg_4_1 = __riscv_vfmacc_vf_f16m1(C_reg_4_1, B[(k) * (6) + 4], A_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 8], C_reg_4_1,(8));
}

// gemm_RVV_16x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 16] @DRAM
// )
void gemm_RVV_16x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_3_1;
vfloat16m1_t C_reg_4_0;
vfloat16m1_t C_reg_4_1;
vfloat16m1_t C_reg_5_0;
vfloat16m1_t C_reg_5_1;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_5_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_5_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (6)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (6) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (6) + 2], A_reg_1,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (6) + 3], A_reg_1,(8));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (6) + 4], A_reg_0,(8));
  C_reg_4_1 = __riscv_vfmacc_vf_f16m1(C_reg_4_1, B[(k) * (6) + 4], A_reg_1,(8));
  C_reg_5_0 = __riscv_vfmacc_vf_f16m1(C_reg_5_0, B[(k) * (6) + 5], A_reg_0,(8));
  C_reg_5_1 = __riscv_vfmacc_vf_f16m1(C_reg_5_1, B[(k) * (6) + 5], A_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 8], C_reg_4_1,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5_0,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc) + 8], C_reg_5_1,(8));
}

// gemm_RVV_16x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 16] @DRAM
// )
void gemm_RVV_16x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_3_1;
vfloat16m1_t C_reg_4_0;
vfloat16m1_t C_reg_4_1;
vfloat16m1_t C_reg_5_0;
vfloat16m1_t C_reg_5_1;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(8));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_3_1 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 8],(8));
C_reg_4_0 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(8));
C_reg_4_1 = __riscv_vle16_v_f16m1(&C[(4) * (ldc) + 8],(8));
C_reg_5_0 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(8));
C_reg_5_1 = __riscv_vle16_v_f16m1(&C[(5) * (ldc) + 8],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (6)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (6) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (6) + 2], A_reg_1,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (6) + 3], A_reg_1,(8));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (6) + 4], A_reg_0,(8));
  C_reg_4_1 = __riscv_vfmacc_vf_f16m1(C_reg_4_1, B[(k) * (6) + 4], A_reg_1,(8));
  C_reg_5_0 = __riscv_vfmacc_vf_f16m1(C_reg_5_0, B[(k) * (6) + 5], A_reg_0,(8));
  C_reg_5_1 = __riscv_vfmacc_vf_f16m1(C_reg_5_1, B[(k) * (6) + 5], A_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 8], C_reg_4_1,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5_0,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc) + 8], C_reg_5_1,(8));
}

// gemm_RVV_1x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 1] @DRAM
// )
void gemm_RVV_1x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
}

// gemm_RVV_1x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 1] @DRAM
// )
void gemm_RVV_1x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
}

// gemm_RVV_1x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 1] @DRAM
// )
void gemm_RVV_1x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
}

// gemm_RVV_1x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 1] @DRAM
// )
void gemm_RVV_1x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
}

// gemm_RVV_1x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 1] @DRAM
// )
void gemm_RVV_1x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
}

// gemm_RVV_1x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 1] @DRAM
// )
void gemm_RVV_1x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
}

// gemm_RVV_1x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 1] @DRAM
// )
void gemm_RVV_1x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
}

// gemm_RVV_1x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 1] @DRAM
// )
void gemm_RVV_1x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
}

// gemm_RVV_1x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 1] @DRAM
// )
void gemm_RVV_1x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (6) + 4], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(1));
}

// gemm_RVV_1x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 1] @DRAM
// )
void gemm_RVV_1x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (6) + 4], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(1));
}

// gemm_RVV_1x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 1] @DRAM
// )
void gemm_RVV_1x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (6) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (6) + 5], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(1));
}

// gemm_RVV_1x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 1] @DRAM
// )
void gemm_RVV_1x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(1));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (6) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (6) + 5], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(1));
}

// gemm_RVV_2x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 2] @DRAM
// )
void gemm_RVV_2x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
}

// gemm_RVV_2x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 2] @DRAM
// )
void gemm_RVV_2x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
}

// gemm_RVV_2x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 2] @DRAM
// )
void gemm_RVV_2x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
}

// gemm_RVV_2x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 2] @DRAM
// )
void gemm_RVV_2x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
}

// gemm_RVV_2x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 2] @DRAM
// )
void gemm_RVV_2x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
}

// gemm_RVV_2x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 2] @DRAM
// )
void gemm_RVV_2x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
}

// gemm_RVV_2x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 2] @DRAM
// )
void gemm_RVV_2x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
}

// gemm_RVV_2x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 2] @DRAM
// )
void gemm_RVV_2x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
}

// gemm_RVV_2x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 2] @DRAM
// )
void gemm_RVV_2x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (6) + 4], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(2));
}

// gemm_RVV_2x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 2] @DRAM
// )
void gemm_RVV_2x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (6) + 4], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(2));
}

// gemm_RVV_2x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 2] @DRAM
// )
void gemm_RVV_2x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (6) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (6) + 5], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(2));
}

// gemm_RVV_2x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 2] @DRAM
// )
void gemm_RVV_2x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(2));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (6) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (6) + 5], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(2));
}

// gemm_RVV_3x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 3] @DRAM
// )
void gemm_RVV_3x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
}

// gemm_RVV_3x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 3] @DRAM
// )
void gemm_RVV_3x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
}

// gemm_RVV_3x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 3] @DRAM
// )
void gemm_RVV_3x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
}

// gemm_RVV_3x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 3] @DRAM
// )
void gemm_RVV_3x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
}

// gemm_RVV_3x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 3] @DRAM
// )
void gemm_RVV_3x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
}

// gemm_RVV_3x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 3] @DRAM
// )
void gemm_RVV_3x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
}

// gemm_RVV_3x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 3] @DRAM
// )
void gemm_RVV_3x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
}

// gemm_RVV_3x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 3] @DRAM
// )
void gemm_RVV_3x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
}

// gemm_RVV_3x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 3] @DRAM
// )
void gemm_RVV_3x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (6) + 4], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(3));
}

// gemm_RVV_3x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 3] @DRAM
// )
void gemm_RVV_3x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (6) + 4], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(3));
}

// gemm_RVV_3x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 3] @DRAM
// )
void gemm_RVV_3x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (6) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (6) + 5], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(3));
}

// gemm_RVV_3x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 3] @DRAM
// )
void gemm_RVV_3x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(3));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (6) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (6) + 5], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(3));
}

// gemm_RVV_4x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 4] @DRAM
// )
void gemm_RVV_4x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
}

// gemm_RVV_4x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 4] @DRAM
// )
void gemm_RVV_4x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
}

// gemm_RVV_4x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 4] @DRAM
// )
void gemm_RVV_4x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
}

// gemm_RVV_4x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 4] @DRAM
// )
void gemm_RVV_4x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
}

// gemm_RVV_4x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 4] @DRAM
// )
void gemm_RVV_4x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
}

// gemm_RVV_4x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 4] @DRAM
// )
void gemm_RVV_4x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
}

// gemm_RVV_4x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 4] @DRAM
// )
void gemm_RVV_4x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
}

// gemm_RVV_4x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 4] @DRAM
// )
void gemm_RVV_4x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
}

// gemm_RVV_4x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 4] @DRAM
// )
void gemm_RVV_4x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (6) + 4], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(4));
}

// gemm_RVV_4x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 4] @DRAM
// )
void gemm_RVV_4x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (6) + 4], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(4));
}

// gemm_RVV_4x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 4] @DRAM
// )
void gemm_RVV_4x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (6) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (6) + 5], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(4));
}

// gemm_RVV_4x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 4] @DRAM
// )
void gemm_RVV_4x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(4));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (6) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (6) + 5], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(4));
}

// gemm_RVV_5x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 5] @DRAM
// )
void gemm_RVV_5x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
}

// gemm_RVV_5x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 5] @DRAM
// )
void gemm_RVV_5x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
}

// gemm_RVV_5x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 5] @DRAM
// )
void gemm_RVV_5x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
}

// gemm_RVV_5x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 5] @DRAM
// )
void gemm_RVV_5x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
}

// gemm_RVV_5x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 5] @DRAM
// )
void gemm_RVV_5x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
}

// gemm_RVV_5x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 5] @DRAM
// )
void gemm_RVV_5x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
}

// gemm_RVV_5x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 5] @DRAM
// )
void gemm_RVV_5x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
}

// gemm_RVV_5x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 5] @DRAM
// )
void gemm_RVV_5x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
}

// gemm_RVV_5x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 5] @DRAM
// )
void gemm_RVV_5x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (6) + 4], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(5));
}

// gemm_RVV_5x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 5] @DRAM
// )
void gemm_RVV_5x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(5));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (6) + 4], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(5));
}

// gemm_RVV_5x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 5] @DRAM
// )
void gemm_RVV_5x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (6) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (6) + 5], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(5));
}

// gemm_RVV_5x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 5] @DRAM
// )
void gemm_RVV_5x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(5));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(5));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (6) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (6) + 5], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(5));
}

// gemm_RVV_6x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 6] @DRAM
// )
void gemm_RVV_6x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
}

// gemm_RVV_6x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 6] @DRAM
// )
void gemm_RVV_6x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
}

// gemm_RVV_6x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 6] @DRAM
// )
void gemm_RVV_6x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
}

// gemm_RVV_6x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 6] @DRAM
// )
void gemm_RVV_6x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
}

// gemm_RVV_6x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 6] @DRAM
// )
void gemm_RVV_6x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
}

// gemm_RVV_6x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 6] @DRAM
// )
void gemm_RVV_6x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
}

// gemm_RVV_6x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 6] @DRAM
// )
void gemm_RVV_6x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
}

// gemm_RVV_6x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 6] @DRAM
// )
void gemm_RVV_6x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
}

// gemm_RVV_6x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 6] @DRAM
// )
void gemm_RVV_6x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (6) + 4], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(6));
}

// gemm_RVV_6x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 6] @DRAM
// )
void gemm_RVV_6x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(6));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (6) + 4], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(6));
}

// gemm_RVV_6x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 6] @DRAM
// )
void gemm_RVV_6x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (6) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (6) + 5], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(6));
}

// gemm_RVV_6x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 6] @DRAM
// )
void gemm_RVV_6x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(6));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(6));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (6) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (6) + 5], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(6));
}

// gemm_RVV_7x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 7] @DRAM
// )
void gemm_RVV_7x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
}

// gemm_RVV_7x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 7] @DRAM
// )
void gemm_RVV_7x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
}

// gemm_RVV_7x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 7] @DRAM
// )
void gemm_RVV_7x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
}

// gemm_RVV_7x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 7] @DRAM
// )
void gemm_RVV_7x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
}

// gemm_RVV_7x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 7] @DRAM
// )
void gemm_RVV_7x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
}

// gemm_RVV_7x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 7] @DRAM
// )
void gemm_RVV_7x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
}

// gemm_RVV_7x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 7] @DRAM
// )
void gemm_RVV_7x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
}

// gemm_RVV_7x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 7] @DRAM
// )
void gemm_RVV_7x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
}

// gemm_RVV_7x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 7] @DRAM
// )
void gemm_RVV_7x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (6) + 4], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(7));
}

// gemm_RVV_7x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 7] @DRAM
// )
void gemm_RVV_7x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(7));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (6) + 4], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(7));
}

// gemm_RVV_7x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 7] @DRAM
// )
void gemm_RVV_7x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (6) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (6) + 5], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(7));
}

// gemm_RVV_7x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 7] @DRAM
// )
void gemm_RVV_7x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(7));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(7));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (6) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (6) + 5], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(7));
}

// gemm_RVV_8x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 8] @DRAM
// )
void gemm_RVV_8x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
}

// gemm_RVV_8x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 8] @DRAM
// )
void gemm_RVV_8x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
}

// gemm_RVV_8x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 8] @DRAM
// )
void gemm_RVV_8x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
}

// gemm_RVV_8x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 8] @DRAM
// )
void gemm_RVV_8x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
}

// gemm_RVV_8x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 8] @DRAM
// )
void gemm_RVV_8x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
}

// gemm_RVV_8x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 8] @DRAM
// )
void gemm_RVV_8x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
}

// gemm_RVV_8x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 8] @DRAM
// )
void gemm_RVV_8x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
}

// gemm_RVV_8x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 8] @DRAM
// )
void gemm_RVV_8x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
}

// gemm_RVV_8x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 8] @DRAM
// )
void gemm_RVV_8x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (6) + 4], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(8));
}

// gemm_RVV_8x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 8] @DRAM
// )
void gemm_RVV_8x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (6) + 4], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(8));
}

// gemm_RVV_8x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 8] @DRAM
// )
void gemm_RVV_8x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (6) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (6) + 5], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(8));
}

// gemm_RVV_8x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 8] @DRAM
// )
void gemm_RVV_8x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(8));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (6)], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (6) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (6) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (6) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (6) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (6) + 5], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(8));
}

// gemm_RVV_9x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 9] @DRAM
// )
void gemm_RVV_9x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(1));
}

// gemm_RVV_9x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 9] @DRAM
// )
void gemm_RVV_9x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(1));
}

// gemm_RVV_9x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 9] @DRAM
// )
void gemm_RVV_9x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(1));
}

// gemm_RVV_9x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 9] @DRAM
// )
void gemm_RVV_9x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(1));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(1));
}

// gemm_RVV_9x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 9] @DRAM
// )
void gemm_RVV_9x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(1));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(1));
}

// gemm_RVV_9x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 9] @DRAM
// )
void gemm_RVV_9x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(1));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(1));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(1));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(1));
}

// gemm_RVV_9x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 9] @DRAM
// )
void gemm_RVV_9x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(1));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(1));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(1));
}

// gemm_RVV_9x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 9] @DRAM
// )
void gemm_RVV_9x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(1));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(1));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(1));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 8],(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(1));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(1));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(1));
}

// gemm_RVV_9x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 9] @DRAM
// )
void gemm_RVV_9x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_regt_4;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_4_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_4 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (6) + 4], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(1));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(1));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(1));
  C_regt_4 = __riscv_vfmacc_vf_f16m1(C_regt_4, B[(k) * (6) + 4], A_regt,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 8], C_regt_4,(1));
}

// gemm_RVV_9x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 9] @DRAM
// )
void gemm_RVV_9x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_regt_4;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_4_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_4_0 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(1));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(1));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(1));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 8],(1));
C_regt_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc) + 8],(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (6) + 4], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(1));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(1));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(1));
  C_regt_4 = __riscv_vfmacc_vf_f16m1(C_regt_4, B[(k) * (6) + 4], A_regt,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 8], C_regt_4,(1));
}

// gemm_RVV_9x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 9] @DRAM
// )
void gemm_RVV_9x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_regt_4;
vfloat16m1_t C_regt_5;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_4_0;
vfloat16m1_t C_reg_5_0;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_5_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_4 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_5 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (6) + 4], A_reg_0,(8));
  C_reg_5_0 = __riscv_vfmacc_vf_f16m1(C_reg_5_0, B[(k) * (6) + 5], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(1));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(1));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(1));
  C_regt_4 = __riscv_vfmacc_vf_f16m1(C_regt_4, B[(k) * (6) + 4], A_regt,(1));
  C_regt_5 = __riscv_vfmacc_vf_f16m1(C_regt_5, B[(k) * (6) + 5], A_regt,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 8], C_regt_4,(1));
__riscv_vse16_v_f16m1(&C[(5) * (ldc) + 8], C_regt_5,(1));
}

// gemm_RVV_9x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 9] @DRAM
// )
void gemm_RVV_9x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_regt_4;
vfloat16m1_t C_regt_5;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_4_0;
vfloat16m1_t C_reg_5_0;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_4_0 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(8));
C_reg_5_0 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[8],(1));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(1));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(1));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 8],(1));
C_regt_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc) + 8],(1));
C_regt_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc) + 8],(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (16)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (16) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (6)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (6) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (6) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (6) + 3], A_reg_0,(8));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (6) + 4], A_reg_0,(8));
  C_reg_5_0 = __riscv_vfmacc_vf_f16m1(C_reg_5_0, B[(k) * (6) + 5], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (6)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (6) + 1], A_regt,(1));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (6) + 2], A_regt,(1));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (6) + 3], A_regt,(1));
  C_regt_4 = __riscv_vfmacc_vf_f16m1(C_regt_4, B[(k) * (6) + 4], A_regt,(1));
  C_regt_5 = __riscv_vfmacc_vf_f16m1(C_regt_5, B[(k) * (6) + 5], A_regt,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_regt_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_regt_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_regt_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 8], C_regt_4,(1));
__riscv_vse16_v_f16m1(&C[(5) * (ldc) + 8], C_regt_5,(1));
}


/* relying on the following instruction..."
rvv_broadcast_8xf16_0(dst,vl)
{dst_data} = __riscv_vfmv_v_f_f16m1(0.0f,{vl});
*/

/* relying on the following instruction..."
rvv_vfmacc_8xf16_1xf16(dst,lhs,rhs,vl)
{dst_data} = __riscv_vfmacc_vf_f16m1({dst_data}, {rhs_data}, {lhs_data},{vl});
*/

/* relying on the following instruction..."
rvv_vld_8xf16(dst,src,vl)
{dst_data} = __riscv_vle16_v_f16m1(&{src_data},{vl});
*/

/* relying on the following instruction..."
rvv_vst_8xf16(dst,src,vl)
__riscv_vse16_v_f16m1(&{dst_data}, {src_data},{vl});
*/
