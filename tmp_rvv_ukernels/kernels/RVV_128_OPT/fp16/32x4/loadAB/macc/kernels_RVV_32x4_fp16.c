#include "kernels_RVV_32x4_fp16.h"



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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(2));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(2));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(2));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(2));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(2));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(2));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(2));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(2));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(2));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(2));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(2));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(2));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(2));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(2));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(3));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(3));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(3));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(3));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(3));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(3));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(3));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(3));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(3));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(3));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(3));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(3));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(3));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(3));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(4));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(4));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(4));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(4));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(4));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(4));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(4));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(4));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(4));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(4));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(4));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(4));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(4));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(4));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(4));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(4));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(4));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(4));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(4));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(4));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(5));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(5));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(5));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(5));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(5));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(5));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(5));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(5));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(5));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(5));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(5));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(5));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(5));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(5));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(5));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(5));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(5));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(5));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(5));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(5));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(6));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(6));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(6));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(6));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(6));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(6));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(6));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(6));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(6));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(6));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(6));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(6));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(6));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(6));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(6));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(6));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(6));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(6));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(6));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(6));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(7));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(7));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(7));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(7));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(7));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(7));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(7));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(7));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(7));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(7));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(7));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(7));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(7));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(7));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(7));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(7));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(7));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(7));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(7));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(7));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (4) + 3], A_reg_1,(8));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (4) + 3], A_reg_1,(8));
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

// gemm_RVV_17x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 17] @DRAM
// )
void gemm_RVV_17x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(1));
}

// gemm_RVV_17x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 17] @DRAM
// )
void gemm_RVV_17x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(1));
}

// gemm_RVV_17x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 17] @DRAM
// )
void gemm_RVV_17x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(1));
}

// gemm_RVV_17x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 17] @DRAM
// )
void gemm_RVV_17x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(1));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(1));
}

// gemm_RVV_17x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 17] @DRAM
// )
void gemm_RVV_17x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(1));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(1));
}

// gemm_RVV_17x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 17] @DRAM
// )
void gemm_RVV_17x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(1));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(1));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(1));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(1));
}

// gemm_RVV_17x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 17] @DRAM
// )
void gemm_RVV_17x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (4) + 3], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(1));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(1));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_regt_3,(1));
}

// gemm_RVV_17x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 17] @DRAM
// )
void gemm_RVV_17x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(1));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(1));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(1));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (4) + 3], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(1));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(1));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_regt_3,(1));
}

// gemm_RVV_18x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 18] @DRAM
// )
void gemm_RVV_18x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(2));
}

// gemm_RVV_18x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 18] @DRAM
// )
void gemm_RVV_18x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(2));
}

// gemm_RVV_18x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 18] @DRAM
// )
void gemm_RVV_18x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(2));
}

// gemm_RVV_18x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 18] @DRAM
// )
void gemm_RVV_18x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(2));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(2));
}

// gemm_RVV_18x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 18] @DRAM
// )
void gemm_RVV_18x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(2));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(2));
}

// gemm_RVV_18x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 18] @DRAM
// )
void gemm_RVV_18x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(2));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(2));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(2));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(2));
}

// gemm_RVV_18x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 18] @DRAM
// )
void gemm_RVV_18x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (4) + 3], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(2));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(2));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_regt_3,(2));
}

// gemm_RVV_18x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 18] @DRAM
// )
void gemm_RVV_18x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(2));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(2));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(2));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (4) + 3], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(2));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(2));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_regt_3,(2));
}

// gemm_RVV_19x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 19] @DRAM
// )
void gemm_RVV_19x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(3));
}

// gemm_RVV_19x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 19] @DRAM
// )
void gemm_RVV_19x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(3));
}

// gemm_RVV_19x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 19] @DRAM
// )
void gemm_RVV_19x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(3));
}

// gemm_RVV_19x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 19] @DRAM
// )
void gemm_RVV_19x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(3));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(3));
}

// gemm_RVV_19x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 19] @DRAM
// )
void gemm_RVV_19x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(3));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(3));
}

// gemm_RVV_19x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 19] @DRAM
// )
void gemm_RVV_19x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(3));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(3));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(3));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(3));
}

// gemm_RVV_19x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 19] @DRAM
// )
void gemm_RVV_19x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (4) + 3], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(3));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(3));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_regt_3,(3));
}

// gemm_RVV_19x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 19] @DRAM
// )
void gemm_RVV_19x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(3));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(3));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(3));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (4) + 3], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(3));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(3));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_regt_3,(3));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(1));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(1));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(1));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(1));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (4) + 2], A_reg,(1));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (4) + 2], A_reg,(1));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (4) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (4) + 3], A_reg,(1));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (4) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (4) + 3], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
}

// gemm_RVV_20x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 20] @DRAM
// )
void gemm_RVV_20x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(4));
}

// gemm_RVV_20x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 20] @DRAM
// )
void gemm_RVV_20x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(4));
}

// gemm_RVV_20x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 20] @DRAM
// )
void gemm_RVV_20x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(4));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(4));
}

// gemm_RVV_20x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 20] @DRAM
// )
void gemm_RVV_20x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(4));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(4));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(4));
}

// gemm_RVV_20x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 20] @DRAM
// )
void gemm_RVV_20x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(4));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(4));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(4));
}

// gemm_RVV_20x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 20] @DRAM
// )
void gemm_RVV_20x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(4));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(4));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(4));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(4));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(4));
}

// gemm_RVV_20x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 20] @DRAM
// )
void gemm_RVV_20x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (4) + 3], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(4));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(4));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(4));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_regt_3,(4));
}

// gemm_RVV_20x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 20] @DRAM
// )
void gemm_RVV_20x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(4));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(4));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(4));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (4) + 3], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(4));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(4));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(4));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_regt_3,(4));
}

// gemm_RVV_21x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 21] @DRAM
// )
void gemm_RVV_21x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(5));
}

// gemm_RVV_21x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 21] @DRAM
// )
void gemm_RVV_21x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(5));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(5));
}

// gemm_RVV_21x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 21] @DRAM
// )
void gemm_RVV_21x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(5));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(5));
}

// gemm_RVV_21x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 21] @DRAM
// )
void gemm_RVV_21x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(5));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(5));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(5));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(5));
}

// gemm_RVV_21x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 21] @DRAM
// )
void gemm_RVV_21x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(5));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(5));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(5));
}

// gemm_RVV_21x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 21] @DRAM
// )
void gemm_RVV_21x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(5));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(5));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(5));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(5));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(5));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(5));
}

// gemm_RVV_21x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 21] @DRAM
// )
void gemm_RVV_21x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (4) + 3], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(5));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(5));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(5));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_regt_3,(5));
}

// gemm_RVV_21x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 21] @DRAM
// )
void gemm_RVV_21x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(5));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(5));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(5));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(5));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (4) + 3], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(5));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(5));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(5));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_regt_3,(5));
}

// gemm_RVV_22x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 22] @DRAM
// )
void gemm_RVV_22x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(6));
}

// gemm_RVV_22x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 22] @DRAM
// )
void gemm_RVV_22x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(6));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(6));
}

// gemm_RVV_22x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 22] @DRAM
// )
void gemm_RVV_22x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(6));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(6));
}

// gemm_RVV_22x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 22] @DRAM
// )
void gemm_RVV_22x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(6));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(6));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(6));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(6));
}

// gemm_RVV_22x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 22] @DRAM
// )
void gemm_RVV_22x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(6));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(6));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(6));
}

// gemm_RVV_22x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 22] @DRAM
// )
void gemm_RVV_22x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(6));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(6));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(6));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(6));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(6));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(6));
}

// gemm_RVV_22x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 22] @DRAM
// )
void gemm_RVV_22x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (4) + 3], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(6));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(6));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(6));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_regt_3,(6));
}

// gemm_RVV_22x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 22] @DRAM
// )
void gemm_RVV_22x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(6));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(6));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(6));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(6));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (4) + 3], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(6));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(6));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(6));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_regt_3,(6));
}

// gemm_RVV_23x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 23] @DRAM
// )
void gemm_RVV_23x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(7));
}

// gemm_RVV_23x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 23] @DRAM
// )
void gemm_RVV_23x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(7));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(7));
}

// gemm_RVV_23x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 23] @DRAM
// )
void gemm_RVV_23x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(7));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(7));
}

// gemm_RVV_23x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 23] @DRAM
// )
void gemm_RVV_23x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(7));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(7));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(7));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(7));
}

// gemm_RVV_23x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 23] @DRAM
// )
void gemm_RVV_23x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(7));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(7));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(7));
}

// gemm_RVV_23x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 23] @DRAM
// )
void gemm_RVV_23x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(7));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(7));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(7));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(7));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(7));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(7));
}

// gemm_RVV_23x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 23] @DRAM
// )
void gemm_RVV_23x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (4) + 3], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(7));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(7));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(7));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_regt_3,(7));
}

// gemm_RVV_23x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 23] @DRAM
// )
void gemm_RVV_23x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(7));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(7));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(7));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(7));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (4) + 3], A_reg_1,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(7));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(7));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(7));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_regt_3,(7));
}

// gemm_RVV_24x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 24] @DRAM
// )
void gemm_RVV_24x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
}

// gemm_RVV_24x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 24] @DRAM
// )
void gemm_RVV_24x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
}

// gemm_RVV_24x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 24] @DRAM
// )
void gemm_RVV_24x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
}

// gemm_RVV_24x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 24] @DRAM
// )
void gemm_RVV_24x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C[ldc + 16],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
}

// gemm_RVV_24x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 24] @DRAM
// )
void gemm_RVV_24x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
}

// gemm_RVV_24x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 24] @DRAM
// )
void gemm_RVV_24x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C[ldc + 16],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(8));
C_reg_2_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
}

// gemm_RVV_24x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 24] @DRAM
// )
void gemm_RVV_24x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_3_1;
vfloat16m1_t C_reg_3_2;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (4) + 3], A_reg_1,(8));
  C_reg_3_2 = __riscv_vfmacc_vf_f16m1(C_reg_3_2, B[(k) * (4) + 3], A_reg_2,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_2,(8));
}

// gemm_RVV_24x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 24] @DRAM
// )
void gemm_RVV_24x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_3_1;
vfloat16m1_t C_reg_3_2;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C[ldc + 16],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(8));
C_reg_2_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(8));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_3_1 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 8],(8));
C_reg_3_2 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (4) + 3], A_reg_1,(8));
  C_reg_3_2 = __riscv_vfmacc_vf_f16m1(C_reg_3_2, B[(k) * (4) + 3], A_reg_2,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_2,(8));
}

// gemm_RVV_25x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 25] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 25] @DRAM
// )
void gemm_RVV_25x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(1));
}

// gemm_RVV_25x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 25] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 25] @DRAM
// )
void gemm_RVV_25x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[24],(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(1));
}

// gemm_RVV_25x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 25] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 25] @DRAM
// )
void gemm_RVV_25x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(1));
}

// gemm_RVV_25x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 25] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 25] @DRAM
// )
void gemm_RVV_25x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C[ldc + 16],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[24],(1));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 24],(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(1));
}

// gemm_RVV_25x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 25] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 25] @DRAM
// )
void gemm_RVV_25x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(1));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 24], C_regt_2,(1));
}

// gemm_RVV_25x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 25] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 25] @DRAM
// )
void gemm_RVV_25x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C[ldc + 16],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(8));
C_reg_2_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[24],(1));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 24],(1));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 24],(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(1));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 24], C_regt_2,(1));
}

// gemm_RVV_25x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 25] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 25] @DRAM
// )
void gemm_RVV_25x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_3_1;
vfloat16m1_t C_reg_3_2;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (4) + 3], A_reg_1,(8));
  C_reg_3_2 = __riscv_vfmacc_vf_f16m1(C_reg_3_2, B[(k) * (4) + 3], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(1));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(1));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 24], C_regt_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 24], C_regt_3,(1));
}

// gemm_RVV_25x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 25] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 25] @DRAM
// )
void gemm_RVV_25x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_3_1;
vfloat16m1_t C_reg_3_2;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C[ldc + 16],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(8));
C_reg_2_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(8));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_3_1 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 8],(8));
C_reg_3_2 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[24],(1));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 24],(1));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 24],(1));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 24],(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (4) + 3], A_reg_1,(8));
  C_reg_3_2 = __riscv_vfmacc_vf_f16m1(C_reg_3_2, B[(k) * (4) + 3], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(1));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(1));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 24], C_regt_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 24], C_regt_3,(1));
}

// gemm_RVV_26x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 26] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 26] @DRAM
// )
void gemm_RVV_26x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(2));
}

// gemm_RVV_26x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 26] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 26] @DRAM
// )
void gemm_RVV_26x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[24],(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(2));
}

// gemm_RVV_26x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 26] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 26] @DRAM
// )
void gemm_RVV_26x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(2));
}

// gemm_RVV_26x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 26] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 26] @DRAM
// )
void gemm_RVV_26x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C[ldc + 16],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[24],(2));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 24],(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(2));
}

// gemm_RVV_26x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 26] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 26] @DRAM
// )
void gemm_RVV_26x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(2));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 24], C_regt_2,(2));
}

// gemm_RVV_26x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 26] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 26] @DRAM
// )
void gemm_RVV_26x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C[ldc + 16],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(8));
C_reg_2_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[24],(2));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 24],(2));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 24],(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(2));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 24], C_regt_2,(2));
}

// gemm_RVV_26x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 26] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 26] @DRAM
// )
void gemm_RVV_26x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_3_1;
vfloat16m1_t C_reg_3_2;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (4) + 3], A_reg_1,(8));
  C_reg_3_2 = __riscv_vfmacc_vf_f16m1(C_reg_3_2, B[(k) * (4) + 3], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(2));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(2));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 24], C_regt_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 24], C_regt_3,(2));
}

// gemm_RVV_26x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 26] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 26] @DRAM
// )
void gemm_RVV_26x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_3_1;
vfloat16m1_t C_reg_3_2;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C[ldc + 16],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(8));
C_reg_2_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(8));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_3_1 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 8],(8));
C_reg_3_2 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[24],(2));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 24],(2));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 24],(2));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 24],(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (4) + 3], A_reg_1,(8));
  C_reg_3_2 = __riscv_vfmacc_vf_f16m1(C_reg_3_2, B[(k) * (4) + 3], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(2));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(2));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 24], C_regt_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 24], C_regt_3,(2));
}

// gemm_RVV_27x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 27] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 27] @DRAM
// )
void gemm_RVV_27x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(3));
}

// gemm_RVV_27x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 27] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 27] @DRAM
// )
void gemm_RVV_27x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[24],(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(3));
}

// gemm_RVV_27x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 27] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 27] @DRAM
// )
void gemm_RVV_27x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(3));
}

// gemm_RVV_27x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 27] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 27] @DRAM
// )
void gemm_RVV_27x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C[ldc + 16],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[24],(3));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 24],(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(3));
}

// gemm_RVV_27x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 27] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 27] @DRAM
// )
void gemm_RVV_27x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(3));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 24], C_regt_2,(3));
}

// gemm_RVV_27x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 27] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 27] @DRAM
// )
void gemm_RVV_27x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C[ldc + 16],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(8));
C_reg_2_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[24],(3));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 24],(3));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 24],(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(3));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 24], C_regt_2,(3));
}

// gemm_RVV_27x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 27] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 27] @DRAM
// )
void gemm_RVV_27x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_3_1;
vfloat16m1_t C_reg_3_2;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (4) + 3], A_reg_1,(8));
  C_reg_3_2 = __riscv_vfmacc_vf_f16m1(C_reg_3_2, B[(k) * (4) + 3], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(3));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(3));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 24], C_regt_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 24], C_regt_3,(3));
}

// gemm_RVV_27x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 27] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 27] @DRAM
// )
void gemm_RVV_27x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_3_1;
vfloat16m1_t C_reg_3_2;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C[ldc + 16],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(8));
C_reg_2_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(8));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_3_1 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 8],(8));
C_reg_3_2 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[24],(3));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 24],(3));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 24],(3));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 24],(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (4) + 3], A_reg_1,(8));
  C_reg_3_2 = __riscv_vfmacc_vf_f16m1(C_reg_3_2, B[(k) * (4) + 3], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(3));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(3));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 24], C_regt_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 24], C_regt_3,(3));
}

// gemm_RVV_28x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 28] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 28] @DRAM
// )
void gemm_RVV_28x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(4));
}

// gemm_RVV_28x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 28] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 28] @DRAM
// )
void gemm_RVV_28x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[24],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(4));
}

// gemm_RVV_28x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 28] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 28] @DRAM
// )
void gemm_RVV_28x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(4));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(4));
}

// gemm_RVV_28x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 28] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 28] @DRAM
// )
void gemm_RVV_28x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C[ldc + 16],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[24],(4));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 24],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(4));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(4));
}

// gemm_RVV_28x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 28] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 28] @DRAM
// )
void gemm_RVV_28x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(4));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(4));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 24], C_regt_2,(4));
}

// gemm_RVV_28x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 28] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 28] @DRAM
// )
void gemm_RVV_28x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C[ldc + 16],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(8));
C_reg_2_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[24],(4));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 24],(4));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 24],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(4));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(4));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 24], C_regt_2,(4));
}

// gemm_RVV_28x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 28] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 28] @DRAM
// )
void gemm_RVV_28x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_3_1;
vfloat16m1_t C_reg_3_2;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (4) + 3], A_reg_1,(8));
  C_reg_3_2 = __riscv_vfmacc_vf_f16m1(C_reg_3_2, B[(k) * (4) + 3], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(4));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(4));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(4));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 24], C_regt_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 24], C_regt_3,(4));
}

// gemm_RVV_28x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 28] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 28] @DRAM
// )
void gemm_RVV_28x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_3_1;
vfloat16m1_t C_reg_3_2;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C[ldc + 16],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(8));
C_reg_2_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(8));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_3_1 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 8],(8));
C_reg_3_2 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[24],(4));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 24],(4));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 24],(4));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 24],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (4) + 3], A_reg_1,(8));
  C_reg_3_2 = __riscv_vfmacc_vf_f16m1(C_reg_3_2, B[(k) * (4) + 3], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(4));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(4));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(4));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 24], C_regt_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 24], C_regt_3,(4));
}

// gemm_RVV_29x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 29] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 29] @DRAM
// )
void gemm_RVV_29x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(5));
}

// gemm_RVV_29x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 29] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 29] @DRAM
// )
void gemm_RVV_29x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[24],(5));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(5));
}

// gemm_RVV_29x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 29] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 29] @DRAM
// )
void gemm_RVV_29x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(5));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(5));
}

// gemm_RVV_29x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 29] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 29] @DRAM
// )
void gemm_RVV_29x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C[ldc + 16],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[24],(5));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 24],(5));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(5));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(5));
}

// gemm_RVV_29x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 29] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 29] @DRAM
// )
void gemm_RVV_29x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(5));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(5));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 24], C_regt_2,(5));
}

// gemm_RVV_29x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 29] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 29] @DRAM
// )
void gemm_RVV_29x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C[ldc + 16],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(8));
C_reg_2_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[24],(5));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 24],(5));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 24],(5));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(5));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(5));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 24], C_regt_2,(5));
}

// gemm_RVV_29x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 29] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 29] @DRAM
// )
void gemm_RVV_29x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_3_1;
vfloat16m1_t C_reg_3_2;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (4) + 3], A_reg_1,(8));
  C_reg_3_2 = __riscv_vfmacc_vf_f16m1(C_reg_3_2, B[(k) * (4) + 3], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(5));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(5));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(5));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 24], C_regt_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 24], C_regt_3,(5));
}

// gemm_RVV_29x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 29] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 29] @DRAM
// )
void gemm_RVV_29x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_3_1;
vfloat16m1_t C_reg_3_2;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C[ldc + 16],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(8));
C_reg_2_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(8));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_3_1 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 8],(8));
C_reg_3_2 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[24],(5));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 24],(5));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 24],(5));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 24],(5));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (4) + 3], A_reg_1,(8));
  C_reg_3_2 = __riscv_vfmacc_vf_f16m1(C_reg_3_2, B[(k) * (4) + 3], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(5));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(5));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(5));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 24], C_regt_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 24], C_regt_3,(5));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(2));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(2));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(2));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(2));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (4) + 2], A_reg,(2));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (4) + 2], A_reg,(2));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (4) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (4) + 3], A_reg,(2));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (4) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (4) + 3], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
}

// gemm_RVV_30x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 30] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 30] @DRAM
// )
void gemm_RVV_30x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(6));
}

// gemm_RVV_30x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 30] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 30] @DRAM
// )
void gemm_RVV_30x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[24],(6));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(6));
}

// gemm_RVV_30x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 30] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 30] @DRAM
// )
void gemm_RVV_30x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(6));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(6));
}

// gemm_RVV_30x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 30] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 30] @DRAM
// )
void gemm_RVV_30x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C[ldc + 16],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[24],(6));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 24],(6));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(6));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(6));
}

// gemm_RVV_30x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 30] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 30] @DRAM
// )
void gemm_RVV_30x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(6));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(6));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 24], C_regt_2,(6));
}

// gemm_RVV_30x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 30] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 30] @DRAM
// )
void gemm_RVV_30x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C[ldc + 16],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(8));
C_reg_2_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[24],(6));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 24],(6));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 24],(6));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(6));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(6));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 24], C_regt_2,(6));
}

// gemm_RVV_30x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 30] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 30] @DRAM
// )
void gemm_RVV_30x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_3_1;
vfloat16m1_t C_reg_3_2;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (4) + 3], A_reg_1,(8));
  C_reg_3_2 = __riscv_vfmacc_vf_f16m1(C_reg_3_2, B[(k) * (4) + 3], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(6));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(6));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(6));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 24], C_regt_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 24], C_regt_3,(6));
}

// gemm_RVV_30x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 30] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 30] @DRAM
// )
void gemm_RVV_30x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_3_1;
vfloat16m1_t C_reg_3_2;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C[ldc + 16],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(8));
C_reg_2_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(8));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_3_1 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 8],(8));
C_reg_3_2 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[24],(6));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 24],(6));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 24],(6));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 24],(6));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (4) + 3], A_reg_1,(8));
  C_reg_3_2 = __riscv_vfmacc_vf_f16m1(C_reg_3_2, B[(k) * (4) + 3], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(6));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(6));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(6));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 24], C_regt_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 24], C_regt_3,(6));
}

// gemm_RVV_31x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 31] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 31] @DRAM
// )
void gemm_RVV_31x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(7));
}

// gemm_RVV_31x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 31] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 31] @DRAM
// )
void gemm_RVV_31x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[24],(7));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(7));
}

// gemm_RVV_31x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 31] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 31] @DRAM
// )
void gemm_RVV_31x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(7));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(7));
}

// gemm_RVV_31x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 31] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 31] @DRAM
// )
void gemm_RVV_31x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C[ldc + 16],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[24],(7));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 24],(7));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(7));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(7));
}

// gemm_RVV_31x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 31] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 31] @DRAM
// )
void gemm_RVV_31x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(7));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(7));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 24], C_regt_2,(7));
}

// gemm_RVV_31x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 31] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 31] @DRAM
// )
void gemm_RVV_31x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C[ldc + 16],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(8));
C_reg_2_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[24],(7));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 24],(7));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 24],(7));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(7));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(7));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 24], C_regt_2,(7));
}

// gemm_RVV_31x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 31] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 31] @DRAM
// )
void gemm_RVV_31x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_3_1;
vfloat16m1_t C_reg_3_2;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (4) + 3], A_reg_1,(8));
  C_reg_3_2 = __riscv_vfmacc_vf_f16m1(C_reg_3_2, B[(k) * (4) + 3], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(7));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(7));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(7));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 24], C_regt_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 24], C_regt_3,(7));
}

// gemm_RVV_31x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 31] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 31] @DRAM
// )
void gemm_RVV_31x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_regt;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_3_1;
vfloat16m1_t C_reg_3_2;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C[ldc + 16],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(8));
C_reg_2_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(8));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_3_1 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 8],(8));
C_reg_3_2 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(8));
C_regt_0 = __riscv_vle16_v_f16m1(&C[24],(7));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 24],(7));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 24],(7));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 24],(7));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (4) + 3], A_reg_1,(8));
  C_reg_3_2 = __riscv_vfmacc_vf_f16m1(C_reg_3_2, B[(k) * (4) + 3], A_reg_2,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(7));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(7));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(7));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_regt_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 24], C_regt_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 24], C_regt_3,(7));
}

// gemm_RVV_32x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 32] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 32] @DRAM
// )
void gemm_RVV_32x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_0_3 = __riscv_vfmacc_vf_f16m1(C_reg_0_3, B[(k) * (4)], A_reg_3,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_reg_0_3,(8));
}

// gemm_RVV_32x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 32] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 32] @DRAM
// )
void gemm_RVV_32x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C[24],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_0_3 = __riscv_vfmacc_vf_f16m1(C_reg_0_3, B[(k) * (4)], A_reg_3,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_reg_0_3,(8));
}

// gemm_RVV_32x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 32] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 32] @DRAM
// )
void gemm_RVV_32x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_0_3 = __riscv_vfmacc_vf_f16m1(C_reg_0_3, B[(k) * (4)], A_reg_3,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_1_3 = __riscv_vfmacc_vf_f16m1(C_reg_1_3, B[(k) * (4) + 1], A_reg_3,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_reg_1_3,(8));
}

// gemm_RVV_32x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 32] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 32] @DRAM
// )
void gemm_RVV_32x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C[24],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C[ldc + 16],(8));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C[ldc + 24],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_0_3 = __riscv_vfmacc_vf_f16m1(C_reg_0_3, B[(k) * (4)], A_reg_3,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_1_3 = __riscv_vfmacc_vf_f16m1(C_reg_1_3, B[(k) * (4) + 1], A_reg_3,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_reg_1_3,(8));
}

// gemm_RVV_32x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 32] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 32] @DRAM
// )
void gemm_RVV_32x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
vfloat16m1_t C_reg_2_3;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_0_3 = __riscv_vfmacc_vf_f16m1(C_reg_0_3, B[(k) * (4)], A_reg_3,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_1_3 = __riscv_vfmacc_vf_f16m1(C_reg_1_3, B[(k) * (4) + 1], A_reg_3,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
  C_reg_2_3 = __riscv_vfmacc_vf_f16m1(C_reg_2_3, B[(k) * (4) + 2], A_reg_3,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 24], C_reg_2_3,(8));
}

// gemm_RVV_32x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 32] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 32] @DRAM
// )
void gemm_RVV_32x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
vfloat16m1_t C_reg_2_3;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C[24],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C[ldc + 16],(8));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C[ldc + 24],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(8));
C_reg_2_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(8));
C_reg_2_3 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 24],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_0_3 = __riscv_vfmacc_vf_f16m1(C_reg_0_3, B[(k) * (4)], A_reg_3,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_1_3 = __riscv_vfmacc_vf_f16m1(C_reg_1_3, B[(k) * (4) + 1], A_reg_3,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
  C_reg_2_3 = __riscv_vfmacc_vf_f16m1(C_reg_2_3, B[(k) * (4) + 2], A_reg_3,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 24], C_reg_2_3,(8));
}

// gemm_RVV_32x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 32] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 32] @DRAM
// )
void gemm_RVV_32x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
vfloat16m1_t C_reg_2_3;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_3_1;
vfloat16m1_t C_reg_3_2;
vfloat16m1_t C_reg_3_3;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_0_3 = __riscv_vfmacc_vf_f16m1(C_reg_0_3, B[(k) * (4)], A_reg_3,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_1_3 = __riscv_vfmacc_vf_f16m1(C_reg_1_3, B[(k) * (4) + 1], A_reg_3,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
  C_reg_2_3 = __riscv_vfmacc_vf_f16m1(C_reg_2_3, B[(k) * (4) + 2], A_reg_3,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (4) + 3], A_reg_1,(8));
  C_reg_3_2 = __riscv_vfmacc_vf_f16m1(C_reg_3_2, B[(k) * (4) + 3], A_reg_2,(8));
  C_reg_3_3 = __riscv_vfmacc_vf_f16m1(C_reg_3_3, B[(k) * (4) + 3], A_reg_3,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 24], C_reg_2_3,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 24], C_reg_3_3,(8));
}

// gemm_RVV_32x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 32] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 32] @DRAM
// )
void gemm_RVV_32x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_0_3;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_1_3;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
vfloat16m1_t C_reg_2_3;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_3_1;
vfloat16m1_t C_reg_3_2;
vfloat16m1_t C_reg_3_3;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[8],(8));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[16],(8));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C[24],(8));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 8],(8));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C[ldc + 16],(8));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C[ldc + 24],(8));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 8],(8));
C_reg_2_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(8));
C_reg_2_3 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 24],(8));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_3_1 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 8],(8));
C_reg_3_2 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(8));
C_reg_3_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 24],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(8));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 16],(8));
  A_reg_3 = __riscv_vle16_v_f16m1(&A[(k) * (32) + 24],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (4)], A_reg_1,(8));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (4)], A_reg_2,(8));
  C_reg_0_3 = __riscv_vfmacc_vf_f16m1(C_reg_0_3, B[(k) * (4)], A_reg_3,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (4) + 1], A_reg_1,(8));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (4) + 1], A_reg_2,(8));
  C_reg_1_3 = __riscv_vfmacc_vf_f16m1(C_reg_1_3, B[(k) * (4) + 1], A_reg_3,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (4) + 2], A_reg_1,(8));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (4) + 2], A_reg_2,(8));
  C_reg_2_3 = __riscv_vfmacc_vf_f16m1(C_reg_2_3, B[(k) * (4) + 2], A_reg_3,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (4) + 3], A_reg_1,(8));
  C_reg_3_2 = __riscv_vfmacc_vf_f16m1(C_reg_3_2, B[(k) * (4) + 3], A_reg_2,(8));
  C_reg_3_3 = __riscv_vfmacc_vf_f16m1(C_reg_3_3, B[(k) * (4) + 3], A_reg_3,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(8));
__riscv_vse16_v_f16m1(&C[8], C_reg_0_1,(8));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_2,(8));
__riscv_vse16_v_f16m1(&C[24], C_reg_0_3,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 8], C_reg_1_1,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_2,(8));
__riscv_vse16_v_f16m1(&C[ldc + 24], C_reg_1_3,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 8], C_reg_2_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_2,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 24], C_reg_2_3,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 8], C_reg_3_1,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 24], C_reg_3_3,(8));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(3));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(3));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(3));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(3));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (4) + 2], A_reg,(3));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (4) + 2], A_reg,(3));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (4) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (4) + 3], A_reg,(3));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (4) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (4) + 3], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(4));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(4));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(4));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(4));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (4) + 2], A_reg,(4));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (4) + 2], A_reg,(4));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (4) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (4) + 3], A_reg,(4));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (4) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (4) + 3], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(5));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(5));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(5));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(5));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (4) + 2], A_reg,(5));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (4) + 2], A_reg,(5));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (4) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (4) + 3], A_reg,(5));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (4) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (4) + 3], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(6));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(6));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(6));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(6));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (4) + 2], A_reg,(6));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (4) + 2], A_reg,(6));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (4) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (4) + 3], A_reg,(6));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (4) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (4) + 3], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(7));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(7));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(7));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(7));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (4) + 2], A_reg,(7));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (4) + 2], A_reg,(7));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (4) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (4) + 3], A_reg,(7));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (4) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (4) + 3], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(8));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(8));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(8));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(8));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (4) + 2], A_reg,(8));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (4) + 2], A_reg,(8));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (4) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (4) + 3], A_reg,(8));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (4)], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (4) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (4) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (4) + 3], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(1));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(1));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(1));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(1));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(1));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(1));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(1));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(1));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(1));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(1));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(1));
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
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (32)],(8));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (32) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (4)], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (4) + 1], A_reg_0,(8));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (4) + 2], A_reg_0,(8));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (4) + 3], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f16m1(C_regt_0, B[(k) * (4)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f16m1(C_regt_1, B[(k) * (4) + 1], A_regt,(1));
  C_regt_2 = __riscv_vfmacc_vf_f16m1(C_regt_2, B[(k) * (4) + 2], A_regt,(1));
  C_regt_3 = __riscv_vfmacc_vf_f16m1(C_regt_3, B[(k) * (4) + 3], A_regt,(1));
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
