#include "kernels_RVV_8x8_fp16.h"



#include <stdio.h>
#include <stdlib.h>

#include <riscv_vector.h>


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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(1));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(1));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(1));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(1));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(1));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(1));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(1));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(1));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(1));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(1));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(1));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(1));
}

// gemm_RVV_1x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 1] @DRAM
// )
void gemm_RVV_1x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(1));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B[(k) * (8) + 6], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(1));
}

// gemm_RVV_1x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 1] @DRAM
// )
void gemm_RVV_1x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(1));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(1));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(1));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B[(k) * (8) + 6], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(1));
}

// gemm_RVV_1x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 1] @DRAM
// )
void gemm_RVV_1x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(1));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B[(k) * (8) + 6], A_reg,(1));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B[(k) * (8) + 7], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(1));
}

// gemm_RVV_1x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 1] @DRAM
// )
void gemm_RVV_1x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(1));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(1));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(1));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(1));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(1));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(1));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B[(k) * (8) + 6], A_reg,(1));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B[(k) * (8) + 7], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(1));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(2));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(2));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(2));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(2));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(2));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(2));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(2));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(2));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(2));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(2));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(2));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(2));
}

// gemm_RVV_2x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 2] @DRAM
// )
void gemm_RVV_2x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(2));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B[(k) * (8) + 6], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(2));
}

// gemm_RVV_2x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 2] @DRAM
// )
void gemm_RVV_2x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(2));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(2));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(2));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B[(k) * (8) + 6], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(2));
}

// gemm_RVV_2x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 2] @DRAM
// )
void gemm_RVV_2x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(2));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B[(k) * (8) + 6], A_reg,(2));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B[(k) * (8) + 7], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(2));
}

// gemm_RVV_2x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 2] @DRAM
// )
void gemm_RVV_2x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(2));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(2));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(2));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(2));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(2));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(2));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B[(k) * (8) + 6], A_reg,(2));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B[(k) * (8) + 7], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(2));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(3));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(3));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(3));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(3));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(3));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(3));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(3));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(3));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(3));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(3));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(3));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(3));
}

// gemm_RVV_3x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 3] @DRAM
// )
void gemm_RVV_3x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(3));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B[(k) * (8) + 6], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(3));
}

// gemm_RVV_3x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 3] @DRAM
// )
void gemm_RVV_3x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(3));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(3));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(3));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B[(k) * (8) + 6], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(3));
}

// gemm_RVV_3x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 3] @DRAM
// )
void gemm_RVV_3x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(3));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B[(k) * (8) + 6], A_reg,(3));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B[(k) * (8) + 7], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(3));
}

// gemm_RVV_3x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 3] @DRAM
// )
void gemm_RVV_3x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(3));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(3));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(3));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(3));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(3));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(3));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B[(k) * (8) + 6], A_reg,(3));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B[(k) * (8) + 7], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(3));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(4));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(4));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(4));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(4));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(4));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(4));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(4));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(4));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(4));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(4));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(4));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(4));
}

// gemm_RVV_4x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 4] @DRAM
// )
void gemm_RVV_4x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(4));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B[(k) * (8) + 6], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(4));
}

// gemm_RVV_4x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 4] @DRAM
// )
void gemm_RVV_4x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(4));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(4));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(4));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B[(k) * (8) + 6], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(4));
}

// gemm_RVV_4x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 4] @DRAM
// )
void gemm_RVV_4x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(4));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B[(k) * (8) + 6], A_reg,(4));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B[(k) * (8) + 7], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(4));
}

// gemm_RVV_4x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 4] @DRAM
// )
void gemm_RVV_4x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(4));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(4));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(4));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(4));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(4));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(4));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B[(k) * (8) + 6], A_reg,(4));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B[(k) * (8) + 7], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(4));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(5));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(5));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(5));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(5));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(5));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(5));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(5));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(5));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(5));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(5));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(5));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(5));
}

// gemm_RVV_5x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 5] @DRAM
// )
void gemm_RVV_5x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(5));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B[(k) * (8) + 6], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(5));
}

// gemm_RVV_5x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 5] @DRAM
// )
void gemm_RVV_5x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(5));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(5));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(5));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(5));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B[(k) * (8) + 6], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(5));
}

// gemm_RVV_5x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 5] @DRAM
// )
void gemm_RVV_5x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(5));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B[(k) * (8) + 6], A_reg,(5));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B[(k) * (8) + 7], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(5));
}

// gemm_RVV_5x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 5] @DRAM
// )
void gemm_RVV_5x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(5));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(5));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(5));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(5));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(5));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(5));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(5));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B[(k) * (8) + 6], A_reg,(5));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B[(k) * (8) + 7], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(5));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(6));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(6));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(6));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(6));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(6));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(6));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(6));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(6));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(6));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(6));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(6));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(6));
}

// gemm_RVV_6x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 6] @DRAM
// )
void gemm_RVV_6x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(6));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B[(k) * (8) + 6], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(6));
}

// gemm_RVV_6x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 6] @DRAM
// )
void gemm_RVV_6x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(6));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(6));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(6));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(6));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B[(k) * (8) + 6], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(6));
}

// gemm_RVV_6x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 6] @DRAM
// )
void gemm_RVV_6x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(6));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B[(k) * (8) + 6], A_reg,(6));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B[(k) * (8) + 7], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(6));
}

// gemm_RVV_6x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 6] @DRAM
// )
void gemm_RVV_6x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(6));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(6));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(6));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(6));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(6));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(6));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(6));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B[(k) * (8) + 6], A_reg,(6));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B[(k) * (8) + 7], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(6));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(7));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(7));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(7));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(7));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(7));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(7));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(7));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(7));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(7));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(7));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(7));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(7));
}

// gemm_RVV_7x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 7] @DRAM
// )
void gemm_RVV_7x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(7));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B[(k) * (8) + 6], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(7));
}

// gemm_RVV_7x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 7] @DRAM
// )
void gemm_RVV_7x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(7));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(7));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(7));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(7));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B[(k) * (8) + 6], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(7));
}

// gemm_RVV_7x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 7] @DRAM
// )
void gemm_RVV_7x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(7));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B[(k) * (8) + 6], A_reg,(7));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B[(k) * (8) + 7], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(7));
}

// gemm_RVV_7x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 7] @DRAM
// )
void gemm_RVV_7x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(7));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(7));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(7));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(7));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(7));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(7));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(7));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B[(k) * (8) + 6], A_reg,(7));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B[(k) * (8) + 7], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(7));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(8));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(8));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(8));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(8));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(8));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(8));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(8));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(8));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(8));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(8));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(8));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(8));
}

// gemm_RVV_8x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 8] @DRAM
// )
void gemm_RVV_8x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(8));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B[(k) * (8) + 6], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(8));
}

// gemm_RVV_8x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 8] @DRAM
// )
void gemm_RVV_8x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(8));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(8));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(8));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B[(k) * (8) + 6], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(8));
}

// gemm_RVV_8x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 8] @DRAM
// )
void gemm_RVV_8x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(8));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B[(k) * (8) + 6], A_reg,(8));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B[(k) * (8) + 7], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(8));
}

// gemm_RVV_8x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 8] @DRAM
// )
void gemm_RVV_8x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(8));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(8));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(8));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B[(k) * (8)], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B[(k) * (8) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B[(k) * (8) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B[(k) * (8) + 3], A_reg,(8));
  C_reg_4 = __riscv_vfmacc_vf_f16m1(C_reg_4, B[(k) * (8) + 4], A_reg,(8));
  C_reg_5 = __riscv_vfmacc_vf_f16m1(C_reg_5, B[(k) * (8) + 5], A_reg,(8));
  C_reg_6 = __riscv_vfmacc_vf_f16m1(C_reg_6, B[(k) * (8) + 6], A_reg,(8));
  C_reg_7 = __riscv_vfmacc_vf_f16m1(C_reg_7, B[(k) * (8) + 7], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(8));
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
