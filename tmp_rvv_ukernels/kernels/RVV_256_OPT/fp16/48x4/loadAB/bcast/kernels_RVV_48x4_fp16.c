#include "kernels_RVV_48x4_fp16.h"



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
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(10));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(10));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(10));
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
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(10));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(10));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(10));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(10));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(10));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(10));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(10));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(10));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(10));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(10));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(10));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(10));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(10));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(10));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(10));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(10));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(10));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(10));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(10));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(10));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(10));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(10));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(10));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(10));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(10));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(10));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(10));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(10));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(10));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(10));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(10));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(10));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(10));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(10));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(10));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(10));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(10));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(10));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(10));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(10));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(10));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(10));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(10));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(10));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(10));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(10));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(10));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(10));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(10));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(10));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(10));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(10));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(10));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(10));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(10));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(10));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(10));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(10));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(10));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(10));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(10));
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
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(11));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(11));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(11));
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
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(11));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(11));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(11));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(11));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(11));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(11));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(11));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(11));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(11));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(11));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(11));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(11));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(11));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(11));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(11));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(11));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(11));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(11));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(11));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(11));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(11));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(11));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(11));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(11));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(11));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(11));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(11));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(11));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(11));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(11));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(11));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(11));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(11));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(11));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(11));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(11));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(11));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(11));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(11));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(11));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(11));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(11));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(11));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(11));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(11));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(11));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(11));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(11));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(11));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(11));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(11));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(11));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(11));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(11));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(11));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(11));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(11));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(11));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(11));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(11));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(11));
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
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(12));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(12));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(12));
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
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(12));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(12));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(12));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(12));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(12));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(12));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(12));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(12));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(12));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(12));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(12));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(12));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(12));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(12));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(12));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(12));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(12));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(12));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(12));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(12));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(12));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(12));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(12));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(12));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(12));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(12));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(12));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(12));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(12));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(12));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(12));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(12));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(12));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(12));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(12));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(12));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(12));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(12));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(12));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(12));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(12));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(12));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(12));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(12));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(12));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(12));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(12));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(12));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(12));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(12));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(12));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(12));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(12));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(12));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(12));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(12));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(12));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(12));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(12));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(12));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(12));
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
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(13));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(13));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(13));
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
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(13));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(13));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(13));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(13));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(13));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(13));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(13));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(13));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(13));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(13));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(13));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(13));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(13));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(13));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(13));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(13));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(13));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(13));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(13));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(13));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(13));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(13));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(13));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(13));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(13));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(13));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(13));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(13));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(13));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(13));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(13));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(13));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(13));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(13));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(13));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(13));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(13));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(13));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(13));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(13));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(13));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(13));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(13));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(13));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(13));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(13));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(13));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(13));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(13));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(13));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(13));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(13));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(13));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(13));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(13));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(13));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(13));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(13));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(13));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(13));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(13));
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
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(14));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(14));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(14));
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
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(14));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(14));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(14));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(14));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(14));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(14));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(14));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(14));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(14));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(14));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(14));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(14));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(14));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(14));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(14));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(14));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(14));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(14));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(14));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(14));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(14));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(14));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(14));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(14));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(14));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(14));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(14));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(14));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(14));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(14));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(14));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(14));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(14));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(14));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(14));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(14));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(14));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(14));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(14));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(14));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(14));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(14));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(14));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(14));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(14));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(14));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(14));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(14));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(14));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(14));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(14));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(14));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(14));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(14));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(14));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(14));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(14));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(14));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(14));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(14));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(14));
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
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(15));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(15));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(15));
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
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(15));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(15));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(15));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(15));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(15));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(15));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(15));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(15));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(15));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(15));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(15));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(15));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(15));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(15));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(15));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(15));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(15));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(15));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(15));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(15));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(15));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(15));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(15));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(15));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(15));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(15));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(15));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(15));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(15));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(15));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(15));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(15));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(15));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(15));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(15));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(15));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(15));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(15));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(15));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(15));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(15));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(15));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(15));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(15));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(15));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(15));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(15));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(15));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(15));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(15));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(15));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(15));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(15));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(15));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(15));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(15));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(15));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(15));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(15));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(15));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(15));
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
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(16));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(16));
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
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(16));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(16));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(16));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(16));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(16));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(16));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(16));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(16));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(16));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(16));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(16));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(16));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(16));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(16));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(16));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(16));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(16));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(16));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(16));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(16));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(16));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(16));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(16));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(16));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(16));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(16));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(16));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(16));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(1));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(1));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(1));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(1));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(1));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(1));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(1));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(1));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(1));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(1));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(2));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(2));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(2));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(2));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(2));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(2));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(2));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(2));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(2));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(2));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(3));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(3));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(3));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(3));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(3));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(3));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(3));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(3));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(3));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(3));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
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
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
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
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(1));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(1));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(1));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(1));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(4));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(4));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(4));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(4));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(4));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(4));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(4));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(4));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(4));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(4));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(4));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(4));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(4));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(4));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(4));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(4));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(5));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(5));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(5));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(5));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(5));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(5));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(5));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(5));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(5));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(5));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(5));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(5));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(5));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(5));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(5));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(5));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(5));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(5));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(6));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(6));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(6));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(6));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(6));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(6));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(6));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(6));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(6));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(6));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(6));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(6));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(6));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(6));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(6));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(6));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(6));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(6));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(7));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(7));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(7));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(7));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(7));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(7));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(7));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(7));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(7));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(7));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(7));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(7));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(7));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(7));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(7));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(7));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(7));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(7));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(8));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(8));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(8));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(8));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(8));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(8));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(8));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(8));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(8));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(8));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(8));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(8));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(8));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(8));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(8));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(8));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(8));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_regt_3,(8));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(8));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(8));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(8));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(8));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(8));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(8));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(8));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_regt_3,(8));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(9));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(9));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(9));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(9));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(9));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(9));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(9));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(9));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(9));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(9));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(9));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(9));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(9));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(9));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(9));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(9));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(9));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(9));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(9));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(9));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(9));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(9));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(9));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(9));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(9));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(9));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(9));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(9));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(9));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(9));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(9));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(9));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(9));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(9));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_regt_3,(9));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(9));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(9));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(9));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(9));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(9));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(9));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(9));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(9));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(9));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(9));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(9));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(9));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_regt_3,(9));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(10));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(10));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(10));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(10));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(10));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(10));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(10));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(10));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(10));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(10));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(10));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(10));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(10));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(10));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(10));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(10));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(10));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(10));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(10));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(10));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(10));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(10));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(10));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(10));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(10));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(10));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(10));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(10));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(10));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(10));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(10));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(10));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(10));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(10));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_regt_3,(10));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(10));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(10));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(10));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(10));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(10));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(10));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(10));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(10));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(10));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(10));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(10));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(10));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_regt_3,(10));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(11));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(11));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(11));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(11));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(11));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(11));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(11));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(11));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(11));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(11));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(11));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(11));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(11));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(11));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(11));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(11));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(11));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(11));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(11));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(11));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(11));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(11));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(11));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(11));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(11));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(11));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(11));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(11));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(11));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(11));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(11));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(11));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(11));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(11));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_regt_3,(11));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(11));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(11));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(11));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(11));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(11));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(11));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(11));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(11));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(11));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(11));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(11));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(11));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_regt_3,(11));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(12));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(12));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(12));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(12));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(12));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(12));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(12));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(12));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(12));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(12));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(12));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(12));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(12));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(12));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(12));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(12));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(12));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(12));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(12));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(12));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(12));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(12));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(12));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(12));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(12));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(12));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(12));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(12));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(12));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(12));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(12));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(12));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(12));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(12));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_regt_3,(12));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(12));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(12));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(12));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(12));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(12));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(12));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(12));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(12));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(12));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(12));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(12));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(12));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_regt_3,(12));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(13));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(13));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(13));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(13));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(13));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(13));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(13));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(13));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(13));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(13));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(13));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(13));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(13));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(13));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(13));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(13));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(13));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(13));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(13));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(13));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(13));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(13));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(13));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(13));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(13));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(13));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(13));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(13));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(13));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(13));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(13));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(13));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(13));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(13));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_regt_3,(13));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(13));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(13));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(13));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(13));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(13));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(13));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(13));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(13));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(13));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(13));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(13));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(13));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_regt_3,(13));
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
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
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
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(2));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(2));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(2));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(2));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(14));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(14));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(14));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(14));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(14));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(14));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(14));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(14));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(14));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(14));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(14));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(14));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(14));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(14));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(14));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(14));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(14));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(14));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(14));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(14));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(14));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(14));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(14));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(14));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(14));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(14));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(14));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(14));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(14));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(14));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(14));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(14));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(14));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(14));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_regt_3,(14));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(14));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(14));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(14));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(14));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(14));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(14));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(14));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(14));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(14));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(14));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(14));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(14));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_regt_3,(14));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(15));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_reg_0_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(15));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(15));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(15));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(15));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(15));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(15));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(15));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(15));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(15));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(15));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(15));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(15));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(15));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(15));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(15));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(15));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(15));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(15));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(15));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(15));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(15));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(15));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(15));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(15));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(15));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(15));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(15));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(15));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(15));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(15));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(15));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(15));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(15));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_regt_3,(15));
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
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_regt_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_3_0;
C_regt_0 = __riscv_vle16_v_f16m1(&C[16],(15));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(15));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(15));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(15));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(15));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(15));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(15));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(15));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(15));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_regt_0,(15));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_regt_1,(15));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_regt_2,(15));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_regt_3,(15));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_3_1;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_reg_3_1 = __riscv_vfmacc_vv_f16m1(C_reg_3_1, A_reg_1, B_reg_3,(16));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_3_1;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(16));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(16));
C_reg_3_1 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_reg_3_1 = __riscv_vfmacc_vv_f16m1(C_reg_3_1, A_reg_1, B_reg_3,(16));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
}

// gemm_RVV_33x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 33] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 33] @DRAM
// )
void gemm_RVV_33x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(1));
}

// gemm_RVV_33x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 33] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 33] @DRAM
// )
void gemm_RVV_33x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(1));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(1));
}

// gemm_RVV_33x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 33] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 33] @DRAM
// )
void gemm_RVV_33x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(1));
}

// gemm_RVV_33x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 33] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 33] @DRAM
// )
void gemm_RVV_33x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(1));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(1));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(1));
}

// gemm_RVV_33x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 33] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 33] @DRAM
// )
void gemm_RVV_33x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(1));
}

// gemm_RVV_33x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 33] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 33] @DRAM
// )
void gemm_RVV_33x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(1));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(1));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 32],(1));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(1));
}

// gemm_RVV_33x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 33] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 33] @DRAM
// )
void gemm_RVV_33x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_reg_3_1 = __riscv_vfmacc_vv_f16m1(C_reg_3_1, A_reg_1, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 32], C_regt_3,(1));
}

// gemm_RVV_33x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 33] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 33] @DRAM
// )
void gemm_RVV_33x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(1));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(1));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 32],(1));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 32],(1));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(16));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(16));
C_reg_3_1 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_reg_3_1 = __riscv_vfmacc_vv_f16m1(C_reg_3_1, A_reg_1, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(1));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 32], C_regt_3,(1));
}

// gemm_RVV_34x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 34] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 34] @DRAM
// )
void gemm_RVV_34x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(2));
}

// gemm_RVV_34x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 34] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 34] @DRAM
// )
void gemm_RVV_34x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(2));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(2));
}

// gemm_RVV_34x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 34] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 34] @DRAM
// )
void gemm_RVV_34x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(2));
}

// gemm_RVV_34x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 34] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 34] @DRAM
// )
void gemm_RVV_34x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(2));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(2));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(2));
}

// gemm_RVV_34x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 34] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 34] @DRAM
// )
void gemm_RVV_34x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(2));
}

// gemm_RVV_34x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 34] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 34] @DRAM
// )
void gemm_RVV_34x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(2));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(2));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 32],(2));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(2));
}

// gemm_RVV_34x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 34] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 34] @DRAM
// )
void gemm_RVV_34x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_reg_3_1 = __riscv_vfmacc_vv_f16m1(C_reg_3_1, A_reg_1, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 32], C_regt_3,(2));
}

// gemm_RVV_34x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 34] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 34] @DRAM
// )
void gemm_RVV_34x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(2));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(2));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 32],(2));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 32],(2));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(16));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(16));
C_reg_3_1 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_reg_3_1 = __riscv_vfmacc_vv_f16m1(C_reg_3_1, A_reg_1, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(2));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 32], C_regt_3,(2));
}

// gemm_RVV_35x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 35] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 35] @DRAM
// )
void gemm_RVV_35x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(3));
}

// gemm_RVV_35x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 35] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 35] @DRAM
// )
void gemm_RVV_35x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(3));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(3));
}

// gemm_RVV_35x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 35] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 35] @DRAM
// )
void gemm_RVV_35x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(3));
}

// gemm_RVV_35x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 35] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 35] @DRAM
// )
void gemm_RVV_35x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(3));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(3));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(3));
}

// gemm_RVV_35x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 35] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 35] @DRAM
// )
void gemm_RVV_35x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(3));
}

// gemm_RVV_35x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 35] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 35] @DRAM
// )
void gemm_RVV_35x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(3));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(3));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 32],(3));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(3));
}

// gemm_RVV_35x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 35] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 35] @DRAM
// )
void gemm_RVV_35x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_reg_3_1 = __riscv_vfmacc_vv_f16m1(C_reg_3_1, A_reg_1, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 32], C_regt_3,(3));
}

// gemm_RVV_35x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 35] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 35] @DRAM
// )
void gemm_RVV_35x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(3));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(3));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 32],(3));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 32],(3));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(16));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(16));
C_reg_3_1 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_reg_3_1 = __riscv_vfmacc_vv_f16m1(C_reg_3_1, A_reg_1, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(3));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 32], C_regt_3,(3));
}

// gemm_RVV_36x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 36] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 36] @DRAM
// )
void gemm_RVV_36x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(4));
}

// gemm_RVV_36x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 36] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 36] @DRAM
// )
void gemm_RVV_36x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(4));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(4));
}

// gemm_RVV_36x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 36] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 36] @DRAM
// )
void gemm_RVV_36x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(4));
}

// gemm_RVV_36x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 36] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 36] @DRAM
// )
void gemm_RVV_36x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(4));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(4));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(4));
}

// gemm_RVV_36x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 36] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 36] @DRAM
// )
void gemm_RVV_36x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(4));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(4));
}

// gemm_RVV_36x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 36] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 36] @DRAM
// )
void gemm_RVV_36x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(4));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(4));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 32],(4));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(4));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(4));
}

// gemm_RVV_36x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 36] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 36] @DRAM
// )
void gemm_RVV_36x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_reg_3_1 = __riscv_vfmacc_vv_f16m1(C_reg_3_1, A_reg_1, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(4));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(4));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 32], C_regt_3,(4));
}

// gemm_RVV_36x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 36] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 36] @DRAM
// )
void gemm_RVV_36x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(4));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(4));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 32],(4));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 32],(4));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(16));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(16));
C_reg_3_1 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_reg_3_1 = __riscv_vfmacc_vv_f16m1(C_reg_3_1, A_reg_1, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(4));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(4));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(4));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(4));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 32], C_regt_3,(4));
}

// gemm_RVV_37x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 37] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 37] @DRAM
// )
void gemm_RVV_37x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(5));
}

// gemm_RVV_37x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 37] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 37] @DRAM
// )
void gemm_RVV_37x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(5));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(5));
}

// gemm_RVV_37x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 37] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 37] @DRAM
// )
void gemm_RVV_37x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(5));
}

// gemm_RVV_37x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 37] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 37] @DRAM
// )
void gemm_RVV_37x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(5));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(5));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(5));
}

// gemm_RVV_37x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 37] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 37] @DRAM
// )
void gemm_RVV_37x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(5));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(5));
}

// gemm_RVV_37x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 37] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 37] @DRAM
// )
void gemm_RVV_37x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(5));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(5));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 32],(5));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(5));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(5));
}

// gemm_RVV_37x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 37] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 37] @DRAM
// )
void gemm_RVV_37x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(5));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_reg_3_1 = __riscv_vfmacc_vv_f16m1(C_reg_3_1, A_reg_1, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(5));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(5));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 32], C_regt_3,(5));
}

// gemm_RVV_37x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 37] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 37] @DRAM
// )
void gemm_RVV_37x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(5));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(5));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 32],(5));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 32],(5));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(16));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(16));
C_reg_3_1 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(5));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_reg_3_1 = __riscv_vfmacc_vv_f16m1(C_reg_3_1, A_reg_1, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(5));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(5));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(5));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(5));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 32], C_regt_3,(5));
}

// gemm_RVV_38x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 38] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 38] @DRAM
// )
void gemm_RVV_38x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(6));
}

// gemm_RVV_38x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 38] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 38] @DRAM
// )
void gemm_RVV_38x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(6));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(6));
}

// gemm_RVV_38x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 38] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 38] @DRAM
// )
void gemm_RVV_38x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(6));
}

// gemm_RVV_38x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 38] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 38] @DRAM
// )
void gemm_RVV_38x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(6));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(6));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(6));
}

// gemm_RVV_38x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 38] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 38] @DRAM
// )
void gemm_RVV_38x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(6));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(6));
}

// gemm_RVV_38x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 38] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 38] @DRAM
// )
void gemm_RVV_38x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(6));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(6));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 32],(6));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(6));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(6));
}

// gemm_RVV_38x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 38] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 38] @DRAM
// )
void gemm_RVV_38x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(6));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_reg_3_1 = __riscv_vfmacc_vv_f16m1(C_reg_3_1, A_reg_1, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(6));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(6));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 32], C_regt_3,(6));
}

// gemm_RVV_38x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 38] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 38] @DRAM
// )
void gemm_RVV_38x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(6));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(6));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 32],(6));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 32],(6));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(16));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(16));
C_reg_3_1 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(6));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_reg_3_1 = __riscv_vfmacc_vv_f16m1(C_reg_3_1, A_reg_1, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(6));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(6));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(6));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(6));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 32], C_regt_3,(6));
}

// gemm_RVV_39x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 39] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 39] @DRAM
// )
void gemm_RVV_39x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(7));
}

// gemm_RVV_39x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 39] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 39] @DRAM
// )
void gemm_RVV_39x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(7));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(7));
}

// gemm_RVV_39x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 39] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 39] @DRAM
// )
void gemm_RVV_39x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(7));
}

// gemm_RVV_39x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 39] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 39] @DRAM
// )
void gemm_RVV_39x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(7));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(7));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(7));
}

// gemm_RVV_39x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 39] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 39] @DRAM
// )
void gemm_RVV_39x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(7));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(7));
}

// gemm_RVV_39x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 39] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 39] @DRAM
// )
void gemm_RVV_39x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(7));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(7));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 32],(7));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(7));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(7));
}

// gemm_RVV_39x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 39] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 39] @DRAM
// )
void gemm_RVV_39x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(7));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_reg_3_1 = __riscv_vfmacc_vv_f16m1(C_reg_3_1, A_reg_1, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(7));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(7));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 32], C_regt_3,(7));
}

// gemm_RVV_39x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 39] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 39] @DRAM
// )
void gemm_RVV_39x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(7));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(7));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 32],(7));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 32],(7));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(16));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(16));
C_reg_3_1 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(7));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_reg_3_1 = __riscv_vfmacc_vv_f16m1(C_reg_3_1, A_reg_1, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(7));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(7));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(7));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(7));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 32], C_regt_3,(7));
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
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
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
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(3));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(3));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(3));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
}

// gemm_RVV_40x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 40] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 40] @DRAM
// )
void gemm_RVV_40x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(8));
}

// gemm_RVV_40x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 40] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 40] @DRAM
// )
void gemm_RVV_40x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(8));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(8));
}

// gemm_RVV_40x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 40] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 40] @DRAM
// )
void gemm_RVV_40x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(8));
}

// gemm_RVV_40x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 40] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 40] @DRAM
// )
void gemm_RVV_40x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(8));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(8));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(8));
}

// gemm_RVV_40x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 40] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 40] @DRAM
// )
void gemm_RVV_40x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(8));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(8));
}

// gemm_RVV_40x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 40] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 40] @DRAM
// )
void gemm_RVV_40x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(8));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(8));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 32],(8));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(8));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(8));
}

// gemm_RVV_40x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 40] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 40] @DRAM
// )
void gemm_RVV_40x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(8));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_reg_3_1 = __riscv_vfmacc_vv_f16m1(C_reg_3_1, A_reg_1, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(8));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(8));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 32], C_regt_3,(8));
}

// gemm_RVV_40x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 40] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 40] @DRAM
// )
void gemm_RVV_40x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(8));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(8));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 32],(8));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 32],(8));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(16));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(16));
C_reg_3_1 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(8));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_reg_3_1 = __riscv_vfmacc_vv_f16m1(C_reg_3_1, A_reg_1, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(8));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(8));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(8));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(8));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 32], C_regt_3,(8));
}

// gemm_RVV_41x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 41] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 41] @DRAM
// )
void gemm_RVV_41x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(9));
}

// gemm_RVV_41x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 41] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 41] @DRAM
// )
void gemm_RVV_41x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(9));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(9));
}

// gemm_RVV_41x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 41] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 41] @DRAM
// )
void gemm_RVV_41x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(9));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(9));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(9));
}

// gemm_RVV_41x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 41] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 41] @DRAM
// )
void gemm_RVV_41x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(9));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(9));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(9));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(9));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(9));
}

// gemm_RVV_41x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 41] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 41] @DRAM
// )
void gemm_RVV_41x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(9));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(9));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(9));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(9));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(9));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(9));
}

// gemm_RVV_41x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 41] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 41] @DRAM
// )
void gemm_RVV_41x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(9));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(9));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 32],(9));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(9));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(9));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(9));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(9));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(9));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(9));
}

// gemm_RVV_41x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 41] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 41] @DRAM
// )
void gemm_RVV_41x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(9));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(9));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_reg_3_1 = __riscv_vfmacc_vv_f16m1(C_reg_3_1, A_reg_1, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(9));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(9));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(9));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(9));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(9));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(9));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 32], C_regt_3,(9));
}

// gemm_RVV_41x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 41] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 41] @DRAM
// )
void gemm_RVV_41x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(9));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(9));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 32],(9));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 32],(9));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(16));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(16));
C_reg_3_1 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(9));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(9));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_reg_3_1 = __riscv_vfmacc_vv_f16m1(C_reg_3_1, A_reg_1, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(9));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(9));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(9));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(9));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(9));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(9));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(9));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 32], C_regt_3,(9));
}

// gemm_RVV_42x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 42] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 42] @DRAM
// )
void gemm_RVV_42x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(10));
}

// gemm_RVV_42x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 42] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 42] @DRAM
// )
void gemm_RVV_42x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(10));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(10));
}

// gemm_RVV_42x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 42] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 42] @DRAM
// )
void gemm_RVV_42x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(10));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(10));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(10));
}

// gemm_RVV_42x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 42] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 42] @DRAM
// )
void gemm_RVV_42x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(10));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(10));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(10));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(10));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(10));
}

// gemm_RVV_42x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 42] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 42] @DRAM
// )
void gemm_RVV_42x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(10));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(10));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(10));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(10));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(10));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(10));
}

// gemm_RVV_42x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 42] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 42] @DRAM
// )
void gemm_RVV_42x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(10));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(10));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 32],(10));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(10));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(10));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(10));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(10));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(10));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(10));
}

// gemm_RVV_42x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 42] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 42] @DRAM
// )
void gemm_RVV_42x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(10));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(10));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(10));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_reg_3_1 = __riscv_vfmacc_vv_f16m1(C_reg_3_1, A_reg_1, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(10));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(10));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(10));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(10));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(10));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(10));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 32], C_regt_3,(10));
}

// gemm_RVV_42x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 42] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 42] @DRAM
// )
void gemm_RVV_42x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(10));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(10));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 32],(10));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 32],(10));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(16));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(16));
C_reg_3_1 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(10));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(10));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_reg_3_1 = __riscv_vfmacc_vv_f16m1(C_reg_3_1, A_reg_1, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(10));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(10));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(10));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(10));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(10));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(10));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(10));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 32], C_regt_3,(10));
}

// gemm_RVV_43x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 43] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 43] @DRAM
// )
void gemm_RVV_43x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(11));
}

// gemm_RVV_43x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 43] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 43] @DRAM
// )
void gemm_RVV_43x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(11));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(11));
}

// gemm_RVV_43x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 43] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 43] @DRAM
// )
void gemm_RVV_43x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(11));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(11));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(11));
}

// gemm_RVV_43x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 43] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 43] @DRAM
// )
void gemm_RVV_43x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(11));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(11));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(11));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(11));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(11));
}

// gemm_RVV_43x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 43] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 43] @DRAM
// )
void gemm_RVV_43x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(11));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(11));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(11));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(11));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(11));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(11));
}

// gemm_RVV_43x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 43] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 43] @DRAM
// )
void gemm_RVV_43x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(11));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(11));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 32],(11));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(11));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(11));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(11));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(11));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(11));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(11));
}

// gemm_RVV_43x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 43] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 43] @DRAM
// )
void gemm_RVV_43x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(11));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(11));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(11));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_reg_3_1 = __riscv_vfmacc_vv_f16m1(C_reg_3_1, A_reg_1, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(11));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(11));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(11));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(11));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(11));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(11));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 32], C_regt_3,(11));
}

// gemm_RVV_43x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 43] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 43] @DRAM
// )
void gemm_RVV_43x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(11));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(11));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 32],(11));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 32],(11));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(16));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(16));
C_reg_3_1 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(11));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(11));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_reg_3_1 = __riscv_vfmacc_vv_f16m1(C_reg_3_1, A_reg_1, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(11));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(11));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(11));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(11));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(11));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(11));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(11));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 32], C_regt_3,(11));
}

// gemm_RVV_44x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 44] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 44] @DRAM
// )
void gemm_RVV_44x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(12));
}

// gemm_RVV_44x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 44] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 44] @DRAM
// )
void gemm_RVV_44x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(12));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(12));
}

// gemm_RVV_44x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 44] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 44] @DRAM
// )
void gemm_RVV_44x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(12));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(12));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(12));
}

// gemm_RVV_44x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 44] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 44] @DRAM
// )
void gemm_RVV_44x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(12));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(12));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(12));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(12));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(12));
}

// gemm_RVV_44x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 44] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 44] @DRAM
// )
void gemm_RVV_44x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(12));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(12));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(12));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(12));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(12));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(12));
}

// gemm_RVV_44x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 44] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 44] @DRAM
// )
void gemm_RVV_44x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(12));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(12));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 32],(12));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(12));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(12));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(12));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(12));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(12));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(12));
}

// gemm_RVV_44x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 44] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 44] @DRAM
// )
void gemm_RVV_44x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(12));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(12));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(12));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_reg_3_1 = __riscv_vfmacc_vv_f16m1(C_reg_3_1, A_reg_1, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(12));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(12));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(12));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(12));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(12));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(12));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 32], C_regt_3,(12));
}

// gemm_RVV_44x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 44] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 44] @DRAM
// )
void gemm_RVV_44x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(12));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(12));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 32],(12));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 32],(12));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(16));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(16));
C_reg_3_1 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(12));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(12));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_reg_3_1 = __riscv_vfmacc_vv_f16m1(C_reg_3_1, A_reg_1, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(12));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(12));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(12));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(12));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(12));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(12));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(12));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 32], C_regt_3,(12));
}

// gemm_RVV_45x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 45] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 45] @DRAM
// )
void gemm_RVV_45x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(13));
}

// gemm_RVV_45x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 45] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 45] @DRAM
// )
void gemm_RVV_45x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(13));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(13));
}

// gemm_RVV_45x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 45] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 45] @DRAM
// )
void gemm_RVV_45x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(13));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(13));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(13));
}

// gemm_RVV_45x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 45] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 45] @DRAM
// )
void gemm_RVV_45x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(13));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(13));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(13));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(13));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(13));
}

// gemm_RVV_45x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 45] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 45] @DRAM
// )
void gemm_RVV_45x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(13));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(13));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(13));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(13));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(13));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(13));
}

// gemm_RVV_45x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 45] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 45] @DRAM
// )
void gemm_RVV_45x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(13));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(13));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 32],(13));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(13));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(13));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(13));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(13));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(13));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(13));
}

// gemm_RVV_45x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 45] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 45] @DRAM
// )
void gemm_RVV_45x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(13));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(13));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(13));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_reg_3_1 = __riscv_vfmacc_vv_f16m1(C_reg_3_1, A_reg_1, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(13));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(13));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(13));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(13));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(13));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(13));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 32], C_regt_3,(13));
}

// gemm_RVV_45x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 45] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 45] @DRAM
// )
void gemm_RVV_45x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(13));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(13));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 32],(13));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 32],(13));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(16));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(16));
C_reg_3_1 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(13));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(13));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_reg_3_1 = __riscv_vfmacc_vv_f16m1(C_reg_3_1, A_reg_1, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(13));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(13));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(13));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(13));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(13));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(13));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(13));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 32], C_regt_3,(13));
}

// gemm_RVV_46x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 46] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 46] @DRAM
// )
void gemm_RVV_46x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(14));
}

// gemm_RVV_46x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 46] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 46] @DRAM
// )
void gemm_RVV_46x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(14));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(14));
}

// gemm_RVV_46x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 46] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 46] @DRAM
// )
void gemm_RVV_46x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(14));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(14));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(14));
}

// gemm_RVV_46x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 46] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 46] @DRAM
// )
void gemm_RVV_46x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(14));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(14));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(14));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(14));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(14));
}

// gemm_RVV_46x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 46] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 46] @DRAM
// )
void gemm_RVV_46x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(14));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(14));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(14));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(14));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(14));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(14));
}

// gemm_RVV_46x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 46] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 46] @DRAM
// )
void gemm_RVV_46x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(14));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(14));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 32],(14));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(14));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(14));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(14));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(14));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(14));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(14));
}

// gemm_RVV_46x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 46] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 46] @DRAM
// )
void gemm_RVV_46x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(14));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(14));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(14));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_reg_3_1 = __riscv_vfmacc_vv_f16m1(C_reg_3_1, A_reg_1, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(14));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(14));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(14));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(14));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(14));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(14));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 32], C_regt_3,(14));
}

// gemm_RVV_46x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 46] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 46] @DRAM
// )
void gemm_RVV_46x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(14));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(14));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 32],(14));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 32],(14));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(16));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(16));
C_reg_3_1 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(14));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(14));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_reg_3_1 = __riscv_vfmacc_vv_f16m1(C_reg_3_1, A_reg_1, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(14));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(14));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(14));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(14));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(14));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(14));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(14));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 32], C_regt_3,(14));
}

// gemm_RVV_47x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 47] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 47] @DRAM
// )
void gemm_RVV_47x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(15));
}

// gemm_RVV_47x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 47] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 47] @DRAM
// )
void gemm_RVV_47x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(15));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(15));
}

// gemm_RVV_47x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 47] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 47] @DRAM
// )
void gemm_RVV_47x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(15));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(15));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(15));
}

// gemm_RVV_47x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 47] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 47] @DRAM
// )
void gemm_RVV_47x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(15));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(15));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(15));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(15));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(15));
}

// gemm_RVV_47x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 47] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 47] @DRAM
// )
void gemm_RVV_47x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(15));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(15));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(15));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(15));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(15));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(15));
}

// gemm_RVV_47x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 47] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 47] @DRAM
// )
void gemm_RVV_47x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_regt_0;
vfloat16m1_t C_regt_1;
vfloat16m1_t C_regt_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(15));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(15));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 32],(15));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(15));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(15));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(15));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(15));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(15));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(15));
}

// gemm_RVV_47x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 47] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 47] @DRAM
// )
void gemm_RVV_47x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
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
C_regt_0 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
C_regt_1 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
C_regt_2 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
C_regt_3 = __riscv_vfmv_v_f_f16m1(0.0f,(15));
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(15));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(15));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_reg_3_1 = __riscv_vfmacc_vv_f16m1(C_reg_3_1, A_reg_1, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(15));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(15));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(15));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(15));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(15));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(15));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 32], C_regt_3,(15));
}

// gemm_RVV_47x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 47] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 47] @DRAM
// )
void gemm_RVV_47x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_regt;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
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
C_regt_0 = __riscv_vle16_v_f16m1(&C[32],(15));
C_regt_1 = __riscv_vle16_v_f16m1(&C[ldc + 32],(15));
C_regt_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 32],(15));
C_regt_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 32],(15));
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(16));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(16));
C_reg_3_1 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_regt = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(15));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(15));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_reg_3_1 = __riscv_vfmacc_vv_f16m1(C_reg_3_1, A_reg_1, B_reg_3,(16));
  C_regt_0 = __riscv_vfmacc_vv_f16m1(C_regt_0, A_regt, B_reg_0,(15));
  C_regt_1 = __riscv_vfmacc_vv_f16m1(C_regt_1, A_regt, B_reg_1,(15));
  C_regt_2 = __riscv_vfmacc_vv_f16m1(C_regt_2, A_regt, B_reg_2,(15));
  C_regt_3 = __riscv_vfmacc_vv_f16m1(C_regt_3, A_regt, B_reg_3,(15));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_regt_0,(15));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_regt_1,(15));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_regt_2,(15));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 32], C_regt_3,(15));
}

// gemm_RVV_48x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 48] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 48] @DRAM
// )
void gemm_RVV_48x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(16));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_reg_0_2,(16));
}

// gemm_RVV_48x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 48] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 48] @DRAM
// )
void gemm_RVV_48x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(16));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_reg_0_2,(16));
}

// gemm_RVV_48x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 48] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 48] @DRAM
// )
void gemm_RVV_48x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(16));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_reg_1_2,(16));
}

// gemm_RVV_48x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 48] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 48] @DRAM
// )
void gemm_RVV_48x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[32],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C[ldc + 32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(16));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_reg_1_2,(16));
}

// gemm_RVV_48x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 48] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 48] @DRAM
// )
void gemm_RVV_48x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(16));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_2_2 = __riscv_vfmacc_vv_f16m1(C_reg_2_2, A_reg_2, B_reg_2,(16));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_reg_2_2,(16));
}

// gemm_RVV_48x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 48] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 48] @DRAM
// )
void gemm_RVV_48x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t C_reg_0_0;
vfloat16m1_t C_reg_0_1;
vfloat16m1_t C_reg_0_2;
vfloat16m1_t C_reg_1_0;
vfloat16m1_t C_reg_1_1;
vfloat16m1_t C_reg_1_2;
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[32],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C[ldc + 32],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(16));
C_reg_2_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(16));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_2_2 = __riscv_vfmacc_vv_f16m1(C_reg_2_2, A_reg_2, B_reg_2,(16));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_reg_2_2,(16));
}

// gemm_RVV_48x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 48] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 48] @DRAM
// )
void gemm_RVV_48x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
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
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(16));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_2_2 = __riscv_vfmacc_vv_f16m1(C_reg_2_2, A_reg_2, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_reg_3_1 = __riscv_vfmacc_vv_f16m1(C_reg_3_1, A_reg_1, B_reg_3,(16));
  C_reg_3_2 = __riscv_vfmacc_vv_f16m1(C_reg_3_2, A_reg_2, B_reg_3,(16));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_reg_2_2,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 32], C_reg_3_2,(16));
}

// gemm_RVV_48x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 48] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 48] @DRAM
// )
void gemm_RVV_48x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
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
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[32],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C[ldc + 32],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(16));
C_reg_2_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 32],(16));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(16));
C_reg_3_1 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(16));
C_reg_3_2 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(16));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(16));
  C_reg_0_0 = __riscv_vfmacc_vv_f16m1(C_reg_0_0, A_reg_0, B_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vv_f16m1(C_reg_0_1, A_reg_1, B_reg_0,(16));
  C_reg_0_2 = __riscv_vfmacc_vv_f16m1(C_reg_0_2, A_reg_2, B_reg_0,(16));
  C_reg_1_0 = __riscv_vfmacc_vv_f16m1(C_reg_1_0, A_reg_0, B_reg_1,(16));
  C_reg_1_1 = __riscv_vfmacc_vv_f16m1(C_reg_1_1, A_reg_1, B_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vv_f16m1(C_reg_1_2, A_reg_2, B_reg_1,(16));
  C_reg_2_0 = __riscv_vfmacc_vv_f16m1(C_reg_2_0, A_reg_0, B_reg_2,(16));
  C_reg_2_1 = __riscv_vfmacc_vv_f16m1(C_reg_2_1, A_reg_1, B_reg_2,(16));
  C_reg_2_2 = __riscv_vfmacc_vv_f16m1(C_reg_2_2, A_reg_2, B_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vv_f16m1(C_reg_3_0, A_reg_0, B_reg_3,(16));
  C_reg_3_1 = __riscv_vfmacc_vv_f16m1(C_reg_3_1, A_reg_1, B_reg_3,(16));
  C_reg_3_2 = __riscv_vfmacc_vv_f16m1(C_reg_3_2, A_reg_2, B_reg_3,(16));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_reg_2_2,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 32], C_reg_3_2,(16));
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
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
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
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(4));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(4));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(4));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(4));
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
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
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
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(5));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(5));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(5));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(5));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(5));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(5));
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
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
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
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(6));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(6));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(6));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(6));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(6));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(6));
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
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
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
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(7));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(7));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(7));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(7));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(7));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(7));
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
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
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
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(8));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(8));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(8));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(8));
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
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(8));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(8));
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
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(9));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(9));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(9));
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
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(9));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(9));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(9));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(9));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(9));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(9));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(9));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(9));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(9));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(9));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(9));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(9));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(9));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(9));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(9));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(9));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(9));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(9));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(9));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(9));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(9));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(9));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(9));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(9));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(9));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(9));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(9));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(9));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(9));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(9));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(9));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(9));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(9));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(9));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(9));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(9));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(9));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(9));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(9));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(9));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(9));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(9));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(9));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(9));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(9));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(9));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(9));
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
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(9));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(9));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(9));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(9));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (48)],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (4)],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 1],(9));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 2],(9));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (4) + 3],(9));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(9));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(9));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(9));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(9));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(9));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(9));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(9));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(9));
}


/* relying on the following instruction..."
rvv_broadcast_16xf16(dst,src,vl)
{dst_data} = __riscv_vfmv_v_f_f16m1({src_data},{vl});
*/

/* relying on the following instruction..."
rvv_broadcast_16xf16_0(dst,vl)
{dst_data} = __riscv_vfmv_v_f_f16m1(0.0f,{vl});
*/

/* relying on the following instruction..."
rvv_vfmacc_16xf16_16xf16(dst,lhs,rhs,vl)
{dst_data} = __riscv_vfmacc_vv_f16m1({dst_data}, {lhs_data}, {rhs_data},{vl});
*/

/* relying on the following instruction..."
rvv_vld_16xf16(dst,src,vl)
{dst_data} = __riscv_vle16_v_f16m1(&{src_data},{vl});
*/

/* relying on the following instruction..."
rvv_vst_16xf16(dst,src,vl)
__riscv_vse16_v_f16m1(&{dst_data}, {src_data},{vl});
*/
