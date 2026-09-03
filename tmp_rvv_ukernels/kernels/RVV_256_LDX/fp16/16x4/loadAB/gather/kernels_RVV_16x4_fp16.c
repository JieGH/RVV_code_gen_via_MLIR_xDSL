#include "kernels_RVV_16x4_fp16.h"



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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(10));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(10));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(10));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(10));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(10));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(10));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(10));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(10));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(10));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(10));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 3],(10));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(10));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(10));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(10));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(10));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 3],(10));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(11));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(11));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(11));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(11));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(11));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(11));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(11));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(11));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(11));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(11));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 3],(11));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(11));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(11));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(11));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(11));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 3],(11));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(12));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(12));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(12));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(12));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(12));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(12));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(12));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(12));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(12));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(12));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 3],(12));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(12));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(12));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(12));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(12));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 3],(12));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(13));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(13));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(13));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(13));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(13));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(13));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(13));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(13));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(13));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(13));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 3],(13));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(13));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(13));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(13));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(13));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 3],(13));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(14));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(14));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(14));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(14));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(14));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(14));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(14));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(14));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(14));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(14));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 3],(14));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(14));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(14));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(14));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(14));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 3],(14));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(15));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(15));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(15));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(15));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(15));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(15));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(15));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(15));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(15));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(15));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 3],(15));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(15));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(15));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(15));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(15));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 3],(15));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(16));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(16));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(16));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(16));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(16));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(16));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(16));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(16));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(16));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(16));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(16));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(16));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(16));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 3],(16));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(16));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(16));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(16));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(16));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 3],(16));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(1));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(1));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(1));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(1));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(1));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(1));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 3],(1));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 3],(1));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(2));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(2));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(2));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(2));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(2));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(2));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 3],(2));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 3],(2));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(3));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(3));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(3));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(3));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(3));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(3));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 3],(3));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 3],(3));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(4));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(4));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(4));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(4));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(4));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(4));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 3],(4));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 3],(4));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(5));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(5));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(5));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(5));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(5));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(5));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(5));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 3],(5));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(5));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 3],(5));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(6));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(6));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(6));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(6));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(6));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(6));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(6));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 3],(6));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(6));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 3],(6));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(7));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(7));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(7));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(7));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(7));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(7));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(7));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 3],(7));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(7));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 3],(7));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(8));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(8));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(8));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(8));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(8));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(8));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(8));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 3],(8));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(8));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 3],(8));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(9));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(9));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(9));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(9));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(9));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(9));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(9));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(9));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(9));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(9));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 3],(9));
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
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(9));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb)],(9));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 1],(9));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 2],(9));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (ldb) + 3],(9));
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
