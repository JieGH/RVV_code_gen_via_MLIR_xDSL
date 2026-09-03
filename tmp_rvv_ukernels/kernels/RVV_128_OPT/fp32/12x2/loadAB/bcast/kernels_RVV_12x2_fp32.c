#include "kernels_RVV_12x2_fp32.h"



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
void gemm_RVV_10x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (12) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (12) + 8],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(2));
}

// gemm_RVV_10x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 10] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 10] @DRAM
// )
void gemm_RVV_10x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
C_regt_0 = __riscv_vle32_v_f32m1(&C[8],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (12) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (12) + 8],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(2));
}

// gemm_RVV_10x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 10] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 10] @DRAM
// )
void gemm_RVV_10x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (12) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (12) + 8],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_regt_1,(2));
}

// gemm_RVV_10x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 10] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 10] @DRAM
// )
void gemm_RVV_10x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
C_regt_0 = __riscv_vle32_v_f32m1(&C[8],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 8],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (12) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (12) + 8],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_regt_1,(2));
}

// gemm_RVV_11x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 11] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 11] @DRAM
// )
void gemm_RVV_11x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (12) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (12) + 8],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(3));
}

// gemm_RVV_11x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 11] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 11] @DRAM
// )
void gemm_RVV_11x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
C_regt_0 = __riscv_vle32_v_f32m1(&C[8],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (12) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (12) + 8],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(3));
}

// gemm_RVV_11x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 11] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 11] @DRAM
// )
void gemm_RVV_11x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (12) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (12) + 8],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_regt_1,(3));
}

// gemm_RVV_11x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 11] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 11] @DRAM
// )
void gemm_RVV_11x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
C_regt_0 = __riscv_vle32_v_f32m1(&C[8],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 8],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (12) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (12) + 8],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_regt_1,(3));
}

// gemm_RVV_12x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 12] @DRAM
// )
void gemm_RVV_12x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (12) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (12) + 8],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
}

// gemm_RVV_12x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 12] @DRAM
// )
void gemm_RVV_12x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (12) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (12) + 8],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
}

// gemm_RVV_12x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 12] @DRAM
// )
void gemm_RVV_12x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (12) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (12) + 8],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
}

// gemm_RVV_12x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 12] @DRAM
// )
void gemm_RVV_12x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (12) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (12) + 8],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
}

// gemm_RVV_1x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 1] @DRAM
// )
void gemm_RVV_1x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (12)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
}

// gemm_RVV_1x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 1] @DRAM
// )
void gemm_RVV_1x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (12)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
}

// gemm_RVV_1x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 1] @DRAM
// )
void gemm_RVV_1x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (12)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
}

// gemm_RVV_1x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 1] @DRAM
// )
void gemm_RVV_1x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (12)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
}

// gemm_RVV_2x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 2] @DRAM
// )
void gemm_RVV_2x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (12)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
}

// gemm_RVV_2x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 2] @DRAM
// )
void gemm_RVV_2x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (12)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
}

// gemm_RVV_2x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 2] @DRAM
// )
void gemm_RVV_2x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (12)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
}

// gemm_RVV_2x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 2] @DRAM
// )
void gemm_RVV_2x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (12)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
}

// gemm_RVV_3x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 3] @DRAM
// )
void gemm_RVV_3x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (12)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
}

// gemm_RVV_3x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 3] @DRAM
// )
void gemm_RVV_3x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (12)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
}

// gemm_RVV_3x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 3] @DRAM
// )
void gemm_RVV_3x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (12)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
}

// gemm_RVV_3x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 3] @DRAM
// )
void gemm_RVV_3x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (12)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
}

// gemm_RVV_4x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 4] @DRAM
// )
void gemm_RVV_4x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
}

// gemm_RVV_4x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 4] @DRAM
// )
void gemm_RVV_4x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
}

// gemm_RVV_4x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 4] @DRAM
// )
void gemm_RVV_4x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
}

// gemm_RVV_4x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 4] @DRAM
// )
void gemm_RVV_4x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
}

// gemm_RVV_5x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 5] @DRAM
// )
void gemm_RVV_5x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (12) + 4],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(1));
}

// gemm_RVV_5x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 5] @DRAM
// )
void gemm_RVV_5x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (12) + 4],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(1));
}

// gemm_RVV_5x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 5] @DRAM
// )
void gemm_RVV_5x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (12) + 4],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(1));
}

// gemm_RVV_5x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 5] @DRAM
// )
void gemm_RVV_5x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (12) + 4],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(1));
}

// gemm_RVV_6x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 6] @DRAM
// )
void gemm_RVV_6x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (12) + 4],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(2));
}

// gemm_RVV_6x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 6] @DRAM
// )
void gemm_RVV_6x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (12) + 4],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(2));
}

// gemm_RVV_6x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 6] @DRAM
// )
void gemm_RVV_6x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (12) + 4],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(2));
}

// gemm_RVV_6x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 6] @DRAM
// )
void gemm_RVV_6x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (12) + 4],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(2));
}

// gemm_RVV_7x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 7] @DRAM
// )
void gemm_RVV_7x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (12) + 4],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(3));
}

// gemm_RVV_7x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 7] @DRAM
// )
void gemm_RVV_7x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (12) + 4],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(3));
}

// gemm_RVV_7x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 7] @DRAM
// )
void gemm_RVV_7x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (12) + 4],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(3));
}

// gemm_RVV_7x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 7] @DRAM
// )
void gemm_RVV_7x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (12) + 4],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(3));
}

// gemm_RVV_8x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 8] @DRAM
// )
void gemm_RVV_8x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (12) + 4],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
}

// gemm_RVV_8x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 8] @DRAM
// )
void gemm_RVV_8x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (12) + 4],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
}

// gemm_RVV_8x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 8] @DRAM
// )
void gemm_RVV_8x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (12) + 4],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
}

// gemm_RVV_8x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 8] @DRAM
// )
void gemm_RVV_8x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (12) + 4],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
}

// gemm_RVV_9x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 9] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 9] @DRAM
// )
void gemm_RVV_9x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (12) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (12) + 8],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(1));
}

// gemm_RVV_9x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 9] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 9] @DRAM
// )
void gemm_RVV_9x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
C_regt_0 = __riscv_vle32_v_f32m1(&C[8],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (12) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (12) + 8],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(1));
}

// gemm_RVV_9x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 9] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 9] @DRAM
// )
void gemm_RVV_9x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (12) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (12) + 8],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_regt_1,(1));
}

// gemm_RVV_9x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 9] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 9] @DRAM
// )
void gemm_RVV_9x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
C_regt_0 = __riscv_vle32_v_f32m1(&C[8],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 8],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (12)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (12) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (12) + 8],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (2)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (2) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_regt_1,(1));
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
