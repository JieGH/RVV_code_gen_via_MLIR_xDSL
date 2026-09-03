#include "kernels_RVV_20x2_fp32.h"



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
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(2));
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
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[8],(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(2));
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
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B[(k) * (ldb) + 1], A_reg_0,(4));
  C_reg_1_1 = __riscv_vfmacc_vf_f32m1(C_reg_1_1, B[(k) * (ldb) + 1], A_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B[(k) * (ldb) + 1], A_regt,(2));
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
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[8],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 8],(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B[(k) * (ldb) + 1], A_reg_0,(4));
  C_reg_1_1 = __riscv_vfmacc_vf_f32m1(C_reg_1_1, B[(k) * (ldb) + 1], A_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B[(k) * (ldb) + 1], A_regt,(2));
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
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(3));
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
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[8],(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(3));
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
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B[(k) * (ldb) + 1], A_reg_0,(4));
  C_reg_1_1 = __riscv_vfmacc_vf_f32m1(C_reg_1_1, B[(k) * (ldb) + 1], A_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B[(k) * (ldb) + 1], A_regt,(3));
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
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[8],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 8],(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B[(k) * (ldb) + 1], A_reg_0,(4));
  C_reg_1_1 = __riscv_vfmacc_vf_f32m1(C_reg_1_1, B[(k) * (ldb) + 1], A_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B[(k) * (ldb) + 1], A_regt,(3));
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
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
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
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
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
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B[(k) * (ldb) + 1], A_reg_0,(4));
  C_reg_1_1 = __riscv_vfmacc_vf_f32m1(C_reg_1_1, B[(k) * (ldb) + 1], A_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vf_f32m1(C_reg_1_2, B[(k) * (ldb) + 1], A_reg_2,(4));
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
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B[(k) * (ldb) + 1], A_reg_0,(4));
  C_reg_1_1 = __riscv_vfmacc_vf_f32m1(C_reg_1_1, B[(k) * (ldb) + 1], A_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vf_f32m1(C_reg_1_2, B[(k) * (ldb) + 1], A_reg_2,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
}

// gemm_RVV_13x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 13] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 13] @DRAM
// )
void gemm_RVV_13x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 12],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(1));
}

// gemm_RVV_13x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 13] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 13] @DRAM
// )
void gemm_RVV_13x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[12],(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 12],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(1));
}

// gemm_RVV_13x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 13] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 13] @DRAM
// )
void gemm_RVV_13x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
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
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 12],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B[(k) * (ldb) + 1], A_reg_0,(4));
  C_reg_1_1 = __riscv_vfmacc_vf_f32m1(C_reg_1_1, B[(k) * (ldb) + 1], A_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vf_f32m1(C_reg_1_2, B[(k) * (ldb) + 1], A_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B[(k) * (ldb) + 1], A_regt,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_regt_1,(1));
}

// gemm_RVV_13x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 13] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 13] @DRAM
// )
void gemm_RVV_13x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
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
C_regt_0 = __riscv_vle32_v_f32m1(&C[12],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 12],(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 12],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B[(k) * (ldb) + 1], A_reg_0,(4));
  C_reg_1_1 = __riscv_vfmacc_vf_f32m1(C_reg_1_1, B[(k) * (ldb) + 1], A_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vf_f32m1(C_reg_1_2, B[(k) * (ldb) + 1], A_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B[(k) * (ldb) + 1], A_regt,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_regt_1,(1));
}

// gemm_RVV_14x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 14] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 14] @DRAM
// )
void gemm_RVV_14x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 12],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(2));
}

// gemm_RVV_14x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 14] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 14] @DRAM
// )
void gemm_RVV_14x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[12],(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 12],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(2));
}

// gemm_RVV_14x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 14] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 14] @DRAM
// )
void gemm_RVV_14x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
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
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 12],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B[(k) * (ldb) + 1], A_reg_0,(4));
  C_reg_1_1 = __riscv_vfmacc_vf_f32m1(C_reg_1_1, B[(k) * (ldb) + 1], A_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vf_f32m1(C_reg_1_2, B[(k) * (ldb) + 1], A_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B[(k) * (ldb) + 1], A_regt,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_regt_1,(2));
}

// gemm_RVV_14x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 14] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 14] @DRAM
// )
void gemm_RVV_14x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
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
C_regt_0 = __riscv_vle32_v_f32m1(&C[12],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 12],(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 12],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B[(k) * (ldb) + 1], A_reg_0,(4));
  C_reg_1_1 = __riscv_vfmacc_vf_f32m1(C_reg_1_1, B[(k) * (ldb) + 1], A_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vf_f32m1(C_reg_1_2, B[(k) * (ldb) + 1], A_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B[(k) * (ldb) + 1], A_regt,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_regt_1,(2));
}

// gemm_RVV_15x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 15] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 15] @DRAM
// )
void gemm_RVV_15x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 12],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(3));
}

// gemm_RVV_15x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 15] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 15] @DRAM
// )
void gemm_RVV_15x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[12],(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 12],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(3));
}

// gemm_RVV_15x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 15] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 15] @DRAM
// )
void gemm_RVV_15x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
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
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 12],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B[(k) * (ldb) + 1], A_reg_0,(4));
  C_reg_1_1 = __riscv_vfmacc_vf_f32m1(C_reg_1_1, B[(k) * (ldb) + 1], A_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vf_f32m1(C_reg_1_2, B[(k) * (ldb) + 1], A_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B[(k) * (ldb) + 1], A_regt,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_regt_1,(3));
}

// gemm_RVV_15x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 15] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 15] @DRAM
// )
void gemm_RVV_15x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
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
C_regt_0 = __riscv_vle32_v_f32m1(&C[12],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 12],(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 12],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B[(k) * (ldb) + 1], A_reg_0,(4));
  C_reg_1_1 = __riscv_vfmacc_vf_f32m1(C_reg_1_1, B[(k) * (ldb) + 1], A_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vf_f32m1(C_reg_1_2, B[(k) * (ldb) + 1], A_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B[(k) * (ldb) + 1], A_regt,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_regt_1,(3));
}

// gemm_RVV_16x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 16] @DRAM
// )
void gemm_RVV_16x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 12],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
  C_reg_0_3 = __riscv_vfmacc_vf_f32m1(C_reg_0_3, B[(k) * (ldb)], A_reg_3,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
}

// gemm_RVV_16x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 16] @DRAM
// )
void gemm_RVV_16x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 12],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
  C_reg_0_3 = __riscv_vfmacc_vf_f32m1(C_reg_0_3, B[(k) * (ldb)], A_reg_3,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
}

// gemm_RVV_16x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 16] @DRAM
// )
void gemm_RVV_16x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
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
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 12],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
  C_reg_0_3 = __riscv_vfmacc_vf_f32m1(C_reg_0_3, B[(k) * (ldb)], A_reg_3,(4));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B[(k) * (ldb) + 1], A_reg_0,(4));
  C_reg_1_1 = __riscv_vfmacc_vf_f32m1(C_reg_1_1, B[(k) * (ldb) + 1], A_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vf_f32m1(C_reg_1_2, B[(k) * (ldb) + 1], A_reg_2,(4));
  C_reg_1_3 = __riscv_vfmacc_vf_f32m1(C_reg_1_3, B[(k) * (ldb) + 1], A_reg_3,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
}

// gemm_RVV_16x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 16] @DRAM
// )
void gemm_RVV_16x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C[ldc + 12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 12],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
  C_reg_0_3 = __riscv_vfmacc_vf_f32m1(C_reg_0_3, B[(k) * (ldb)], A_reg_3,(4));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B[(k) * (ldb) + 1], A_reg_0,(4));
  C_reg_1_1 = __riscv_vfmacc_vf_f32m1(C_reg_1_1, B[(k) * (ldb) + 1], A_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vf_f32m1(C_reg_1_2, B[(k) * (ldb) + 1], A_reg_2,(4));
  C_reg_1_3 = __riscv_vfmacc_vf_f32m1(C_reg_1_3, B[(k) * (ldb) + 1], A_reg_3,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
}

// gemm_RVV_17x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 17] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 17] @DRAM
// )
void gemm_RVV_17x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 16],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
  C_reg_0_3 = __riscv_vfmacc_vf_f32m1(C_reg_0_3, B[(k) * (ldb)], A_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(1));
}

// gemm_RVV_17x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 17] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 17] @DRAM
// )
void gemm_RVV_17x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[16],(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 16],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
  C_reg_0_3 = __riscv_vfmacc_vf_f32m1(C_reg_0_3, B[(k) * (ldb)], A_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(1));
}

// gemm_RVV_17x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 17] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 17] @DRAM
// )
void gemm_RVV_17x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
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
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 16],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
  C_reg_0_3 = __riscv_vfmacc_vf_f32m1(C_reg_0_3, B[(k) * (ldb)], A_reg_3,(4));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B[(k) * (ldb) + 1], A_reg_0,(4));
  C_reg_1_1 = __riscv_vfmacc_vf_f32m1(C_reg_1_1, B[(k) * (ldb) + 1], A_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vf_f32m1(C_reg_1_2, B[(k) * (ldb) + 1], A_reg_2,(4));
  C_reg_1_3 = __riscv_vfmacc_vf_f32m1(C_reg_1_3, B[(k) * (ldb) + 1], A_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B[(k) * (ldb) + 1], A_regt,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_regt_1,(1));
}

// gemm_RVV_17x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 17] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 17] @DRAM
// )
void gemm_RVV_17x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
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
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C[ldc + 12],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[16],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 16],(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 16],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
  C_reg_0_3 = __riscv_vfmacc_vf_f32m1(C_reg_0_3, B[(k) * (ldb)], A_reg_3,(4));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B[(k) * (ldb) + 1], A_reg_0,(4));
  C_reg_1_1 = __riscv_vfmacc_vf_f32m1(C_reg_1_1, B[(k) * (ldb) + 1], A_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vf_f32m1(C_reg_1_2, B[(k) * (ldb) + 1], A_reg_2,(4));
  C_reg_1_3 = __riscv_vfmacc_vf_f32m1(C_reg_1_3, B[(k) * (ldb) + 1], A_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B[(k) * (ldb) + 1], A_regt,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_regt_1,(1));
}

// gemm_RVV_18x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 18] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 18] @DRAM
// )
void gemm_RVV_18x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 16],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
  C_reg_0_3 = __riscv_vfmacc_vf_f32m1(C_reg_0_3, B[(k) * (ldb)], A_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(2));
}

// gemm_RVV_18x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 18] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 18] @DRAM
// )
void gemm_RVV_18x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[16],(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 16],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
  C_reg_0_3 = __riscv_vfmacc_vf_f32m1(C_reg_0_3, B[(k) * (ldb)], A_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(2));
}

// gemm_RVV_18x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 18] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 18] @DRAM
// )
void gemm_RVV_18x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
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
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 16],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
  C_reg_0_3 = __riscv_vfmacc_vf_f32m1(C_reg_0_3, B[(k) * (ldb)], A_reg_3,(4));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B[(k) * (ldb) + 1], A_reg_0,(4));
  C_reg_1_1 = __riscv_vfmacc_vf_f32m1(C_reg_1_1, B[(k) * (ldb) + 1], A_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vf_f32m1(C_reg_1_2, B[(k) * (ldb) + 1], A_reg_2,(4));
  C_reg_1_3 = __riscv_vfmacc_vf_f32m1(C_reg_1_3, B[(k) * (ldb) + 1], A_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B[(k) * (ldb) + 1], A_regt,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_regt_1,(2));
}

// gemm_RVV_18x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 18] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 18] @DRAM
// )
void gemm_RVV_18x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
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
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C[ldc + 12],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[16],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 16],(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 16],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
  C_reg_0_3 = __riscv_vfmacc_vf_f32m1(C_reg_0_3, B[(k) * (ldb)], A_reg_3,(4));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B[(k) * (ldb) + 1], A_reg_0,(4));
  C_reg_1_1 = __riscv_vfmacc_vf_f32m1(C_reg_1_1, B[(k) * (ldb) + 1], A_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vf_f32m1(C_reg_1_2, B[(k) * (ldb) + 1], A_reg_2,(4));
  C_reg_1_3 = __riscv_vfmacc_vf_f32m1(C_reg_1_3, B[(k) * (ldb) + 1], A_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B[(k) * (ldb) + 1], A_regt,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_regt_1,(2));
}

// gemm_RVV_19x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 19] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 19] @DRAM
// )
void gemm_RVV_19x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 16],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
  C_reg_0_3 = __riscv_vfmacc_vf_f32m1(C_reg_0_3, B[(k) * (ldb)], A_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(3));
}

// gemm_RVV_19x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 19] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 19] @DRAM
// )
void gemm_RVV_19x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[16],(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 16],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
  C_reg_0_3 = __riscv_vfmacc_vf_f32m1(C_reg_0_3, B[(k) * (ldb)], A_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(3));
}

// gemm_RVV_19x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 19] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 19] @DRAM
// )
void gemm_RVV_19x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
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
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 16],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
  C_reg_0_3 = __riscv_vfmacc_vf_f32m1(C_reg_0_3, B[(k) * (ldb)], A_reg_3,(4));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B[(k) * (ldb) + 1], A_reg_0,(4));
  C_reg_1_1 = __riscv_vfmacc_vf_f32m1(C_reg_1_1, B[(k) * (ldb) + 1], A_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vf_f32m1(C_reg_1_2, B[(k) * (ldb) + 1], A_reg_2,(4));
  C_reg_1_3 = __riscv_vfmacc_vf_f32m1(C_reg_1_3, B[(k) * (ldb) + 1], A_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B[(k) * (ldb) + 1], A_regt,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_regt_1,(3));
}

// gemm_RVV_19x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 19] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 19] @DRAM
// )
void gemm_RVV_19x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
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
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C[ldc + 12],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[16],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 16],(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 16],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
  C_reg_0_3 = __riscv_vfmacc_vf_f32m1(C_reg_0_3, B[(k) * (ldb)], A_reg_3,(4));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B[(k) * (ldb) + 1], A_reg_0,(4));
  C_reg_1_1 = __riscv_vfmacc_vf_f32m1(C_reg_1_1, B[(k) * (ldb) + 1], A_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vf_f32m1(C_reg_1_2, B[(k) * (ldb) + 1], A_reg_2,(4));
  C_reg_1_3 = __riscv_vfmacc_vf_f32m1(C_reg_1_3, B[(k) * (ldb) + 1], A_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B[(k) * (ldb) + 1], A_regt,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_regt_1,(3));
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
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B[(k) * (ldb)], A_reg,(1));
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
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B[(k) * (ldb)], A_reg,(1));
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
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B[(k) * (ldb)], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f32m1(C_reg_1, B[(k) * (ldb) + 1], A_reg,(1));
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
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B[(k) * (ldb)], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f32m1(C_reg_1, B[(k) * (ldb) + 1], A_reg,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
}

// gemm_RVV_20x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 20] @DRAM
// )
void gemm_RVV_20x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
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
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 16],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
  C_reg_0_3 = __riscv_vfmacc_vf_f32m1(C_reg_0_3, B[(k) * (ldb)], A_reg_3,(4));
  C_reg_0_4 = __riscv_vfmacc_vf_f32m1(C_reg_0_4, B[(k) * (ldb)], A_reg_4,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
}

// gemm_RVV_20x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 20] @DRAM
// )
void gemm_RVV_20x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_0_4 = __riscv_vle32_v_f32m1(&C[16],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 16],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
  C_reg_0_3 = __riscv_vfmacc_vf_f32m1(C_reg_0_3, B[(k) * (ldb)], A_reg_3,(4));
  C_reg_0_4 = __riscv_vfmacc_vf_f32m1(C_reg_0_4, B[(k) * (ldb)], A_reg_4,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
}

// gemm_RVV_20x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 20] @DRAM
// )
void gemm_RVV_20x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
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
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 16],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
  C_reg_0_3 = __riscv_vfmacc_vf_f32m1(C_reg_0_3, B[(k) * (ldb)], A_reg_3,(4));
  C_reg_0_4 = __riscv_vfmacc_vf_f32m1(C_reg_0_4, B[(k) * (ldb)], A_reg_4,(4));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B[(k) * (ldb) + 1], A_reg_0,(4));
  C_reg_1_1 = __riscv_vfmacc_vf_f32m1(C_reg_1_1, B[(k) * (ldb) + 1], A_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vf_f32m1(C_reg_1_2, B[(k) * (ldb) + 1], A_reg_2,(4));
  C_reg_1_3 = __riscv_vfmacc_vf_f32m1(C_reg_1_3, B[(k) * (ldb) + 1], A_reg_3,(4));
  C_reg_1_4 = __riscv_vfmacc_vf_f32m1(C_reg_1_4, B[(k) * (ldb) + 1], A_reg_4,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_reg_1_4,(4));
}

// gemm_RVV_20x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 20] @DRAM
// )
void gemm_RVV_20x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
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
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_0_4 = __riscv_vle32_v_f32m1(&C[16],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C[ldc + 12],(4));
C_reg_1_4 = __riscv_vle32_v_f32m1(&C[ldc + 16],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 16],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_0_2 = __riscv_vfmacc_vf_f32m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(4));
  C_reg_0_3 = __riscv_vfmacc_vf_f32m1(C_reg_0_3, B[(k) * (ldb)], A_reg_3,(4));
  C_reg_0_4 = __riscv_vfmacc_vf_f32m1(C_reg_0_4, B[(k) * (ldb)], A_reg_4,(4));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B[(k) * (ldb) + 1], A_reg_0,(4));
  C_reg_1_1 = __riscv_vfmacc_vf_f32m1(C_reg_1_1, B[(k) * (ldb) + 1], A_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vf_f32m1(C_reg_1_2, B[(k) * (ldb) + 1], A_reg_2,(4));
  C_reg_1_3 = __riscv_vfmacc_vf_f32m1(C_reg_1_3, B[(k) * (ldb) + 1], A_reg_3,(4));
  C_reg_1_4 = __riscv_vfmacc_vf_f32m1(C_reg_1_4, B[(k) * (ldb) + 1], A_reg_4,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_reg_1_4,(4));
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
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B[(k) * (ldb)], A_reg,(2));
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
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B[(k) * (ldb)], A_reg,(2));
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
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B[(k) * (ldb)], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f32m1(C_reg_1, B[(k) * (ldb) + 1], A_reg,(2));
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
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B[(k) * (ldb)], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f32m1(C_reg_1, B[(k) * (ldb) + 1], A_reg,(2));
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
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B[(k) * (ldb)], A_reg,(3));
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
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B[(k) * (ldb)], A_reg,(3));
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
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B[(k) * (ldb)], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f32m1(C_reg_1, B[(k) * (ldb) + 1], A_reg,(3));
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
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B[(k) * (ldb)], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f32m1(C_reg_1, B[(k) * (ldb) + 1], A_reg,(3));
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
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B[(k) * (ldb)], A_reg,(4));
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
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B[(k) * (ldb)], A_reg,(4));
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
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B[(k) * (ldb)], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f32m1(C_reg_1, B[(k) * (ldb) + 1], A_reg,(4));
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
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B[(k) * (ldb)], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f32m1(C_reg_1, B[(k) * (ldb) + 1], A_reg,(4));
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
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(1));
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
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(1));
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
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B[(k) * (ldb) + 1], A_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B[(k) * (ldb) + 1], A_regt,(1));
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
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B[(k) * (ldb) + 1], A_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B[(k) * (ldb) + 1], A_regt,(1));
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
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(2));
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
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(2));
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
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B[(k) * (ldb) + 1], A_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B[(k) * (ldb) + 1], A_regt,(2));
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
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B[(k) * (ldb) + 1], A_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B[(k) * (ldb) + 1], A_regt,(2));
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
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(3));
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
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(3));
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
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B[(k) * (ldb) + 1], A_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B[(k) * (ldb) + 1], A_regt,(3));
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
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B[(k) * (ldb) + 1], A_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B[(k) * (ldb) + 1], A_regt,(3));
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
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
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
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
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
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B[(k) * (ldb) + 1], A_reg_0,(4));
  C_reg_1_1 = __riscv_vfmacc_vf_f32m1(C_reg_1_1, B[(k) * (ldb) + 1], A_reg_1,(4));
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
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B[(k) * (ldb) + 1], A_reg_0,(4));
  C_reg_1_1 = __riscv_vfmacc_vf_f32m1(C_reg_1_1, B[(k) * (ldb) + 1], A_reg_1,(4));
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
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(1));
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
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[8],(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(1));
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
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B[(k) * (ldb) + 1], A_reg_0,(4));
  C_reg_1_1 = __riscv_vfmacc_vf_f32m1(C_reg_1_1, B[(k) * (ldb) + 1], A_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B[(k) * (ldb) + 1], A_regt,(1));
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
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[8],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 8],(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(4));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B[(k) * (ldb) + 1], A_reg_0,(4));
  C_reg_1_1 = __riscv_vfmacc_vf_f32m1(C_reg_1_1, B[(k) * (ldb) + 1], A_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B[(k) * (ldb)], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B[(k) * (ldb) + 1], A_regt,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_regt_1,(1));
}


/* relying on the following instruction..."
rvv_broadcast_4xf32_0(dst,vl)
{dst_data} = __riscv_vfmv_v_f_f32m1(0.0f,{vl});
*/

/* relying on the following instruction..."
rvv_vfmacc_4xf32_1xf32(dst,lhs,rhs,vl)
{dst_data} = __riscv_vfmacc_vf_f32m1({dst_data}, {rhs_data}, {lhs_data},{vl});
*/

/* relying on the following instruction..."
rvv_vld_4xf32(dst,src,vl)
{dst_data} = __riscv_vle32_v_f32m1(&{src_data},{vl});
*/

/* relying on the following instruction..."
rvv_vst_4xf32(dst,src,vl)
__riscv_vse32_v_f32m1(&{dst_data}, {src_data},{vl});
*/
