#include "kernels_RVV_8x10_fp32.h"



#include <stdio.h>
#include <stdlib.h>

#include <riscv_vector.h>


// gemm_RVV_1x10_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 1] @DRAM
// )
void gemm_RVV_1x10_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_8 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_9 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(1));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(1));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 9],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(1));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(1));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(1));
}

// gemm_RVV_1x10_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 1] @DRAM
// )
void gemm_RVV_1x10_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(1));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(1));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(1));
C_reg_8 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(1));
C_reg_9 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(1));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(1));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 9],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(1));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(1));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(1));
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
}

// gemm_RVV_1x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 1] @DRAM
// )
void gemm_RVV_1x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
}

// gemm_RVV_1x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 1] @DRAM
// )
void gemm_RVV_1x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
}

// gemm_RVV_1x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 1] @DRAM
// )
void gemm_RVV_1x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
}

// gemm_RVV_1x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 1] @DRAM
// )
void gemm_RVV_1x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
}

// gemm_RVV_1x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 1] @DRAM
// )
void gemm_RVV_1x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
}

// gemm_RVV_1x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 1] @DRAM
// )
void gemm_RVV_1x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
}

// gemm_RVV_1x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 1] @DRAM
// )
void gemm_RVV_1x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
}

// gemm_RVV_1x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 1] @DRAM
// )
void gemm_RVV_1x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
}

// gemm_RVV_1x7_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 1] @DRAM
// )
void gemm_RVV_1x7_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
}

// gemm_RVV_1x7_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 1] @DRAM
// )
void gemm_RVV_1x7_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(1));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
}

// gemm_RVV_1x8_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 1] @DRAM
// )
void gemm_RVV_1x8_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
}

// gemm_RVV_1x8_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 1] @DRAM
// )
void gemm_RVV_1x8_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(1));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(1));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
}

// gemm_RVV_1x9_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 1] @DRAM
// )
void gemm_RVV_1x9_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_8 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(1));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(1));
}

// gemm_RVV_1x9_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 1] @DRAM
// )
void gemm_RVV_1x9_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(1));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(1));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(1));
C_reg_8 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(1));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(1));
}

// gemm_RVV_2x10_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 2] @DRAM
// )
void gemm_RVV_2x10_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_8 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_9 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(2));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(2));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 9],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(2));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(2));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(2));
}

// gemm_RVV_2x10_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 2] @DRAM
// )
void gemm_RVV_2x10_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(2));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(2));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(2));
C_reg_8 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(2));
C_reg_9 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(2));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(2));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 9],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(2));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(2));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(2));
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
}

// gemm_RVV_2x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 2] @DRAM
// )
void gemm_RVV_2x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
}

// gemm_RVV_2x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 2] @DRAM
// )
void gemm_RVV_2x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
}

// gemm_RVV_2x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 2] @DRAM
// )
void gemm_RVV_2x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
}

// gemm_RVV_2x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 2] @DRAM
// )
void gemm_RVV_2x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
}

// gemm_RVV_2x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 2] @DRAM
// )
void gemm_RVV_2x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
}

// gemm_RVV_2x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 2] @DRAM
// )
void gemm_RVV_2x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
}

// gemm_RVV_2x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 2] @DRAM
// )
void gemm_RVV_2x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
}

// gemm_RVV_2x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 2] @DRAM
// )
void gemm_RVV_2x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
}

// gemm_RVV_2x7_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 2] @DRAM
// )
void gemm_RVV_2x7_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
}

// gemm_RVV_2x7_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 2] @DRAM
// )
void gemm_RVV_2x7_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(2));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
}

// gemm_RVV_2x8_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 2] @DRAM
// )
void gemm_RVV_2x8_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
}

// gemm_RVV_2x8_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 2] @DRAM
// )
void gemm_RVV_2x8_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(2));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(2));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
}

// gemm_RVV_2x9_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 2] @DRAM
// )
void gemm_RVV_2x9_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_8 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(2));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(2));
}

// gemm_RVV_2x9_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 2] @DRAM
// )
void gemm_RVV_2x9_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(2));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(2));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(2));
C_reg_8 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(2));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(2));
}

// gemm_RVV_3x10_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 3] @DRAM
// )
void gemm_RVV_3x10_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_8 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_9 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(3));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(3));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 9],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(3));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(3));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(3));
}

// gemm_RVV_3x10_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 3] @DRAM
// )
void gemm_RVV_3x10_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(3));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(3));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(3));
C_reg_8 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(3));
C_reg_9 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(3));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(3));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 9],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(3));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(3));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(3));
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
}

// gemm_RVV_3x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 3] @DRAM
// )
void gemm_RVV_3x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
}

// gemm_RVV_3x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 3] @DRAM
// )
void gemm_RVV_3x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
}

// gemm_RVV_3x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 3] @DRAM
// )
void gemm_RVV_3x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
}

// gemm_RVV_3x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 3] @DRAM
// )
void gemm_RVV_3x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
}

// gemm_RVV_3x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 3] @DRAM
// )
void gemm_RVV_3x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
}

// gemm_RVV_3x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 3] @DRAM
// )
void gemm_RVV_3x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
}

// gemm_RVV_3x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 3] @DRAM
// )
void gemm_RVV_3x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
}

// gemm_RVV_3x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 3] @DRAM
// )
void gemm_RVV_3x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
}

// gemm_RVV_3x7_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 3] @DRAM
// )
void gemm_RVV_3x7_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
}

// gemm_RVV_3x7_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 3] @DRAM
// )
void gemm_RVV_3x7_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(3));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
}

// gemm_RVV_3x8_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 3] @DRAM
// )
void gemm_RVV_3x8_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
}

// gemm_RVV_3x8_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 3] @DRAM
// )
void gemm_RVV_3x8_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(3));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(3));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
}

// gemm_RVV_3x9_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 3] @DRAM
// )
void gemm_RVV_3x9_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_8 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(3));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(3));
}

// gemm_RVV_3x9_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 3] @DRAM
// )
void gemm_RVV_3x9_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(3));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(3));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(3));
C_reg_8 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(3));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(3));
}

// gemm_RVV_4x10_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 4] @DRAM
// )
void gemm_RVV_4x10_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_8 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_9 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(4));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 9],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(4));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(4));
}

// gemm_RVV_4x10_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 4] @DRAM
// )
void gemm_RVV_4x10_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(4));
C_reg_8 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(4));
C_reg_9 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(4));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 9],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(4));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(4));
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
}

// gemm_RVV_4x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 4] @DRAM
// )
void gemm_RVV_4x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
}

// gemm_RVV_4x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 4] @DRAM
// )
void gemm_RVV_4x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
}

// gemm_RVV_4x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 4] @DRAM
// )
void gemm_RVV_4x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
}

// gemm_RVV_4x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 4] @DRAM
// )
void gemm_RVV_4x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
}

// gemm_RVV_4x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 4] @DRAM
// )
void gemm_RVV_4x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
}

// gemm_RVV_4x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 4] @DRAM
// )
void gemm_RVV_4x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
}

// gemm_RVV_4x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 4] @DRAM
// )
void gemm_RVV_4x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
}

// gemm_RVV_4x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 4] @DRAM
// )
void gemm_RVV_4x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
}

// gemm_RVV_4x7_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 4] @DRAM
// )
void gemm_RVV_4x7_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
}

// gemm_RVV_4x7_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 4] @DRAM
// )
void gemm_RVV_4x7_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
}

// gemm_RVV_4x8_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 4] @DRAM
// )
void gemm_RVV_4x8_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
}

// gemm_RVV_4x8_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 4] @DRAM
// )
void gemm_RVV_4x8_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
}

// gemm_RVV_4x9_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 4] @DRAM
// )
void gemm_RVV_4x9_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_8 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(4));
}

// gemm_RVV_4x9_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 4] @DRAM
// )
void gemm_RVV_4x9_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(4));
C_reg_8 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(4));
}

// gemm_RVV_5x10_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 5] @DRAM
// )
void gemm_RVV_5x10_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_regt_6;
vfloat32m1_t C_regt_7;
vfloat32m1_t C_regt_8;
vfloat32m1_t C_regt_9;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_6_0;
vfloat32m1_t C_reg_7_0;
vfloat32m1_t C_reg_8_0;
vfloat32m1_t C_reg_9_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_5 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_6 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_7 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_8 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_9 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_7_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_8_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_9_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(1));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(1));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 9],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(4));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 9],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_6_0 = __riscv_vfmacc_vv_f32m1(C_reg_6_0, A_reg_0, B_reg_6,(4));
  C_reg_7_0 = __riscv_vfmacc_vv_f32m1(C_reg_7_0, A_reg_0, B_reg_7,(4));
  C_reg_8_0 = __riscv_vfmacc_vv_f32m1(C_reg_8_0, A_reg_0, B_reg_8,(4));
  C_reg_9_0 = __riscv_vfmacc_vv_f32m1(C_reg_9_0, A_reg_0, B_reg_9,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(1));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(1));
  C_regt_6 = __riscv_vfmacc_vv_f32m1(C_regt_6, A_regt, B_reg_6,(1));
  C_regt_7 = __riscv_vfmacc_vv_f32m1(C_regt_7, A_regt, B_reg_7,(1));
  C_regt_8 = __riscv_vfmacc_vv_f32m1(C_regt_8, A_regt, B_reg_8,(1));
  C_regt_9 = __riscv_vfmacc_vv_f32m1(C_regt_9, A_regt, B_reg_9,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6_0,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7_0,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8_0,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_regt_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc) + 4], C_regt_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc) + 4], C_regt_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc) + 4], C_regt_8,(1));
__riscv_vse32_v_f32m1(&C[(9) * (ldc) + 4], C_regt_9,(1));
}

// gemm_RVV_5x10_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 5] @DRAM
// )
void gemm_RVV_5x10_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_regt_6;
vfloat32m1_t C_regt_7;
vfloat32m1_t C_regt_8;
vfloat32m1_t C_regt_9;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_6_0;
vfloat32m1_t C_reg_7_0;
vfloat32m1_t C_reg_8_0;
vfloat32m1_t C_reg_9_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(1));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(1));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(1));
C_regt_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc) + 4],(1));
C_regt_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc) + 4],(1));
C_regt_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc) + 4],(1));
C_regt_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc) + 4],(1));
C_regt_8 = __riscv_vle32_v_f32m1(&C[(8) * (ldc) + 4],(1));
C_regt_9 = __riscv_vle32_v_f32m1(&C[(9) * (ldc) + 4],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_6_0 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
C_reg_7_0 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(4));
C_reg_8_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(4));
C_reg_9_0 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(1));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(1));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 9],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(4));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 9],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_6_0 = __riscv_vfmacc_vv_f32m1(C_reg_6_0, A_reg_0, B_reg_6,(4));
  C_reg_7_0 = __riscv_vfmacc_vv_f32m1(C_reg_7_0, A_reg_0, B_reg_7,(4));
  C_reg_8_0 = __riscv_vfmacc_vv_f32m1(C_reg_8_0, A_reg_0, B_reg_8,(4));
  C_reg_9_0 = __riscv_vfmacc_vv_f32m1(C_reg_9_0, A_reg_0, B_reg_9,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(1));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(1));
  C_regt_6 = __riscv_vfmacc_vv_f32m1(C_regt_6, A_regt, B_reg_6,(1));
  C_regt_7 = __riscv_vfmacc_vv_f32m1(C_regt_7, A_regt, B_reg_7,(1));
  C_regt_8 = __riscv_vfmacc_vv_f32m1(C_regt_8, A_regt, B_reg_8,(1));
  C_regt_9 = __riscv_vfmacc_vv_f32m1(C_regt_9, A_regt, B_reg_9,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6_0,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7_0,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8_0,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_regt_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc) + 4], C_regt_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc) + 4], C_regt_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc) + 4], C_regt_8,(1));
__riscv_vse32_v_f32m1(&C[(9) * (ldc) + 4], C_regt_9,(1));
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(1));
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(1));
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(1));
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(1));
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

// gemm_RVV_5x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 5] @DRAM
// )
void gemm_RVV_5x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(1));
}

// gemm_RVV_5x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 5] @DRAM
// )
void gemm_RVV_5x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(1));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(1));
}

// gemm_RVV_5x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 5] @DRAM
// )
void gemm_RVV_5x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(1));
}

// gemm_RVV_5x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 5] @DRAM
// )
void gemm_RVV_5x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(1));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(1));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(1));
}

// gemm_RVV_5x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 5] @DRAM
// )
void gemm_RVV_5x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(1));
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
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(1));
}

// gemm_RVV_5x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 5] @DRAM
// )
void gemm_RVV_5x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(1));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(1));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(1));
C_regt_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc) + 4],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(1));
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
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(1));
}

// gemm_RVV_5x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 5] @DRAM
// )
void gemm_RVV_5x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(1));
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
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_regt_5,(1));
}

// gemm_RVV_5x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 5] @DRAM
// )
void gemm_RVV_5x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(1));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(1));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(1));
C_regt_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc) + 4],(1));
C_regt_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc) + 4],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(1));
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
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_regt_5,(1));
}

// gemm_RVV_5x7_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 5] @DRAM
// )
void gemm_RVV_5x7_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t B_reg_6;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_regt_6;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_6_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_5 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_6 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_6_0 = __riscv_vfmacc_vv_f32m1(C_reg_6_0, A_reg_0, B_reg_6,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(1));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(1));
  C_regt_6 = __riscv_vfmacc_vv_f32m1(C_regt_6, A_regt, B_reg_6,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_regt_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc) + 4], C_regt_6,(1));
}

// gemm_RVV_5x7_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 5] @DRAM
// )
void gemm_RVV_5x7_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t B_reg_6;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_regt_6;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_6_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(1));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(1));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(1));
C_regt_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc) + 4],(1));
C_regt_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc) + 4],(1));
C_regt_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc) + 4],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_6_0 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_6_0 = __riscv_vfmacc_vv_f32m1(C_reg_6_0, A_reg_0, B_reg_6,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(1));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(1));
  C_regt_6 = __riscv_vfmacc_vv_f32m1(C_regt_6, A_regt, B_reg_6,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_regt_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc) + 4], C_regt_6,(1));
}

// gemm_RVV_5x8_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 5] @DRAM
// )
void gemm_RVV_5x8_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_regt_6;
vfloat32m1_t C_regt_7;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_6_0;
vfloat32m1_t C_reg_7_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_5 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_6 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_7 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_7_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_6_0 = __riscv_vfmacc_vv_f32m1(C_reg_6_0, A_reg_0, B_reg_6,(4));
  C_reg_7_0 = __riscv_vfmacc_vv_f32m1(C_reg_7_0, A_reg_0, B_reg_7,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(1));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(1));
  C_regt_6 = __riscv_vfmacc_vv_f32m1(C_regt_6, A_regt, B_reg_6,(1));
  C_regt_7 = __riscv_vfmacc_vv_f32m1(C_regt_7, A_regt, B_reg_7,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6_0,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_regt_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc) + 4], C_regt_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc) + 4], C_regt_7,(1));
}

// gemm_RVV_5x8_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 5] @DRAM
// )
void gemm_RVV_5x8_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_regt_6;
vfloat32m1_t C_regt_7;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_6_0;
vfloat32m1_t C_reg_7_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(1));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(1));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(1));
C_regt_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc) + 4],(1));
C_regt_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc) + 4],(1));
C_regt_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc) + 4],(1));
C_regt_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc) + 4],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_6_0 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
C_reg_7_0 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_6_0 = __riscv_vfmacc_vv_f32m1(C_reg_6_0, A_reg_0, B_reg_6,(4));
  C_reg_7_0 = __riscv_vfmacc_vv_f32m1(C_reg_7_0, A_reg_0, B_reg_7,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(1));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(1));
  C_regt_6 = __riscv_vfmacc_vv_f32m1(C_regt_6, A_regt, B_reg_6,(1));
  C_regt_7 = __riscv_vfmacc_vv_f32m1(C_regt_7, A_regt, B_reg_7,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6_0,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_regt_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc) + 4], C_regt_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc) + 4], C_regt_7,(1));
}

// gemm_RVV_5x9_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 5] @DRAM
// )
void gemm_RVV_5x9_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_regt_6;
vfloat32m1_t C_regt_7;
vfloat32m1_t C_regt_8;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_6_0;
vfloat32m1_t C_reg_7_0;
vfloat32m1_t C_reg_8_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_5 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_6 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_7 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_8 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_7_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_8_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(1));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_6_0 = __riscv_vfmacc_vv_f32m1(C_reg_6_0, A_reg_0, B_reg_6,(4));
  C_reg_7_0 = __riscv_vfmacc_vv_f32m1(C_reg_7_0, A_reg_0, B_reg_7,(4));
  C_reg_8_0 = __riscv_vfmacc_vv_f32m1(C_reg_8_0, A_reg_0, B_reg_8,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(1));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(1));
  C_regt_6 = __riscv_vfmacc_vv_f32m1(C_regt_6, A_regt, B_reg_6,(1));
  C_regt_7 = __riscv_vfmacc_vv_f32m1(C_regt_7, A_regt, B_reg_7,(1));
  C_regt_8 = __riscv_vfmacc_vv_f32m1(C_regt_8, A_regt, B_reg_8,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6_0,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7_0,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_regt_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc) + 4], C_regt_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc) + 4], C_regt_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc) + 4], C_regt_8,(1));
}

// gemm_RVV_5x9_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 5] @DRAM
// )
void gemm_RVV_5x9_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_regt_6;
vfloat32m1_t C_regt_7;
vfloat32m1_t C_regt_8;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_6_0;
vfloat32m1_t C_reg_7_0;
vfloat32m1_t C_reg_8_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(1));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(1));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(1));
C_regt_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc) + 4],(1));
C_regt_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc) + 4],(1));
C_regt_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc) + 4],(1));
C_regt_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc) + 4],(1));
C_regt_8 = __riscv_vle32_v_f32m1(&C[(8) * (ldc) + 4],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_6_0 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
C_reg_7_0 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(4));
C_reg_8_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(1));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(1));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_6_0 = __riscv_vfmacc_vv_f32m1(C_reg_6_0, A_reg_0, B_reg_6,(4));
  C_reg_7_0 = __riscv_vfmacc_vv_f32m1(C_reg_7_0, A_reg_0, B_reg_7,(4));
  C_reg_8_0 = __riscv_vfmacc_vv_f32m1(C_reg_8_0, A_reg_0, B_reg_8,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(1));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(1));
  C_regt_6 = __riscv_vfmacc_vv_f32m1(C_regt_6, A_regt, B_reg_6,(1));
  C_regt_7 = __riscv_vfmacc_vv_f32m1(C_regt_7, A_regt, B_reg_7,(1));
  C_regt_8 = __riscv_vfmacc_vv_f32m1(C_regt_8, A_regt, B_reg_8,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6_0,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7_0,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_regt_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc) + 4], C_regt_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc) + 4], C_regt_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc) + 4], C_regt_8,(1));
}

// gemm_RVV_6x10_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 6] @DRAM
// )
void gemm_RVV_6x10_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_regt_6;
vfloat32m1_t C_regt_7;
vfloat32m1_t C_regt_8;
vfloat32m1_t C_regt_9;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_6_0;
vfloat32m1_t C_reg_7_0;
vfloat32m1_t C_reg_8_0;
vfloat32m1_t C_reg_9_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_5 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_6 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_7 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_8 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_9 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_7_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_8_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_9_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(2));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(2));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 9],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(4));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 9],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_6_0 = __riscv_vfmacc_vv_f32m1(C_reg_6_0, A_reg_0, B_reg_6,(4));
  C_reg_7_0 = __riscv_vfmacc_vv_f32m1(C_reg_7_0, A_reg_0, B_reg_7,(4));
  C_reg_8_0 = __riscv_vfmacc_vv_f32m1(C_reg_8_0, A_reg_0, B_reg_8,(4));
  C_reg_9_0 = __riscv_vfmacc_vv_f32m1(C_reg_9_0, A_reg_0, B_reg_9,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(2));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(2));
  C_regt_6 = __riscv_vfmacc_vv_f32m1(C_regt_6, A_regt, B_reg_6,(2));
  C_regt_7 = __riscv_vfmacc_vv_f32m1(C_regt_7, A_regt, B_reg_7,(2));
  C_regt_8 = __riscv_vfmacc_vv_f32m1(C_regt_8, A_regt, B_reg_8,(2));
  C_regt_9 = __riscv_vfmacc_vv_f32m1(C_regt_9, A_regt, B_reg_9,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6_0,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7_0,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8_0,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_regt_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc) + 4], C_regt_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc) + 4], C_regt_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc) + 4], C_regt_8,(2));
__riscv_vse32_v_f32m1(&C[(9) * (ldc) + 4], C_regt_9,(2));
}

// gemm_RVV_6x10_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 6] @DRAM
// )
void gemm_RVV_6x10_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_regt_6;
vfloat32m1_t C_regt_7;
vfloat32m1_t C_regt_8;
vfloat32m1_t C_regt_9;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_6_0;
vfloat32m1_t C_reg_7_0;
vfloat32m1_t C_reg_8_0;
vfloat32m1_t C_reg_9_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(2));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(2));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(2));
C_regt_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc) + 4],(2));
C_regt_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc) + 4],(2));
C_regt_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc) + 4],(2));
C_regt_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc) + 4],(2));
C_regt_8 = __riscv_vle32_v_f32m1(&C[(8) * (ldc) + 4],(2));
C_regt_9 = __riscv_vle32_v_f32m1(&C[(9) * (ldc) + 4],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_6_0 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
C_reg_7_0 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(4));
C_reg_8_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(4));
C_reg_9_0 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(2));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(2));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 9],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(4));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 9],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_6_0 = __riscv_vfmacc_vv_f32m1(C_reg_6_0, A_reg_0, B_reg_6,(4));
  C_reg_7_0 = __riscv_vfmacc_vv_f32m1(C_reg_7_0, A_reg_0, B_reg_7,(4));
  C_reg_8_0 = __riscv_vfmacc_vv_f32m1(C_reg_8_0, A_reg_0, B_reg_8,(4));
  C_reg_9_0 = __riscv_vfmacc_vv_f32m1(C_reg_9_0, A_reg_0, B_reg_9,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(2));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(2));
  C_regt_6 = __riscv_vfmacc_vv_f32m1(C_regt_6, A_regt, B_reg_6,(2));
  C_regt_7 = __riscv_vfmacc_vv_f32m1(C_regt_7, A_regt, B_reg_7,(2));
  C_regt_8 = __riscv_vfmacc_vv_f32m1(C_regt_8, A_regt, B_reg_8,(2));
  C_regt_9 = __riscv_vfmacc_vv_f32m1(C_regt_9, A_regt, B_reg_9,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6_0,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7_0,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8_0,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_regt_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc) + 4], C_regt_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc) + 4], C_regt_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc) + 4], C_regt_8,(2));
__riscv_vse32_v_f32m1(&C[(9) * (ldc) + 4], C_regt_9,(2));
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(2));
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(2));
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(2));
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(2));
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

// gemm_RVV_6x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 6] @DRAM
// )
void gemm_RVV_6x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(2));
}

// gemm_RVV_6x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 6] @DRAM
// )
void gemm_RVV_6x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(2));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(2));
}

// gemm_RVV_6x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 6] @DRAM
// )
void gemm_RVV_6x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(2));
}

// gemm_RVV_6x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 6] @DRAM
// )
void gemm_RVV_6x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(2));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(2));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(2));
}

// gemm_RVV_6x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 6] @DRAM
// )
void gemm_RVV_6x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(2));
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
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(2));
}

// gemm_RVV_6x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 6] @DRAM
// )
void gemm_RVV_6x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(2));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(2));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(2));
C_regt_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc) + 4],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(2));
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
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(2));
}

// gemm_RVV_6x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 6] @DRAM
// )
void gemm_RVV_6x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(2));
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
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_regt_5,(2));
}

// gemm_RVV_6x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 6] @DRAM
// )
void gemm_RVV_6x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(2));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(2));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(2));
C_regt_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc) + 4],(2));
C_regt_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc) + 4],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(2));
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
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_regt_5,(2));
}

// gemm_RVV_6x7_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 6] @DRAM
// )
void gemm_RVV_6x7_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t B_reg_6;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_regt_6;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_6_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_5 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_6 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_6_0 = __riscv_vfmacc_vv_f32m1(C_reg_6_0, A_reg_0, B_reg_6,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(2));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(2));
  C_regt_6 = __riscv_vfmacc_vv_f32m1(C_regt_6, A_regt, B_reg_6,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_regt_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc) + 4], C_regt_6,(2));
}

// gemm_RVV_6x7_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 6] @DRAM
// )
void gemm_RVV_6x7_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t B_reg_6;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_regt_6;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_6_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(2));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(2));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(2));
C_regt_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc) + 4],(2));
C_regt_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc) + 4],(2));
C_regt_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc) + 4],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_6_0 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_6_0 = __riscv_vfmacc_vv_f32m1(C_reg_6_0, A_reg_0, B_reg_6,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(2));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(2));
  C_regt_6 = __riscv_vfmacc_vv_f32m1(C_regt_6, A_regt, B_reg_6,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_regt_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc) + 4], C_regt_6,(2));
}

// gemm_RVV_6x8_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 6] @DRAM
// )
void gemm_RVV_6x8_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_regt_6;
vfloat32m1_t C_regt_7;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_6_0;
vfloat32m1_t C_reg_7_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_5 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_6 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_7 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_7_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_6_0 = __riscv_vfmacc_vv_f32m1(C_reg_6_0, A_reg_0, B_reg_6,(4));
  C_reg_7_0 = __riscv_vfmacc_vv_f32m1(C_reg_7_0, A_reg_0, B_reg_7,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(2));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(2));
  C_regt_6 = __riscv_vfmacc_vv_f32m1(C_regt_6, A_regt, B_reg_6,(2));
  C_regt_7 = __riscv_vfmacc_vv_f32m1(C_regt_7, A_regt, B_reg_7,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6_0,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_regt_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc) + 4], C_regt_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc) + 4], C_regt_7,(2));
}

// gemm_RVV_6x8_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 6] @DRAM
// )
void gemm_RVV_6x8_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_regt_6;
vfloat32m1_t C_regt_7;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_6_0;
vfloat32m1_t C_reg_7_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(2));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(2));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(2));
C_regt_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc) + 4],(2));
C_regt_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc) + 4],(2));
C_regt_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc) + 4],(2));
C_regt_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc) + 4],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_6_0 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
C_reg_7_0 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_6_0 = __riscv_vfmacc_vv_f32m1(C_reg_6_0, A_reg_0, B_reg_6,(4));
  C_reg_7_0 = __riscv_vfmacc_vv_f32m1(C_reg_7_0, A_reg_0, B_reg_7,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(2));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(2));
  C_regt_6 = __riscv_vfmacc_vv_f32m1(C_regt_6, A_regt, B_reg_6,(2));
  C_regt_7 = __riscv_vfmacc_vv_f32m1(C_regt_7, A_regt, B_reg_7,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6_0,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_regt_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc) + 4], C_regt_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc) + 4], C_regt_7,(2));
}

// gemm_RVV_6x9_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 6] @DRAM
// )
void gemm_RVV_6x9_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_regt_6;
vfloat32m1_t C_regt_7;
vfloat32m1_t C_regt_8;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_6_0;
vfloat32m1_t C_reg_7_0;
vfloat32m1_t C_reg_8_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_5 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_6 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_7 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_8 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_7_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_8_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(2));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_6_0 = __riscv_vfmacc_vv_f32m1(C_reg_6_0, A_reg_0, B_reg_6,(4));
  C_reg_7_0 = __riscv_vfmacc_vv_f32m1(C_reg_7_0, A_reg_0, B_reg_7,(4));
  C_reg_8_0 = __riscv_vfmacc_vv_f32m1(C_reg_8_0, A_reg_0, B_reg_8,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(2));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(2));
  C_regt_6 = __riscv_vfmacc_vv_f32m1(C_regt_6, A_regt, B_reg_6,(2));
  C_regt_7 = __riscv_vfmacc_vv_f32m1(C_regt_7, A_regt, B_reg_7,(2));
  C_regt_8 = __riscv_vfmacc_vv_f32m1(C_regt_8, A_regt, B_reg_8,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6_0,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7_0,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_regt_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc) + 4], C_regt_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc) + 4], C_regt_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc) + 4], C_regt_8,(2));
}

// gemm_RVV_6x9_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 6] @DRAM
// )
void gemm_RVV_6x9_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_regt_6;
vfloat32m1_t C_regt_7;
vfloat32m1_t C_regt_8;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_6_0;
vfloat32m1_t C_reg_7_0;
vfloat32m1_t C_reg_8_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(2));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(2));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(2));
C_regt_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc) + 4],(2));
C_regt_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc) + 4],(2));
C_regt_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc) + 4],(2));
C_regt_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc) + 4],(2));
C_regt_8 = __riscv_vle32_v_f32m1(&C[(8) * (ldc) + 4],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_6_0 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
C_reg_7_0 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(4));
C_reg_8_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(2));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(2));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_6_0 = __riscv_vfmacc_vv_f32m1(C_reg_6_0, A_reg_0, B_reg_6,(4));
  C_reg_7_0 = __riscv_vfmacc_vv_f32m1(C_reg_7_0, A_reg_0, B_reg_7,(4));
  C_reg_8_0 = __riscv_vfmacc_vv_f32m1(C_reg_8_0, A_reg_0, B_reg_8,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(2));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(2));
  C_regt_6 = __riscv_vfmacc_vv_f32m1(C_regt_6, A_regt, B_reg_6,(2));
  C_regt_7 = __riscv_vfmacc_vv_f32m1(C_regt_7, A_regt, B_reg_7,(2));
  C_regt_8 = __riscv_vfmacc_vv_f32m1(C_regt_8, A_regt, B_reg_8,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6_0,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7_0,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_regt_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc) + 4], C_regt_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc) + 4], C_regt_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc) + 4], C_regt_8,(2));
}

// gemm_RVV_7x10_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 7] @DRAM
// )
void gemm_RVV_7x10_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_regt_6;
vfloat32m1_t C_regt_7;
vfloat32m1_t C_regt_8;
vfloat32m1_t C_regt_9;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_6_0;
vfloat32m1_t C_reg_7_0;
vfloat32m1_t C_reg_8_0;
vfloat32m1_t C_reg_9_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_5 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_6 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_7 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_8 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_9 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_7_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_8_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_9_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(3));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(3));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 9],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(4));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 9],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_6_0 = __riscv_vfmacc_vv_f32m1(C_reg_6_0, A_reg_0, B_reg_6,(4));
  C_reg_7_0 = __riscv_vfmacc_vv_f32m1(C_reg_7_0, A_reg_0, B_reg_7,(4));
  C_reg_8_0 = __riscv_vfmacc_vv_f32m1(C_reg_8_0, A_reg_0, B_reg_8,(4));
  C_reg_9_0 = __riscv_vfmacc_vv_f32m1(C_reg_9_0, A_reg_0, B_reg_9,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(3));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(3));
  C_regt_6 = __riscv_vfmacc_vv_f32m1(C_regt_6, A_regt, B_reg_6,(3));
  C_regt_7 = __riscv_vfmacc_vv_f32m1(C_regt_7, A_regt, B_reg_7,(3));
  C_regt_8 = __riscv_vfmacc_vv_f32m1(C_regt_8, A_regt, B_reg_8,(3));
  C_regt_9 = __riscv_vfmacc_vv_f32m1(C_regt_9, A_regt, B_reg_9,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6_0,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7_0,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8_0,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_regt_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc) + 4], C_regt_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc) + 4], C_regt_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc) + 4], C_regt_8,(3));
__riscv_vse32_v_f32m1(&C[(9) * (ldc) + 4], C_regt_9,(3));
}

// gemm_RVV_7x10_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 7] @DRAM
// )
void gemm_RVV_7x10_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_regt_6;
vfloat32m1_t C_regt_7;
vfloat32m1_t C_regt_8;
vfloat32m1_t C_regt_9;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_6_0;
vfloat32m1_t C_reg_7_0;
vfloat32m1_t C_reg_8_0;
vfloat32m1_t C_reg_9_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(3));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(3));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(3));
C_regt_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc) + 4],(3));
C_regt_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc) + 4],(3));
C_regt_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc) + 4],(3));
C_regt_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc) + 4],(3));
C_regt_8 = __riscv_vle32_v_f32m1(&C[(8) * (ldc) + 4],(3));
C_regt_9 = __riscv_vle32_v_f32m1(&C[(9) * (ldc) + 4],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_6_0 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
C_reg_7_0 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(4));
C_reg_8_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(4));
C_reg_9_0 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(3));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(3));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 9],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(4));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 9],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_6_0 = __riscv_vfmacc_vv_f32m1(C_reg_6_0, A_reg_0, B_reg_6,(4));
  C_reg_7_0 = __riscv_vfmacc_vv_f32m1(C_reg_7_0, A_reg_0, B_reg_7,(4));
  C_reg_8_0 = __riscv_vfmacc_vv_f32m1(C_reg_8_0, A_reg_0, B_reg_8,(4));
  C_reg_9_0 = __riscv_vfmacc_vv_f32m1(C_reg_9_0, A_reg_0, B_reg_9,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(3));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(3));
  C_regt_6 = __riscv_vfmacc_vv_f32m1(C_regt_6, A_regt, B_reg_6,(3));
  C_regt_7 = __riscv_vfmacc_vv_f32m1(C_regt_7, A_regt, B_reg_7,(3));
  C_regt_8 = __riscv_vfmacc_vv_f32m1(C_regt_8, A_regt, B_reg_8,(3));
  C_regt_9 = __riscv_vfmacc_vv_f32m1(C_regt_9, A_regt, B_reg_9,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6_0,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7_0,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8_0,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_regt_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc) + 4], C_regt_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc) + 4], C_regt_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc) + 4], C_regt_8,(3));
__riscv_vse32_v_f32m1(&C[(9) * (ldc) + 4], C_regt_9,(3));
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(3));
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(3));
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(3));
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(3));
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

// gemm_RVV_7x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 7] @DRAM
// )
void gemm_RVV_7x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(3));
}

// gemm_RVV_7x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 7] @DRAM
// )
void gemm_RVV_7x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(3));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(3));
}

// gemm_RVV_7x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 7] @DRAM
// )
void gemm_RVV_7x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(3));
}

// gemm_RVV_7x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 7] @DRAM
// )
void gemm_RVV_7x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(3));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(3));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(3));
}

// gemm_RVV_7x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 7] @DRAM
// )
void gemm_RVV_7x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(3));
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
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(3));
}

// gemm_RVV_7x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 7] @DRAM
// )
void gemm_RVV_7x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(3));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(3));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(3));
C_regt_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc) + 4],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(3));
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
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(3));
}

// gemm_RVV_7x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 7] @DRAM
// )
void gemm_RVV_7x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(3));
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
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_regt_5,(3));
}

// gemm_RVV_7x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 7] @DRAM
// )
void gemm_RVV_7x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(3));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(3));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(3));
C_regt_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc) + 4],(3));
C_regt_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc) + 4],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(3));
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
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_regt_5,(3));
}

// gemm_RVV_7x7_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 7] @DRAM
// )
void gemm_RVV_7x7_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t B_reg_6;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_regt_6;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_6_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_5 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_6 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_6_0 = __riscv_vfmacc_vv_f32m1(C_reg_6_0, A_reg_0, B_reg_6,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(3));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(3));
  C_regt_6 = __riscv_vfmacc_vv_f32m1(C_regt_6, A_regt, B_reg_6,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_regt_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc) + 4], C_regt_6,(3));
}

// gemm_RVV_7x7_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 7] @DRAM
// )
void gemm_RVV_7x7_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t B_reg_6;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_regt_6;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_6_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(3));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(3));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(3));
C_regt_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc) + 4],(3));
C_regt_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc) + 4],(3));
C_regt_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc) + 4],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_6_0 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_6_0 = __riscv_vfmacc_vv_f32m1(C_reg_6_0, A_reg_0, B_reg_6,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(3));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(3));
  C_regt_6 = __riscv_vfmacc_vv_f32m1(C_regt_6, A_regt, B_reg_6,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_regt_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc) + 4], C_regt_6,(3));
}

// gemm_RVV_7x8_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 7] @DRAM
// )
void gemm_RVV_7x8_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_regt_6;
vfloat32m1_t C_regt_7;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_6_0;
vfloat32m1_t C_reg_7_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_5 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_6 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_7 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_7_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_6_0 = __riscv_vfmacc_vv_f32m1(C_reg_6_0, A_reg_0, B_reg_6,(4));
  C_reg_7_0 = __riscv_vfmacc_vv_f32m1(C_reg_7_0, A_reg_0, B_reg_7,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(3));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(3));
  C_regt_6 = __riscv_vfmacc_vv_f32m1(C_regt_6, A_regt, B_reg_6,(3));
  C_regt_7 = __riscv_vfmacc_vv_f32m1(C_regt_7, A_regt, B_reg_7,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6_0,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_regt_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc) + 4], C_regt_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc) + 4], C_regt_7,(3));
}

// gemm_RVV_7x8_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 7] @DRAM
// )
void gemm_RVV_7x8_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_regt_6;
vfloat32m1_t C_regt_7;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_6_0;
vfloat32m1_t C_reg_7_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(3));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(3));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(3));
C_regt_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc) + 4],(3));
C_regt_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc) + 4],(3));
C_regt_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc) + 4],(3));
C_regt_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc) + 4],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_6_0 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
C_reg_7_0 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_6_0 = __riscv_vfmacc_vv_f32m1(C_reg_6_0, A_reg_0, B_reg_6,(4));
  C_reg_7_0 = __riscv_vfmacc_vv_f32m1(C_reg_7_0, A_reg_0, B_reg_7,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(3));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(3));
  C_regt_6 = __riscv_vfmacc_vv_f32m1(C_regt_6, A_regt, B_reg_6,(3));
  C_regt_7 = __riscv_vfmacc_vv_f32m1(C_regt_7, A_regt, B_reg_7,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6_0,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_regt_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc) + 4], C_regt_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc) + 4], C_regt_7,(3));
}

// gemm_RVV_7x9_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 7] @DRAM
// )
void gemm_RVV_7x9_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_regt_6;
vfloat32m1_t C_regt_7;
vfloat32m1_t C_regt_8;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_6_0;
vfloat32m1_t C_reg_7_0;
vfloat32m1_t C_reg_8_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_5 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_6 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_7 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_8 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_7_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_8_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(3));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_6_0 = __riscv_vfmacc_vv_f32m1(C_reg_6_0, A_reg_0, B_reg_6,(4));
  C_reg_7_0 = __riscv_vfmacc_vv_f32m1(C_reg_7_0, A_reg_0, B_reg_7,(4));
  C_reg_8_0 = __riscv_vfmacc_vv_f32m1(C_reg_8_0, A_reg_0, B_reg_8,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(3));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(3));
  C_regt_6 = __riscv_vfmacc_vv_f32m1(C_regt_6, A_regt, B_reg_6,(3));
  C_regt_7 = __riscv_vfmacc_vv_f32m1(C_regt_7, A_regt, B_reg_7,(3));
  C_regt_8 = __riscv_vfmacc_vv_f32m1(C_regt_8, A_regt, B_reg_8,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6_0,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7_0,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_regt_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc) + 4], C_regt_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc) + 4], C_regt_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc) + 4], C_regt_8,(3));
}

// gemm_RVV_7x9_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 7] @DRAM
// )
void gemm_RVV_7x9_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_regt_4;
vfloat32m1_t C_regt_5;
vfloat32m1_t C_regt_6;
vfloat32m1_t C_regt_7;
vfloat32m1_t C_regt_8;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_4_0;
vfloat32m1_t C_reg_5_0;
vfloat32m1_t C_reg_6_0;
vfloat32m1_t C_reg_7_0;
vfloat32m1_t C_reg_8_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(3));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(3));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(3));
C_regt_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc) + 4],(3));
C_regt_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc) + 4],(3));
C_regt_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc) + 4],(3));
C_regt_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc) + 4],(3));
C_regt_8 = __riscv_vle32_v_f32m1(&C[(8) * (ldc) + 4],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_6_0 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
C_reg_7_0 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(4));
C_reg_8_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(3));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(3));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_4_0 = __riscv_vfmacc_vv_f32m1(C_reg_4_0, A_reg_0, B_reg_4,(4));
  C_reg_5_0 = __riscv_vfmacc_vv_f32m1(C_reg_5_0, A_reg_0, B_reg_5,(4));
  C_reg_6_0 = __riscv_vfmacc_vv_f32m1(C_reg_6_0, A_reg_0, B_reg_6,(4));
  C_reg_7_0 = __riscv_vfmacc_vv_f32m1(C_reg_7_0, A_reg_0, B_reg_7,(4));
  C_reg_8_0 = __riscv_vfmacc_vv_f32m1(C_reg_8_0, A_reg_0, B_reg_8,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
  C_regt_4 = __riscv_vfmacc_vv_f32m1(C_regt_4, A_regt, B_reg_4,(3));
  C_regt_5 = __riscv_vfmacc_vv_f32m1(C_regt_5, A_regt, B_reg_5,(3));
  C_regt_6 = __riscv_vfmacc_vv_f32m1(C_regt_6, A_regt, B_reg_6,(3));
  C_regt_7 = __riscv_vfmacc_vv_f32m1(C_regt_7, A_regt, B_reg_7,(3));
  C_regt_8 = __riscv_vfmacc_vv_f32m1(C_regt_8, A_regt, B_reg_8,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6_0,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7_0,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_regt_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_regt_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc) + 4], C_regt_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc) + 4], C_regt_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc) + 4], C_regt_8,(3));
}

// gemm_RVV_8x10_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 8] @DRAM
// )
void gemm_RVV_8x10_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
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
vfloat32m1_t C_reg_6_0;
vfloat32m1_t C_reg_6_1;
vfloat32m1_t C_reg_7_0;
vfloat32m1_t C_reg_7_1;
vfloat32m1_t C_reg_8_0;
vfloat32m1_t C_reg_8_1;
vfloat32m1_t C_reg_9_0;
vfloat32m1_t C_reg_9_1;
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
C_reg_6_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_7_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_7_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_8_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_8_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_9_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_9_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(4));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 9],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
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
  C_reg_6_0 = __riscv_vfmacc_vv_f32m1(C_reg_6_0, A_reg_0, B_reg_6,(4));
  C_reg_6_1 = __riscv_vfmacc_vv_f32m1(C_reg_6_1, A_reg_1, B_reg_6,(4));
  C_reg_7_0 = __riscv_vfmacc_vv_f32m1(C_reg_7_0, A_reg_0, B_reg_7,(4));
  C_reg_7_1 = __riscv_vfmacc_vv_f32m1(C_reg_7_1, A_reg_1, B_reg_7,(4));
  C_reg_8_0 = __riscv_vfmacc_vv_f32m1(C_reg_8_0, A_reg_0, B_reg_8,(4));
  C_reg_8_1 = __riscv_vfmacc_vv_f32m1(C_reg_8_1, A_reg_1, B_reg_8,(4));
  C_reg_9_0 = __riscv_vfmacc_vv_f32m1(C_reg_9_0, A_reg_0, B_reg_9,(4));
  C_reg_9_1 = __riscv_vfmacc_vv_f32m1(C_reg_9_1, A_reg_1, B_reg_9,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_reg_5_1,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6_0,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc) + 4], C_reg_6_1,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7_0,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc) + 4], C_reg_7_1,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8_0,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc) + 4], C_reg_8_1,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9_0,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc) + 4], C_reg_9_1,(4));
}

// gemm_RVV_8x10_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 8] @DRAM
// )
void gemm_RVV_8x10_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
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
vfloat32m1_t C_reg_6_0;
vfloat32m1_t C_reg_6_1;
vfloat32m1_t C_reg_7_0;
vfloat32m1_t C_reg_7_1;
vfloat32m1_t C_reg_8_0;
vfloat32m1_t C_reg_8_1;
vfloat32m1_t C_reg_9_0;
vfloat32m1_t C_reg_9_1;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_4_1 = __riscv_vle32_v_f32m1(&C[(4) * (ldc) + 4],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_5_1 = __riscv_vle32_v_f32m1(&C[(5) * (ldc) + 4],(4));
C_reg_6_0 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
C_reg_6_1 = __riscv_vle32_v_f32m1(&C[(6) * (ldc) + 4],(4));
C_reg_7_0 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(4));
C_reg_7_1 = __riscv_vle32_v_f32m1(&C[(7) * (ldc) + 4],(4));
C_reg_8_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(4));
C_reg_8_1 = __riscv_vle32_v_f32m1(&C[(8) * (ldc) + 4],(4));
C_reg_9_0 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(4));
C_reg_9_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc) + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(4));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 9],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
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
  C_reg_6_0 = __riscv_vfmacc_vv_f32m1(C_reg_6_0, A_reg_0, B_reg_6,(4));
  C_reg_6_1 = __riscv_vfmacc_vv_f32m1(C_reg_6_1, A_reg_1, B_reg_6,(4));
  C_reg_7_0 = __riscv_vfmacc_vv_f32m1(C_reg_7_0, A_reg_0, B_reg_7,(4));
  C_reg_7_1 = __riscv_vfmacc_vv_f32m1(C_reg_7_1, A_reg_1, B_reg_7,(4));
  C_reg_8_0 = __riscv_vfmacc_vv_f32m1(C_reg_8_0, A_reg_0, B_reg_8,(4));
  C_reg_8_1 = __riscv_vfmacc_vv_f32m1(C_reg_8_1, A_reg_1, B_reg_8,(4));
  C_reg_9_0 = __riscv_vfmacc_vv_f32m1(C_reg_9_0, A_reg_0, B_reg_9,(4));
  C_reg_9_1 = __riscv_vfmacc_vv_f32m1(C_reg_9_1, A_reg_1, B_reg_9,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_reg_5_1,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6_0,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc) + 4], C_reg_6_1,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7_0,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc) + 4], C_reg_7_1,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8_0,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc) + 4], C_reg_8_1,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9_0,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc) + 4], C_reg_9_1,(4));
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
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

// gemm_RVV_8x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 8] @DRAM
// )
void gemm_RVV_8x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
}

// gemm_RVV_8x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 8] @DRAM
// )
void gemm_RVV_8x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
}

// gemm_RVV_8x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 8] @DRAM
// )
void gemm_RVV_8x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
}

// gemm_RVV_8x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 8] @DRAM
// )
void gemm_RVV_8x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
}

// gemm_RVV_8x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 8] @DRAM
// )
void gemm_RVV_8x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
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
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_reg_4_1,(4));
}

// gemm_RVV_8x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 8] @DRAM
// )
void gemm_RVV_8x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_4_1 = __riscv_vle32_v_f32m1(&C[(4) * (ldc) + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
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
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_reg_4_1,(4));
}

// gemm_RVV_8x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 8] @DRAM
// )
void gemm_RVV_8x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
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
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_reg_5_1,(4));
}

// gemm_RVV_8x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 8] @DRAM
// )
void gemm_RVV_8x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_4_1 = __riscv_vle32_v_f32m1(&C[(4) * (ldc) + 4],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_5_1 = __riscv_vle32_v_f32m1(&C[(5) * (ldc) + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
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
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_reg_5_1,(4));
}

// gemm_RVV_8x7_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 8] @DRAM
// )
void gemm_RVV_8x7_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t B_reg_6;
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
vfloat32m1_t C_reg_6_0;
vfloat32m1_t C_reg_6_1;
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
C_reg_6_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
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
  C_reg_6_0 = __riscv_vfmacc_vv_f32m1(C_reg_6_0, A_reg_0, B_reg_6,(4));
  C_reg_6_1 = __riscv_vfmacc_vv_f32m1(C_reg_6_1, A_reg_1, B_reg_6,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_reg_5_1,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6_0,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc) + 4], C_reg_6_1,(4));
}

// gemm_RVV_8x7_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 8] @DRAM
// )
void gemm_RVV_8x7_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t B_reg_6;
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
vfloat32m1_t C_reg_6_0;
vfloat32m1_t C_reg_6_1;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_4_1 = __riscv_vle32_v_f32m1(&C[(4) * (ldc) + 4],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_5_1 = __riscv_vle32_v_f32m1(&C[(5) * (ldc) + 4],(4));
C_reg_6_0 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
C_reg_6_1 = __riscv_vle32_v_f32m1(&C[(6) * (ldc) + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
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
  C_reg_6_0 = __riscv_vfmacc_vv_f32m1(C_reg_6_0, A_reg_0, B_reg_6,(4));
  C_reg_6_1 = __riscv_vfmacc_vv_f32m1(C_reg_6_1, A_reg_1, B_reg_6,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_reg_5_1,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6_0,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc) + 4], C_reg_6_1,(4));
}

// gemm_RVV_8x8_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 8] @DRAM
// )
void gemm_RVV_8x8_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
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
vfloat32m1_t C_reg_6_0;
vfloat32m1_t C_reg_6_1;
vfloat32m1_t C_reg_7_0;
vfloat32m1_t C_reg_7_1;
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
C_reg_6_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_7_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_7_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
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
  C_reg_6_0 = __riscv_vfmacc_vv_f32m1(C_reg_6_0, A_reg_0, B_reg_6,(4));
  C_reg_6_1 = __riscv_vfmacc_vv_f32m1(C_reg_6_1, A_reg_1, B_reg_6,(4));
  C_reg_7_0 = __riscv_vfmacc_vv_f32m1(C_reg_7_0, A_reg_0, B_reg_7,(4));
  C_reg_7_1 = __riscv_vfmacc_vv_f32m1(C_reg_7_1, A_reg_1, B_reg_7,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_reg_5_1,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6_0,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc) + 4], C_reg_6_1,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7_0,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc) + 4], C_reg_7_1,(4));
}

// gemm_RVV_8x8_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 8] @DRAM
// )
void gemm_RVV_8x8_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
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
vfloat32m1_t C_reg_6_0;
vfloat32m1_t C_reg_6_1;
vfloat32m1_t C_reg_7_0;
vfloat32m1_t C_reg_7_1;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_4_1 = __riscv_vle32_v_f32m1(&C[(4) * (ldc) + 4],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_5_1 = __riscv_vle32_v_f32m1(&C[(5) * (ldc) + 4],(4));
C_reg_6_0 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
C_reg_6_1 = __riscv_vle32_v_f32m1(&C[(6) * (ldc) + 4],(4));
C_reg_7_0 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(4));
C_reg_7_1 = __riscv_vle32_v_f32m1(&C[(7) * (ldc) + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
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
  C_reg_6_0 = __riscv_vfmacc_vv_f32m1(C_reg_6_0, A_reg_0, B_reg_6,(4));
  C_reg_6_1 = __riscv_vfmacc_vv_f32m1(C_reg_6_1, A_reg_1, B_reg_6,(4));
  C_reg_7_0 = __riscv_vfmacc_vv_f32m1(C_reg_7_0, A_reg_0, B_reg_7,(4));
  C_reg_7_1 = __riscv_vfmacc_vv_f32m1(C_reg_7_1, A_reg_1, B_reg_7,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_reg_5_1,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6_0,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc) + 4], C_reg_6_1,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7_0,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc) + 4], C_reg_7_1,(4));
}

// gemm_RVV_8x9_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 8] @DRAM
// )
void gemm_RVV_8x9_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
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
vfloat32m1_t C_reg_6_0;
vfloat32m1_t C_reg_6_1;
vfloat32m1_t C_reg_7_0;
vfloat32m1_t C_reg_7_1;
vfloat32m1_t C_reg_8_0;
vfloat32m1_t C_reg_8_1;
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
C_reg_6_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_7_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_7_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_8_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_8_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
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
  C_reg_6_0 = __riscv_vfmacc_vv_f32m1(C_reg_6_0, A_reg_0, B_reg_6,(4));
  C_reg_6_1 = __riscv_vfmacc_vv_f32m1(C_reg_6_1, A_reg_1, B_reg_6,(4));
  C_reg_7_0 = __riscv_vfmacc_vv_f32m1(C_reg_7_0, A_reg_0, B_reg_7,(4));
  C_reg_7_1 = __riscv_vfmacc_vv_f32m1(C_reg_7_1, A_reg_1, B_reg_7,(4));
  C_reg_8_0 = __riscv_vfmacc_vv_f32m1(C_reg_8_0, A_reg_0, B_reg_8,(4));
  C_reg_8_1 = __riscv_vfmacc_vv_f32m1(C_reg_8_1, A_reg_1, B_reg_8,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_reg_5_1,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6_0,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc) + 4], C_reg_6_1,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7_0,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc) + 4], C_reg_7_1,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8_0,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc) + 4], C_reg_8_1,(4));
}

// gemm_RVV_8x9_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 8] @DRAM
// )
void gemm_RVV_8x9_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
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
vfloat32m1_t C_reg_6_0;
vfloat32m1_t C_reg_6_1;
vfloat32m1_t C_reg_7_0;
vfloat32m1_t C_reg_7_1;
vfloat32m1_t C_reg_8_0;
vfloat32m1_t C_reg_8_1;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(4));
C_reg_4_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_4_1 = __riscv_vle32_v_f32m1(&C[(4) * (ldc) + 4],(4));
C_reg_5_0 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_5_1 = __riscv_vle32_v_f32m1(&C[(5) * (ldc) + 4],(4));
C_reg_6_0 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
C_reg_6_1 = __riscv_vle32_v_f32m1(&C[(6) * (ldc) + 4],(4));
C_reg_7_0 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(4));
C_reg_7_1 = __riscv_vle32_v_f32m1(&C[(7) * (ldc) + 4],(4));
C_reg_8_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(4));
C_reg_8_1 = __riscv_vle32_v_f32m1(&C[(8) * (ldc) + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 8],(4));
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (lda) + 4],(4));
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
  C_reg_6_0 = __riscv_vfmacc_vv_f32m1(C_reg_6_0, A_reg_0, B_reg_6,(4));
  C_reg_6_1 = __riscv_vfmacc_vv_f32m1(C_reg_6_1, A_reg_1, B_reg_6,(4));
  C_reg_7_0 = __riscv_vfmacc_vv_f32m1(C_reg_7_0, A_reg_0, B_reg_7,(4));
  C_reg_7_1 = __riscv_vfmacc_vv_f32m1(C_reg_7_1, A_reg_1, B_reg_7,(4));
  C_reg_8_0 = __riscv_vfmacc_vv_f32m1(C_reg_8_0, A_reg_0, B_reg_8,(4));
  C_reg_8_1 = __riscv_vfmacc_vv_f32m1(C_reg_8_1, A_reg_1, B_reg_8,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4_0,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc) + 4], C_reg_4_1,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc) + 4], C_reg_5_1,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6_0,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc) + 4], C_reg_6_1,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7_0,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc) + 4], C_reg_7_1,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8_0,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc) + 4], C_reg_8_1,(4));
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
