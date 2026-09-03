#include "kernels_RVV_8x14_fp32.h"



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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (1));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (1));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (1));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(1));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(1));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(1));
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(1));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(1));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(1));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(1));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (1));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (1));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (1));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(1));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(1));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(1));
}

// gemm_RVV_1x11_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 1] @DRAM
// )
void gemm_RVV_1x11_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (1));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (1));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (1));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(1));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(1));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(1));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(1));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(1));
}

// gemm_RVV_1x11_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 1] @DRAM
// )
void gemm_RVV_1x11_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(1));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(1));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(1));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(1));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(1));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (1));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (1));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (1));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(1));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(1));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(1));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(1));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(1));
}

// gemm_RVV_1x12_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 1] @DRAM
// )
void gemm_RVV_1x12_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (1));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (1));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (1));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(4));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(1));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(1));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(1));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(1));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(1));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(1));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(1));
}

// gemm_RVV_1x12_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 1] @DRAM
// )
void gemm_RVV_1x12_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(1));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(1));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(1));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(1));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(1));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(1));
C_reg2_3 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (1));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (1));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (1));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(4));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(1));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(1));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(1));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(1));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(1));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(1));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(1));
}

// gemm_RVV_1x13_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 1] @DRAM
// )
void gemm_RVV_1x13_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
vfloat32m1_t C_reg2_4;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
vfloat32m1_t B_reg2_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (1));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (1));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (1));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(5));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_reg2_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(1));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(1));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(1));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(1));
  C_reg2_4 = __riscv_vfmacc_vv_f32m1(C_reg2_4, A_reg, B_reg2_4,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(1));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(1));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(1));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(1));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_4,(1));
}

// gemm_RVV_1x13_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 1] @DRAM
// )
void gemm_RVV_1x13_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
vfloat32m1_t C_reg2_4;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(1));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(1));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(1));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(1));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(1));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(1));
C_reg2_3 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(1));
C_reg2_4 = __riscv_vle32_v_f32m1(&C[(12) * (ldc)],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
vfloat32m1_t B_reg2_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (1));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (1));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (1));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(5));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_reg2_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(1));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(1));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(1));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(1));
  C_reg2_4 = __riscv_vfmacc_vv_f32m1(C_reg2_4, A_reg, B_reg2_4,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(1));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(1));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(1));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(1));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_4,(1));
}

// gemm_RVV_1x14_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 1] @DRAM
// )
void gemm_RVV_1x14_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
vfloat32m1_t C_reg2_4;
vfloat32m1_t C_reg2_5;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_5 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
vfloat32m1_t B_reg2_4;
vfloat32m1_t B_reg2_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (1));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (1));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (1));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(6));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_reg2_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (1));
  B_reg2_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(1));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(1));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(1));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(1));
  C_reg2_4 = __riscv_vfmacc_vv_f32m1(C_reg2_4, A_reg, B_reg2_4,(1));
  C_reg2_5 = __riscv_vfmacc_vv_f32m1(C_reg2_5, A_reg, B_reg2_5,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(1));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(1));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(1));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(1));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_4,(1));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg2_5,(1));
}

// gemm_RVV_1x14_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 1] @DRAM
// )
void gemm_RVV_1x14_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
vfloat32m1_t C_reg2_4;
vfloat32m1_t C_reg2_5;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(1));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(1));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(1));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(1));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(1));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(1));
C_reg2_3 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(1));
C_reg2_4 = __riscv_vle32_v_f32m1(&C[(12) * (ldc)],(1));
C_reg2_5 = __riscv_vle32_v_f32m1(&C[(13) * (ldc)],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
vfloat32m1_t B_reg2_4;
vfloat32m1_t B_reg2_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (1));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (1));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (1));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(6));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_reg2_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (1));
  B_reg2_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(1));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(1));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(1));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(1));
  C_reg2_4 = __riscv_vfmacc_vv_f32m1(C_reg2_4, A_reg, B_reg2_4,(1));
  C_reg2_5 = __riscv_vfmacc_vv_f32m1(C_reg2_5, A_reg, B_reg2_5,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(1));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(1));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(1));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(1));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_4,(1));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg2_5,(1));
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
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (1));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (1));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (1));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (1));
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
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (1));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (1));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (1));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (1));
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
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (1));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (1));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (1));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(1));
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
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(1));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(1));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(1));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (1));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (1));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (1));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(1));
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (2));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (2));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (2));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(2));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(2));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(2));
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(2));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(2));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(2));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(2));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (2));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (2));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (2));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(2));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(2));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(2));
}

// gemm_RVV_2x11_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 2] @DRAM
// )
void gemm_RVV_2x11_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (2));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (2));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (2));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(2));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(2));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(2));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(2));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(2));
}

// gemm_RVV_2x11_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 2] @DRAM
// )
void gemm_RVV_2x11_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(2));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(2));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(2));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(2));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(2));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (2));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (2));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (2));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(2));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(2));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(2));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(2));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(2));
}

// gemm_RVV_2x12_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 2] @DRAM
// )
void gemm_RVV_2x12_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (2));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (2));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (2));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(4));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(2));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(2));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(2));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(2));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(2));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(2));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(2));
}

// gemm_RVV_2x12_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 2] @DRAM
// )
void gemm_RVV_2x12_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(2));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(2));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(2));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(2));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(2));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(2));
C_reg2_3 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (2));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (2));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (2));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(4));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(2));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(2));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(2));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(2));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(2));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(2));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(2));
}

// gemm_RVV_2x13_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 2] @DRAM
// )
void gemm_RVV_2x13_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
vfloat32m1_t C_reg2_4;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
vfloat32m1_t B_reg2_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (2));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (2));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (2));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(5));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_reg2_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(2));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(2));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(2));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(2));
  C_reg2_4 = __riscv_vfmacc_vv_f32m1(C_reg2_4, A_reg, B_reg2_4,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(2));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(2));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(2));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(2));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_4,(2));
}

// gemm_RVV_2x13_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 2] @DRAM
// )
void gemm_RVV_2x13_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
vfloat32m1_t C_reg2_4;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(2));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(2));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(2));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(2));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(2));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(2));
C_reg2_3 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(2));
C_reg2_4 = __riscv_vle32_v_f32m1(&C[(12) * (ldc)],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
vfloat32m1_t B_reg2_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (2));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (2));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (2));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(5));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_reg2_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(2));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(2));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(2));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(2));
  C_reg2_4 = __riscv_vfmacc_vv_f32m1(C_reg2_4, A_reg, B_reg2_4,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(2));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(2));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(2));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(2));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_4,(2));
}

// gemm_RVV_2x14_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 2] @DRAM
// )
void gemm_RVV_2x14_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
vfloat32m1_t C_reg2_4;
vfloat32m1_t C_reg2_5;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_5 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
vfloat32m1_t B_reg2_4;
vfloat32m1_t B_reg2_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (2));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (2));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (2));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(6));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_reg2_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (2));
  B_reg2_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(2));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(2));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(2));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(2));
  C_reg2_4 = __riscv_vfmacc_vv_f32m1(C_reg2_4, A_reg, B_reg2_4,(2));
  C_reg2_5 = __riscv_vfmacc_vv_f32m1(C_reg2_5, A_reg, B_reg2_5,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(2));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(2));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(2));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(2));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_4,(2));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg2_5,(2));
}

// gemm_RVV_2x14_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 2] @DRAM
// )
void gemm_RVV_2x14_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
vfloat32m1_t C_reg2_4;
vfloat32m1_t C_reg2_5;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(2));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(2));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(2));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(2));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(2));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(2));
C_reg2_3 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(2));
C_reg2_4 = __riscv_vle32_v_f32m1(&C[(12) * (ldc)],(2));
C_reg2_5 = __riscv_vle32_v_f32m1(&C[(13) * (ldc)],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
vfloat32m1_t B_reg2_4;
vfloat32m1_t B_reg2_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (2));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (2));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (2));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(6));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_reg2_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (2));
  B_reg2_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(2));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(2));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(2));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(2));
  C_reg2_4 = __riscv_vfmacc_vv_f32m1(C_reg2_4, A_reg, B_reg2_4,(2));
  C_reg2_5 = __riscv_vfmacc_vv_f32m1(C_reg2_5, A_reg, B_reg2_5,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(2));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(2));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(2));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(2));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_4,(2));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg2_5,(2));
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
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (2));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (2));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (2));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (2));
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
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (2));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (2));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (2));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (2));
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
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (2));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (2));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (2));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(2));
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
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(2));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(2));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(2));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (2));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (2));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (2));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(2));
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (3));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (3));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (3));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(3));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(3));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(3));
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(3));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(3));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(3));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(3));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (3));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (3));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (3));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(3));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(3));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(3));
}

// gemm_RVV_3x11_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 3] @DRAM
// )
void gemm_RVV_3x11_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (3));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (3));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (3));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(3));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(3));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(3));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(3));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(3));
}

// gemm_RVV_3x11_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 3] @DRAM
// )
void gemm_RVV_3x11_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(3));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(3));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(3));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(3));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(3));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (3));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (3));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (3));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(3));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(3));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(3));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(3));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(3));
}

// gemm_RVV_3x12_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 3] @DRAM
// )
void gemm_RVV_3x12_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (3));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (3));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (3));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(4));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(3));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(3));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(3));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(3));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(3));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(3));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(3));
}

// gemm_RVV_3x12_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 3] @DRAM
// )
void gemm_RVV_3x12_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(3));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(3));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(3));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(3));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(3));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(3));
C_reg2_3 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (3));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (3));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (3));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(4));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(3));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(3));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(3));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(3));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(3));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(3));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(3));
}

// gemm_RVV_3x13_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 3] @DRAM
// )
void gemm_RVV_3x13_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
vfloat32m1_t C_reg2_4;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
vfloat32m1_t B_reg2_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (3));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (3));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (3));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(5));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_reg2_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(3));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(3));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(3));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(3));
  C_reg2_4 = __riscv_vfmacc_vv_f32m1(C_reg2_4, A_reg, B_reg2_4,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(3));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(3));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(3));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(3));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_4,(3));
}

// gemm_RVV_3x13_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 3] @DRAM
// )
void gemm_RVV_3x13_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
vfloat32m1_t C_reg2_4;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(3));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(3));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(3));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(3));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(3));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(3));
C_reg2_3 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(3));
C_reg2_4 = __riscv_vle32_v_f32m1(&C[(12) * (ldc)],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
vfloat32m1_t B_reg2_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (3));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (3));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (3));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(5));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_reg2_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(3));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(3));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(3));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(3));
  C_reg2_4 = __riscv_vfmacc_vv_f32m1(C_reg2_4, A_reg, B_reg2_4,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(3));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(3));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(3));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(3));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_4,(3));
}

// gemm_RVV_3x14_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 3] @DRAM
// )
void gemm_RVV_3x14_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
vfloat32m1_t C_reg2_4;
vfloat32m1_t C_reg2_5;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_5 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
vfloat32m1_t B_reg2_4;
vfloat32m1_t B_reg2_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (3));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (3));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (3));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(6));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_reg2_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (3));
  B_reg2_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(3));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(3));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(3));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(3));
  C_reg2_4 = __riscv_vfmacc_vv_f32m1(C_reg2_4, A_reg, B_reg2_4,(3));
  C_reg2_5 = __riscv_vfmacc_vv_f32m1(C_reg2_5, A_reg, B_reg2_5,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(3));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(3));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(3));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(3));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_4,(3));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg2_5,(3));
}

// gemm_RVV_3x14_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 3] @DRAM
// )
void gemm_RVV_3x14_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
vfloat32m1_t C_reg2_4;
vfloat32m1_t C_reg2_5;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(3));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(3));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(3));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(3));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(3));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(3));
C_reg2_3 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(3));
C_reg2_4 = __riscv_vle32_v_f32m1(&C[(12) * (ldc)],(3));
C_reg2_5 = __riscv_vle32_v_f32m1(&C[(13) * (ldc)],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
vfloat32m1_t B_reg2_4;
vfloat32m1_t B_reg2_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (3));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (3));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (3));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(6));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_reg2_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (3));
  B_reg2_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(3));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(3));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(3));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(3));
  C_reg2_4 = __riscv_vfmacc_vv_f32m1(C_reg2_4, A_reg, B_reg2_4,(3));
  C_reg2_5 = __riscv_vfmacc_vv_f32m1(C_reg2_5, A_reg, B_reg2_5,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(3));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(3));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(3));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(3));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_4,(3));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg2_5,(3));
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
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (3));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (3));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (3));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (3));
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
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (3));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (3));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (3));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (3));
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
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (3));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (3));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (3));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(3));
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
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(3));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(3));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(3));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (3));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (3));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (3));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(3));
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (4));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (4));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (4));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(4));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(4));
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(4));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(4));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (4));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (4));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (4));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(4));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(4));
}

// gemm_RVV_4x11_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 4] @DRAM
// )
void gemm_RVV_4x11_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (4));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (4));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (4));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(4));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(4));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(4));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(4));
}

// gemm_RVV_4x11_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 4] @DRAM
// )
void gemm_RVV_4x11_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(4));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(4));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(4));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (4));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (4));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (4));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(4));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(4));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(4));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(4));
}

// gemm_RVV_4x12_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 4] @DRAM
// )
void gemm_RVV_4x12_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (4));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (4));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (4));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(4));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(4));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(4));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(4));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(4));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(4));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(4));
}

// gemm_RVV_4x12_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 4] @DRAM
// )
void gemm_RVV_4x12_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(4));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(4));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(4));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(4));
C_reg2_3 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (4));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (4));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (4));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(4));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(4));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(4));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(4));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(4));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(4));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(4));
}

// gemm_RVV_4x13_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 4] @DRAM
// )
void gemm_RVV_4x13_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
vfloat32m1_t C_reg2_4;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
vfloat32m1_t B_reg2_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (4));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (4));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (4));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(5));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_reg2_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(4));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(4));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(4));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(4));
  C_reg2_4 = __riscv_vfmacc_vv_f32m1(C_reg2_4, A_reg, B_reg2_4,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(4));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(4));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(4));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_4,(4));
}

// gemm_RVV_4x13_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 4] @DRAM
// )
void gemm_RVV_4x13_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
vfloat32m1_t C_reg2_4;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(4));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(4));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(4));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(4));
C_reg2_3 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(4));
C_reg2_4 = __riscv_vle32_v_f32m1(&C[(12) * (ldc)],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
vfloat32m1_t B_reg2_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (4));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (4));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (4));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(5));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_reg2_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(4));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(4));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(4));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(4));
  C_reg2_4 = __riscv_vfmacc_vv_f32m1(C_reg2_4, A_reg, B_reg2_4,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(4));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(4));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(4));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_4,(4));
}

// gemm_RVV_4x14_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 4] @DRAM
// )
void gemm_RVV_4x14_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
vfloat32m1_t C_reg2_4;
vfloat32m1_t C_reg2_5;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_5 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
vfloat32m1_t B_reg2_4;
vfloat32m1_t B_reg2_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (4));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (4));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (4));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(6));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_reg2_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (4));
  B_reg2_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(4));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(4));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(4));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(4));
  C_reg2_4 = __riscv_vfmacc_vv_f32m1(C_reg2_4, A_reg, B_reg2_4,(4));
  C_reg2_5 = __riscv_vfmacc_vv_f32m1(C_reg2_5, A_reg, B_reg2_5,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(4));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(4));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(4));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_4,(4));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg2_5,(4));
}

// gemm_RVV_4x14_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 4] @DRAM
// )
void gemm_RVV_4x14_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
vfloat32m1_t C_reg2_4;
vfloat32m1_t C_reg2_5;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(4));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(4));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(4));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(4));
C_reg2_3 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(4));
C_reg2_4 = __riscv_vle32_v_f32m1(&C[(12) * (ldc)],(4));
C_reg2_5 = __riscv_vle32_v_f32m1(&C[(13) * (ldc)],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
vfloat32m1_t B_reg2_4;
vfloat32m1_t B_reg2_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (4));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (4));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (4));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(6));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_reg2_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (4));
  B_reg2_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(4));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(4));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(4));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(4));
  C_reg2_4 = __riscv_vfmacc_vv_f32m1(C_reg2_4, A_reg, B_reg2_4,(4));
  C_reg2_5 = __riscv_vfmacc_vv_f32m1(C_reg2_5, A_reg, B_reg2_5,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(4));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(4));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(4));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_4,(4));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg2_5,(4));
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
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (4));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (4));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (4));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (4));
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
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (4));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (4));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (4));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (4));
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
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (4));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (4));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (4));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(4));
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
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(4));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (4));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (4));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (4));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(4));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (5));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (5));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (5));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (5));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (5));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (5));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (5));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (5));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (5));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(5));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(5));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(5));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(5));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(5));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(5));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(5));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(5));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(5));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(5));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(5));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(5));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(5));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(5));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(5));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(5));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(5));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(5));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(5));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(5));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (5));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (5));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (5));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (5));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (5));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (5));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (5));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (5));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (5));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(5));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(5));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(5));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(5));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(5));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(5));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(5));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(5));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(5));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(5));
}

// gemm_RVV_5x11_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 5] @DRAM
// )
void gemm_RVV_5x11_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (5));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (5));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (5));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (5));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (5));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (5));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (5));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (5));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (5));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (5));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(5));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(5));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(5));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(5));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(5));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(5));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(5));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(5));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(5));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(5));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(5));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(5));
}

// gemm_RVV_5x11_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 5] @DRAM
// )
void gemm_RVV_5x11_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(5));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(5));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(5));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(5));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(5));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(5));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(5));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(5));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(5));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(5));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(5));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (5));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (5));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (5));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (5));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (5));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (5));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (5));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (5));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (5));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (5));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(5));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(5));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(5));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(5));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(5));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(5));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(5));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(5));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(5));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(5));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(5));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(5));
}

// gemm_RVV_5x12_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 5] @DRAM
// )
void gemm_RVV_5x12_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (5));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (5));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (5));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (5));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (5));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (5));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (5));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (5));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(4));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (5));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (5));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (5));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(5));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(5));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(5));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(5));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(5));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(5));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(5));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(5));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(5));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(5));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(5));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(5));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(5));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(5));
}

// gemm_RVV_5x12_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 5] @DRAM
// )
void gemm_RVV_5x12_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(5));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(5));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(5));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(5));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(5));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(5));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(5));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(5));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(5));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(5));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(5));
C_reg2_3 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(5));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (5));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (5));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (5));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (5));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (5));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (5));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (5));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (5));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(4));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (5));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (5));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (5));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(5));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(5));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(5));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(5));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(5));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(5));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(5));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(5));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(5));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(5));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(5));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(5));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(5));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(5));
}

// gemm_RVV_5x13_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 5] @DRAM
// )
void gemm_RVV_5x13_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
vfloat32m1_t C_reg2_4;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg2_4 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
vfloat32m1_t B_reg2_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (5));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (5));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (5));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (5));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (5));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (5));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (5));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (5));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(5));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (5));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (5));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (5));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (5));
  B_reg2_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(5));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(5));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(5));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(5));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(5));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(5));
  C_reg2_4 = __riscv_vfmacc_vv_f32m1(C_reg2_4, A_reg, B_reg2_4,(5));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(5));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(5));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(5));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(5));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(5));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(5));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(5));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(5));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_4,(5));
}

// gemm_RVV_5x13_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 5] @DRAM
// )
void gemm_RVV_5x13_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
vfloat32m1_t C_reg2_4;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(5));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(5));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(5));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(5));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(5));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(5));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(5));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(5));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(5));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(5));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(5));
C_reg2_3 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(5));
C_reg2_4 = __riscv_vle32_v_f32m1(&C[(12) * (ldc)],(5));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
vfloat32m1_t B_reg2_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (5));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (5));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (5));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (5));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (5));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (5));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (5));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (5));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(5));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (5));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (5));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (5));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (5));
  B_reg2_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(5));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(5));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(5));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(5));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(5));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(5));
  C_reg2_4 = __riscv_vfmacc_vv_f32m1(C_reg2_4, A_reg, B_reg2_4,(5));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(5));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(5));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(5));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(5));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(5));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(5));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(5));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(5));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_4,(5));
}

// gemm_RVV_5x14_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 5] @DRAM
// )
void gemm_RVV_5x14_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
vfloat32m1_t C_reg2_4;
vfloat32m1_t C_reg2_5;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg2_4 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg2_5 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
vfloat32m1_t B_reg2_4;
vfloat32m1_t B_reg2_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (5));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (5));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (5));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (5));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (5));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (5));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (5));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (5));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(6));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (5));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (5));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (5));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (5));
  B_reg2_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (5));
  B_reg2_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(5));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(5));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(5));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(5));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(5));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(5));
  C_reg2_4 = __riscv_vfmacc_vv_f32m1(C_reg2_4, A_reg, B_reg2_4,(5));
  C_reg2_5 = __riscv_vfmacc_vv_f32m1(C_reg2_5, A_reg, B_reg2_5,(5));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(5));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(5));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(5));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(5));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(5));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(5));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(5));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(5));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_4,(5));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg2_5,(5));
}

// gemm_RVV_5x14_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 5] @DRAM
// )
void gemm_RVV_5x14_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
vfloat32m1_t C_reg2_4;
vfloat32m1_t C_reg2_5;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(5));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(5));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(5));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(5));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(5));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(5));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(5));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(5));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(5));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(5));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(5));
C_reg2_3 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(5));
C_reg2_4 = __riscv_vle32_v_f32m1(&C[(12) * (ldc)],(5));
C_reg2_5 = __riscv_vle32_v_f32m1(&C[(13) * (ldc)],(5));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
vfloat32m1_t B_reg2_4;
vfloat32m1_t B_reg2_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (5));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (5));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (5));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (5));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (5));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (5));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (5));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (5));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(6));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (5));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (5));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (5));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (5));
  B_reg2_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (5));
  B_reg2_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(5));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(5));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(5));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(5));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(5));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(5));
  C_reg2_4 = __riscv_vfmacc_vv_f32m1(C_reg2_4, A_reg, B_reg2_4,(5));
  C_reg2_5 = __riscv_vfmacc_vv_f32m1(C_reg2_5, A_reg, B_reg2_5,(5));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(5));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(5));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(5));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(5));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(5));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(5));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(5));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(5));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_4,(5));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg2_5,(5));
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
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(5));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(5));
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
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(5));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(5));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(5));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(5));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(5));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(5));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(5));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(5));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(5));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(5));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(5));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(5));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(5));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(5));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(5));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(5));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(5));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(5));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(5));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(5));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(5));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(5));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(5));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(5));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(5));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(5));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(5));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(5));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(5));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(5));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(5));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(5));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(5));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(5));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(5));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(5));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(5));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(5));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(5));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(5));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(5));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(5));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(5));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(5));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(5));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(5));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(5));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(5));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(5));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(5));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(5));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(5));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(5));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(5));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(5));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(5));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(5));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(5));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(5));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(5));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(5));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(5));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(5));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(5));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(5));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(5));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(5));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(5));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(5));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(5));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(5));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(5));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(5));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(5));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(5));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(5));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(5));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(5));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(5));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(5));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(5));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(5));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(5));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(5));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(5));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(5));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(5));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(5));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(5));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(5));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(5));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(5));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(5));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(5));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(5));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(5));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (5));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (5));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (5));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (5));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (5));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (5));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (5));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(5));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(5));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(5));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(5));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(5));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(5));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(5));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(5));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(5));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(5));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(5));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(5));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(5));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(5));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (5));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (5));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (5));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (5));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (5));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (5));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (5));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(5));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(5));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(5));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(5));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(5));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(5));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (5));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (5));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (5));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (5));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (5));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (5));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (5));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (5));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(5));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(5));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(5));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(5));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(5));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(5));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(5));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(5));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(5));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(5));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(5));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(5));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(5));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(5));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(5));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(5));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(5));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (5));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (5));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (5));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (5));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (5));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (5));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (5));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (5));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(5));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(5));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(5));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(5));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(5));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(5));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(5));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(5));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(5));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (6));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (6));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (6));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (6));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (6));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (6));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (6));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (6));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (6));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(6));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(6));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(6));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(6));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(6));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(6));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(6));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(6));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(6));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(6));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(6));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(6));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(6));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(6));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(6));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(6));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(6));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(6));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(6));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(6));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (6));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (6));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (6));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (6));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (6));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (6));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (6));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (6));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (6));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(6));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(6));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(6));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(6));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(6));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(6));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(6));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(6));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(6));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(6));
}

// gemm_RVV_6x11_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 6] @DRAM
// )
void gemm_RVV_6x11_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (6));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (6));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (6));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (6));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (6));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (6));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (6));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (6));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (6));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (6));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(6));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(6));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(6));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(6));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(6));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(6));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(6));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(6));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(6));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(6));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(6));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(6));
}

// gemm_RVV_6x11_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 6] @DRAM
// )
void gemm_RVV_6x11_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(6));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(6));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(6));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(6));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(6));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(6));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(6));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(6));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(6));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(6));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(6));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (6));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (6));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (6));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (6));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (6));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (6));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (6));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (6));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (6));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (6));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(6));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(6));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(6));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(6));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(6));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(6));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(6));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(6));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(6));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(6));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(6));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(6));
}

// gemm_RVV_6x12_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 6] @DRAM
// )
void gemm_RVV_6x12_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (6));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (6));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (6));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (6));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (6));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (6));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (6));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (6));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(4));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (6));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (6));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (6));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(6));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(6));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(6));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(6));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(6));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(6));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(6));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(6));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(6));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(6));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(6));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(6));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(6));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(6));
}

// gemm_RVV_6x12_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 6] @DRAM
// )
void gemm_RVV_6x12_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(6));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(6));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(6));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(6));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(6));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(6));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(6));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(6));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(6));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(6));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(6));
C_reg2_3 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(6));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (6));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (6));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (6));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (6));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (6));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (6));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (6));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (6));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(4));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (6));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (6));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (6));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(6));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(6));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(6));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(6));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(6));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(6));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(6));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(6));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(6));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(6));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(6));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(6));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(6));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(6));
}

// gemm_RVV_6x13_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 6] @DRAM
// )
void gemm_RVV_6x13_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
vfloat32m1_t C_reg2_4;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg2_4 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
vfloat32m1_t B_reg2_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (6));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (6));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (6));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (6));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (6));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (6));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (6));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (6));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(5));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (6));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (6));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (6));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (6));
  B_reg2_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(6));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(6));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(6));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(6));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(6));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(6));
  C_reg2_4 = __riscv_vfmacc_vv_f32m1(C_reg2_4, A_reg, B_reg2_4,(6));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(6));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(6));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(6));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(6));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(6));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(6));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(6));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(6));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_4,(6));
}

// gemm_RVV_6x13_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 6] @DRAM
// )
void gemm_RVV_6x13_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
vfloat32m1_t C_reg2_4;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(6));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(6));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(6));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(6));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(6));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(6));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(6));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(6));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(6));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(6));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(6));
C_reg2_3 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(6));
C_reg2_4 = __riscv_vle32_v_f32m1(&C[(12) * (ldc)],(6));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
vfloat32m1_t B_reg2_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (6));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (6));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (6));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (6));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (6));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (6));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (6));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (6));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(5));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (6));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (6));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (6));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (6));
  B_reg2_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(6));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(6));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(6));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(6));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(6));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(6));
  C_reg2_4 = __riscv_vfmacc_vv_f32m1(C_reg2_4, A_reg, B_reg2_4,(6));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(6));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(6));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(6));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(6));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(6));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(6));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(6));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(6));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_4,(6));
}

// gemm_RVV_6x14_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 6] @DRAM
// )
void gemm_RVV_6x14_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
vfloat32m1_t C_reg2_4;
vfloat32m1_t C_reg2_5;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg2_4 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg2_5 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
vfloat32m1_t B_reg2_4;
vfloat32m1_t B_reg2_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (6));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (6));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (6));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (6));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (6));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (6));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (6));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (6));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(6));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (6));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (6));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (6));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (6));
  B_reg2_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (6));
  B_reg2_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(6));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(6));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(6));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(6));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(6));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(6));
  C_reg2_4 = __riscv_vfmacc_vv_f32m1(C_reg2_4, A_reg, B_reg2_4,(6));
  C_reg2_5 = __riscv_vfmacc_vv_f32m1(C_reg2_5, A_reg, B_reg2_5,(6));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(6));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(6));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(6));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(6));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(6));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(6));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(6));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(6));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_4,(6));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg2_5,(6));
}

// gemm_RVV_6x14_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 6] @DRAM
// )
void gemm_RVV_6x14_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
vfloat32m1_t C_reg2_4;
vfloat32m1_t C_reg2_5;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(6));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(6));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(6));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(6));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(6));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(6));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(6));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(6));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(6));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(6));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(6));
C_reg2_3 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(6));
C_reg2_4 = __riscv_vle32_v_f32m1(&C[(12) * (ldc)],(6));
C_reg2_5 = __riscv_vle32_v_f32m1(&C[(13) * (ldc)],(6));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
vfloat32m1_t B_reg2_4;
vfloat32m1_t B_reg2_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (6));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (6));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (6));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (6));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (6));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (6));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (6));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (6));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(6));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (6));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (6));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (6));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (6));
  B_reg2_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (6));
  B_reg2_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(6));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(6));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(6));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(6));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(6));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(6));
  C_reg2_4 = __riscv_vfmacc_vv_f32m1(C_reg2_4, A_reg, B_reg2_4,(6));
  C_reg2_5 = __riscv_vfmacc_vv_f32m1(C_reg2_5, A_reg, B_reg2_5,(6));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(6));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(6));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(6));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(6));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(6));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(6));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(6));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(6));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_4,(6));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg2_5,(6));
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
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(6));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(6));
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
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(6));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(6));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(6));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(6));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(6));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(6));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(6));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(6));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(6));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(6));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(6));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(6));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(6));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(6));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(6));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(6));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(6));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(6));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(6));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(6));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(6));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(6));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(6));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(6));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(6));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(6));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(6));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(6));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(6));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(6));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(6));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(6));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(6));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(6));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(6));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(6));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(6));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(6));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(6));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(6));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(6));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(6));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(6));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(6));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(6));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(6));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(6));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(6));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(6));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(6));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(6));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(6));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(6));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(6));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(6));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(6));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(6));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(6));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(6));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(6));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(6));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(6));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(6));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(6));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(6));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(6));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(6));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(6));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(6));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(6));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(6));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(6));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(6));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(6));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(6));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(6));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(6));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(6));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(6));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(6));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(6));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(6));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(6));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(6));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(6));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(6));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(6));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(6));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(6));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(6));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(6));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(6));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(6));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(6));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(6));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(6));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (6));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (6));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (6));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (6));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (6));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (6));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (6));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(6));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(6));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(6));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(6));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(6));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(6));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(6));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(6));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(6));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(6));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(6));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(6));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(6));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(6));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (6));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (6));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (6));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (6));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (6));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (6));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (6));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(6));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(6));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(6));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(6));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(6));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(6));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (6));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (6));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (6));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (6));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (6));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (6));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (6));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (6));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(6));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(6));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(6));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(6));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(6));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(6));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(6));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(6));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(6));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(6));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(6));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(6));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(6));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(6));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(6));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(6));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(6));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (6));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (6));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (6));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (6));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (6));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (6));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (6));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (6));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(6));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(6));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(6));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(6));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(6));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(6));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(6));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(6));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(6));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (7));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (7));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (7));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (7));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (7));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (7));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (7));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (7));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (7));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(7));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(7));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(7));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(7));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(7));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(7));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(7));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(7));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(7));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(7));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(7));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(7));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(7));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(7));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(7));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(7));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(7));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(7));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(7));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(7));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (7));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (7));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (7));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (7));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (7));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (7));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (7));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (7));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (7));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(7));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(7));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(7));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(7));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(7));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(7));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(7));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(7));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(7));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(7));
}

// gemm_RVV_7x11_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 7] @DRAM
// )
void gemm_RVV_7x11_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (7));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (7));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (7));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (7));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (7));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (7));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (7));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (7));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (7));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (7));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(7));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(7));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(7));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(7));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(7));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(7));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(7));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(7));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(7));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(7));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(7));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(7));
}

// gemm_RVV_7x11_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 7] @DRAM
// )
void gemm_RVV_7x11_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(7));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(7));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(7));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(7));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(7));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(7));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(7));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(7));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(7));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(7));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(7));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (7));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (7));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (7));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (7));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (7));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (7));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (7));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (7));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (7));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (7));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(7));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(7));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(7));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(7));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(7));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(7));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(7));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(7));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(7));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(7));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(7));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(7));
}

// gemm_RVV_7x12_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 7] @DRAM
// )
void gemm_RVV_7x12_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (7));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (7));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (7));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (7));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (7));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (7));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (7));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (7));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(4));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (7));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (7));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (7));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(7));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(7));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(7));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(7));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(7));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(7));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(7));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(7));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(7));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(7));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(7));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(7));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(7));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(7));
}

// gemm_RVV_7x12_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 7] @DRAM
// )
void gemm_RVV_7x12_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(7));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(7));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(7));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(7));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(7));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(7));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(7));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(7));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(7));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(7));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(7));
C_reg2_3 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(7));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (7));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (7));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (7));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (7));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (7));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (7));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (7));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (7));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(4));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (7));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (7));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (7));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(7));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(7));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(7));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(7));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(7));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(7));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(7));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(7));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(7));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(7));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(7));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(7));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(7));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(7));
}

// gemm_RVV_7x13_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 7] @DRAM
// )
void gemm_RVV_7x13_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
vfloat32m1_t C_reg2_4;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg2_4 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
vfloat32m1_t B_reg2_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (7));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (7));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (7));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (7));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (7));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (7));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (7));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (7));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(5));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (7));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (7));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (7));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (7));
  B_reg2_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(7));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(7));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(7));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(7));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(7));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(7));
  C_reg2_4 = __riscv_vfmacc_vv_f32m1(C_reg2_4, A_reg, B_reg2_4,(7));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(7));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(7));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(7));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(7));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(7));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(7));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(7));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(7));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_4,(7));
}

// gemm_RVV_7x13_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 7] @DRAM
// )
void gemm_RVV_7x13_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
vfloat32m1_t C_reg2_4;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(7));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(7));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(7));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(7));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(7));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(7));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(7));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(7));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(7));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(7));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(7));
C_reg2_3 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(7));
C_reg2_4 = __riscv_vle32_v_f32m1(&C[(12) * (ldc)],(7));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
vfloat32m1_t B_reg2_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (7));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (7));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (7));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (7));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (7));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (7));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (7));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (7));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(5));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (7));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (7));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (7));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (7));
  B_reg2_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(7));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(7));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(7));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(7));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(7));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(7));
  C_reg2_4 = __riscv_vfmacc_vv_f32m1(C_reg2_4, A_reg, B_reg2_4,(7));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(7));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(7));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(7));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(7));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(7));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(7));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(7));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(7));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_4,(7));
}

// gemm_RVV_7x14_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 7] @DRAM
// )
void gemm_RVV_7x14_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
vfloat32m1_t C_reg2_4;
vfloat32m1_t C_reg2_5;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg2_4 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg2_5 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
vfloat32m1_t B_reg2_4;
vfloat32m1_t B_reg2_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (7));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (7));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (7));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (7));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (7));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (7));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (7));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (7));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(6));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (7));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (7));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (7));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (7));
  B_reg2_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (7));
  B_reg2_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(7));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(7));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(7));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(7));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(7));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(7));
  C_reg2_4 = __riscv_vfmacc_vv_f32m1(C_reg2_4, A_reg, B_reg2_4,(7));
  C_reg2_5 = __riscv_vfmacc_vv_f32m1(C_reg2_5, A_reg, B_reg2_5,(7));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(7));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(7));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(7));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(7));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(7));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(7));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(7));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(7));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_4,(7));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg2_5,(7));
}

// gemm_RVV_7x14_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 7] @DRAM
// )
void gemm_RVV_7x14_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
vfloat32m1_t C_reg2_4;
vfloat32m1_t C_reg2_5;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(7));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(7));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(7));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(7));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(7));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(7));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(7));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(7));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(7));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(7));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(7));
C_reg2_3 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(7));
C_reg2_4 = __riscv_vle32_v_f32m1(&C[(12) * (ldc)],(7));
C_reg2_5 = __riscv_vle32_v_f32m1(&C[(13) * (ldc)],(7));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
vfloat32m1_t B_reg2_4;
vfloat32m1_t B_reg2_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (7));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (7));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (7));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (7));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (7));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (7));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (7));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (7));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(6));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (7));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (7));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (7));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (7));
  B_reg2_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (7));
  B_reg2_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(7));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(7));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(7));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(7));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(7));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(7));
  C_reg2_4 = __riscv_vfmacc_vv_f32m1(C_reg2_4, A_reg, B_reg2_4,(7));
  C_reg2_5 = __riscv_vfmacc_vv_f32m1(C_reg2_5, A_reg, B_reg2_5,(7));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(7));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(7));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(7));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(7));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(7));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(7));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(7));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(7));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_4,(7));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg2_5,(7));
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
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(7));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(7));
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
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(7));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(7));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(7));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(7));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(7));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(7));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(7));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(7));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(7));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(7));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(7));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(7));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(7));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(7));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(7));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(7));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(7));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(7));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(7));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(7));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(7));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(7));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(7));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(7));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(7));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(7));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(7));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(7));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(7));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(7));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(7));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(7));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(7));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(7));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(7));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(7));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(7));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(7));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(7));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(7));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(7));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(7));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(7));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(7));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(7));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(7));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(7));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(7));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(7));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(7));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(7));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(7));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(7));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(7));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(7));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(7));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(7));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(7));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(7));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(7));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(7));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(7));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(7));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(7));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(7));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(7));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(7));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(7));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(7));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(7));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(7));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(7));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(7));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(7));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(7));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(7));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(7));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(7));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(7));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(7));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(7));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(7));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(7));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(7));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(7));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(7));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(7));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(7));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(7));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(7));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(7));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(7));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(7));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(7));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(7));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(7));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (7));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (7));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (7));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (7));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (7));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (7));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (7));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(7));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(7));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(7));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(7));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(7));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(7));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(7));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(7));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(7));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(7));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(7));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(7));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(7));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(7));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (7));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (7));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (7));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (7));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (7));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (7));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (7));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(7));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(7));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(7));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(7));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(7));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(7));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (7));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (7));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (7));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (7));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (7));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (7));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (7));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (7));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(7));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(7));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(7));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(7));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(7));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(7));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(7));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(7));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(7));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(7));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(7));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(7));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(7));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(7));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(7));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(7));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(7));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (7));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (7));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (7));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (7));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (7));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (7));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (7));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (7));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(7));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(7));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(7));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(7));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(7));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(7));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(7));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(7));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(7));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (8));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (8));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (8));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (8));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (8));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (8));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (8));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (8));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (8));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(8));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(8));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(8));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(8));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(8));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(8));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(8));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(8));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(8));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(8));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(8));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(8));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(8));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(8));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(8));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(8));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(8));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(8));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(8));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(8));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (8));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (8));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (8));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (8));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (8));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (8));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (8));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (8));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (8));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(8));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(8));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(8));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(8));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(8));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(8));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(8));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(8));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(8));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(8));
}

// gemm_RVV_8x11_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 8] @DRAM
// )
void gemm_RVV_8x11_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (8));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (8));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (8));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (8));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (8));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (8));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (8));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (8));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (8));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (8));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(8));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(8));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(8));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(8));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(8));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(8));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(8));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(8));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(8));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(8));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(8));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(8));
}

// gemm_RVV_8x11_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 8] @DRAM
// )
void gemm_RVV_8x11_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(8));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(8));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(8));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(8));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(8));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(8));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(8));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(8));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(8));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(8));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(8));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (8));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (8));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (8));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (8));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (8));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (8));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (8));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (8));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (8));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (8));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(8));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(8));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(8));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(8));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(8));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(8));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(8));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(8));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(8));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(8));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(8));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(8));
}

// gemm_RVV_8x12_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 8] @DRAM
// )
void gemm_RVV_8x12_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (8));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (8));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (8));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (8));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (8));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (8));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (8));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (8));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(4));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (8));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (8));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (8));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(8));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(8));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(8));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(8));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(8));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(8));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(8));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(8));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(8));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(8));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(8));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(8));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(8));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(8));
}

// gemm_RVV_8x12_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 8] @DRAM
// )
void gemm_RVV_8x12_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(8));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(8));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(8));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(8));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(8));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(8));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(8));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(8));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(8));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(8));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(8));
C_reg2_3 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(8));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (8));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (8));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (8));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (8));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (8));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (8));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (8));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (8));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(4));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (8));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (8));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (8));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(8));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(8));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(8));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(8));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(8));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(8));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(8));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(8));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(8));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(8));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(8));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(8));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(8));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(8));
}

// gemm_RVV_8x13_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 8] @DRAM
// )
void gemm_RVV_8x13_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
vfloat32m1_t C_reg2_4;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg2_4 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
vfloat32m1_t B_reg2_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (8));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (8));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (8));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (8));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (8));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (8));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (8));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (8));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(5));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (8));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (8));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (8));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (8));
  B_reg2_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(8));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(8));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(8));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(8));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(8));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(8));
  C_reg2_4 = __riscv_vfmacc_vv_f32m1(C_reg2_4, A_reg, B_reg2_4,(8));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(8));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(8));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(8));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(8));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(8));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(8));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(8));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(8));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_4,(8));
}

// gemm_RVV_8x13_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 8] @DRAM
// )
void gemm_RVV_8x13_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
vfloat32m1_t C_reg2_4;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(8));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(8));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(8));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(8));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(8));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(8));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(8));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(8));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(8));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(8));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(8));
C_reg2_3 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(8));
C_reg2_4 = __riscv_vle32_v_f32m1(&C[(12) * (ldc)],(8));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
vfloat32m1_t B_reg2_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (8));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (8));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (8));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (8));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (8));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (8));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (8));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (8));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(5));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (8));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (8));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (8));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (8));
  B_reg2_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(8));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(8));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(8));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(8));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(8));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(8));
  C_reg2_4 = __riscv_vfmacc_vv_f32m1(C_reg2_4, A_reg, B_reg2_4,(8));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(8));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(8));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(8));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(8));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(8));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(8));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(8));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(8));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_4,(8));
}

// gemm_RVV_8x14_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 8] @DRAM
// )
void gemm_RVV_8x14_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
vfloat32m1_t C_reg2_4;
vfloat32m1_t C_reg2_5;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg2_4 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg2_5 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
vfloat32m1_t B_reg2_4;
vfloat32m1_t B_reg2_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (8));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (8));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (8));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (8));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (8));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (8));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (8));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (8));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(6));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (8));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (8));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (8));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (8));
  B_reg2_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (8));
  B_reg2_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(8));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(8));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(8));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(8));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(8));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(8));
  C_reg2_4 = __riscv_vfmacc_vv_f32m1(C_reg2_4, A_reg, B_reg2_4,(8));
  C_reg2_5 = __riscv_vfmacc_vv_f32m1(C_reg2_5, A_reg, B_reg2_5,(8));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(8));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(8));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(8));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(8));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(8));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(8));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(8));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(8));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_4,(8));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg2_5,(8));
}

// gemm_RVV_8x14_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 8] @DRAM
// )
void gemm_RVV_8x14_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
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
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
vfloat32m1_t C_reg2_3;
vfloat32m1_t C_reg2_4;
vfloat32m1_t C_reg2_5;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(8));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(8));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(8));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(8));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(8));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(8));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(8));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(8));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(8));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(8));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(8));
C_reg2_3 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(8));
C_reg2_4 = __riscv_vle32_v_f32m1(&C[(12) * (ldc)],(8));
C_reg2_5 = __riscv_vle32_v_f32m1(&C[(13) * (ldc)],(8));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
vfloat32m1_t B_reg2_3;
vfloat32m1_t B_reg2_4;
vfloat32m1_t B_reg2_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (8));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (8));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (8));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (8));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (8));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (8));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (8));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (8));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(6));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (8));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (8));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (8));
  B_reg2_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (8));
  B_reg2_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (8));
  B_reg2_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(8));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(8));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(8));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(8));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(8));
  C_reg2_3 = __riscv_vfmacc_vv_f32m1(C_reg2_3, A_reg, B_reg2_3,(8));
  C_reg2_4 = __riscv_vfmacc_vv_f32m1(C_reg2_4, A_reg, B_reg2_4,(8));
  C_reg2_5 = __riscv_vfmacc_vv_f32m1(C_reg2_5, A_reg, B_reg2_5,(8));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(8));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(8));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(8));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(8));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(8));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(8));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(8));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg2_3,(8));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_4,(8));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg2_5,(8));
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
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(8));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(8));
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
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(8));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(8));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(8));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(8));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(8));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(8));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(8));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(8));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(8));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(8));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(8));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(8));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(8));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(8));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(8));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(8));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(8));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(8));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(8));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(8));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(8));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(8));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(8));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(8));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(8));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(8));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(8));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(8));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(8));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(8));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(8));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(8));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(8));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(8));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(8));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(8));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(8));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(8));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(8));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(8));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(8));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(8));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(8));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(8));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(8));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(8));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(8));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(8));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(8));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(8));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(8));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(8));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(8));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(8));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(8));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(8));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(8));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(8));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(8));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(8));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(8));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(8));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(8));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(8));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(8));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(8));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(8));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(8));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(8));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(8));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(8));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(8));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(8));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(8));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(8));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(8));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(8));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(8));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(8));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(8));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(8));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(8));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(8));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(8));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(8));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(8));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(8));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(8));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 2],(8));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 3],(8));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 4],(8));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 5],(8));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B[(k) * (ldb) + 6],(8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(8));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(8));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(8));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(8));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (8));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (8));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (8));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (8));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (8));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (8));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (8));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(8));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(8));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(8));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(8));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(8));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(8));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(8));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(8));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(8));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(8));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(8));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(8));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(8));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(8));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (8));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (8));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (8));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (8));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (8));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (8));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (8));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(8));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(8));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(8));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(8));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(8));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(8));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (8));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (8));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (8));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (8));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (8));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (8));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (8));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (8));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(8));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(8));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(8));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(8));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(8));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(8));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(8));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(8));
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
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(8));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(8));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(8));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(8));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(8));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(8));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(8));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(8));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(8));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb)],(8));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (8));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (8));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (8));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (8));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (4), (8));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (5), (8));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (6), (8));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (7), (8));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (ldb) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (lda)],(8));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(8));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(8));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(8));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(8));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(8));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(8));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(8));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(8));
}


/* relying on the following instruction..."
rvv_broadcast_8xf32(dst,src,vl)
{dst_data} = __riscv_vfmv_v_f_f32m1({src_data},{vl});
*/

/* relying on the following instruction..."
rvv_broadcast_8xf32_0(dst,vl)
{dst_data} = __riscv_vfmv_v_f_f32m1(0.0f,{vl});
*/

/* relying on the following instruction..."
rvv_gather_8xf32(dst,src,imm,vl)
{dst_data} = __riscv_vrgather_vx_f32m1({src_data}, {imm}, {vl});
*/

/* relying on the following instruction..."
rvv_vfmacc_8xf32_8xf32(dst,lhs,rhs,vl)
{dst_data} = __riscv_vfmacc_vv_f32m1({dst_data}, {lhs_data}, {rhs_data},{vl});
*/

/* relying on the following instruction..."
rvv_vld_8xf32(dst,src,vl)
{dst_data} = __riscv_vle32_v_f32m1(&{src_data},{vl});
*/

/* relying on the following instruction..."
rvv_vst_8xf32(dst,src,vl)
__riscv_vse32_v_f32m1(&{dst_data}, {src_data},{vl});
*/
