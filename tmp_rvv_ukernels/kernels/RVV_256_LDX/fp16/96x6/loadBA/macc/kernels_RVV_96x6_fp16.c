#include "kernels_RVV_96x6_fp16.h"



#include <stdio.h>
#include <stdlib.h>

#include <riscv_vector.h>


// gemm_RVV_96x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 96] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 96] @DRAM
// )
void gemm_RVV_96x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_reg_5;
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
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
vfloat16m1_t C_reg_2_3;
vfloat16m1_t C_reg_2_4;
vfloat16m1_t C_reg_2_5;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_3_1;
vfloat16m1_t C_reg_3_2;
vfloat16m1_t C_reg_3_3;
vfloat16m1_t C_reg_3_4;
vfloat16m1_t C_reg_3_5;
vfloat16m1_t C_reg_4_0;
vfloat16m1_t C_reg_4_1;
vfloat16m1_t C_reg_4_2;
vfloat16m1_t C_reg_4_3;
vfloat16m1_t C_reg_4_4;
vfloat16m1_t C_reg_4_5;
vfloat16m1_t C_reg_5_0;
vfloat16m1_t C_reg_5_1;
vfloat16m1_t C_reg_5_2;
vfloat16m1_t C_reg_5_3;
vfloat16m1_t C_reg_5_4;
vfloat16m1_t C_reg_5_5;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_5 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_5 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_5 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_5 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_4_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_4_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_4_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_4_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_4_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_4_5 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_5_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_5_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_5_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_5_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_5_4 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_5_5 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (lda) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (lda) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A[(k) * (lda) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A[(k) * (lda) + 64],(16));
  A_reg_5 = __riscv_vle16_v_f16m1(&A[(k) * (lda) + 80],(16));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(16));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(16));
  C_reg_0_3 = __riscv_vfmacc_vf_f16m1(C_reg_0_3, B[(k) * (ldb)], A_reg_3,(16));
  C_reg_0_4 = __riscv_vfmacc_vf_f16m1(C_reg_0_4, B[(k) * (ldb)], A_reg_4,(16));
  C_reg_0_5 = __riscv_vfmacc_vf_f16m1(C_reg_0_5, B[(k) * (ldb)], A_reg_5,(16));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (ldb) + 1], A_reg_0,(16));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (ldb) + 1], A_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (ldb) + 1], A_reg_2,(16));
  C_reg_1_3 = __riscv_vfmacc_vf_f16m1(C_reg_1_3, B[(k) * (ldb) + 1], A_reg_3,(16));
  C_reg_1_4 = __riscv_vfmacc_vf_f16m1(C_reg_1_4, B[(k) * (ldb) + 1], A_reg_4,(16));
  C_reg_1_5 = __riscv_vfmacc_vf_f16m1(C_reg_1_5, B[(k) * (ldb) + 1], A_reg_5,(16));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (ldb) + 2], A_reg_0,(16));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (ldb) + 2], A_reg_1,(16));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (ldb) + 2], A_reg_2,(16));
  C_reg_2_3 = __riscv_vfmacc_vf_f16m1(C_reg_2_3, B[(k) * (ldb) + 2], A_reg_3,(16));
  C_reg_2_4 = __riscv_vfmacc_vf_f16m1(C_reg_2_4, B[(k) * (ldb) + 2], A_reg_4,(16));
  C_reg_2_5 = __riscv_vfmacc_vf_f16m1(C_reg_2_5, B[(k) * (ldb) + 2], A_reg_5,(16));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (ldb) + 3], A_reg_0,(16));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (ldb) + 3], A_reg_1,(16));
  C_reg_3_2 = __riscv_vfmacc_vf_f16m1(C_reg_3_2, B[(k) * (ldb) + 3], A_reg_2,(16));
  C_reg_3_3 = __riscv_vfmacc_vf_f16m1(C_reg_3_3, B[(k) * (ldb) + 3], A_reg_3,(16));
  C_reg_3_4 = __riscv_vfmacc_vf_f16m1(C_reg_3_4, B[(k) * (ldb) + 3], A_reg_4,(16));
  C_reg_3_5 = __riscv_vfmacc_vf_f16m1(C_reg_3_5, B[(k) * (ldb) + 3], A_reg_5,(16));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (ldb) + 4], A_reg_0,(16));
  C_reg_4_1 = __riscv_vfmacc_vf_f16m1(C_reg_4_1, B[(k) * (ldb) + 4], A_reg_1,(16));
  C_reg_4_2 = __riscv_vfmacc_vf_f16m1(C_reg_4_2, B[(k) * (ldb) + 4], A_reg_2,(16));
  C_reg_4_3 = __riscv_vfmacc_vf_f16m1(C_reg_4_3, B[(k) * (ldb) + 4], A_reg_3,(16));
  C_reg_4_4 = __riscv_vfmacc_vf_f16m1(C_reg_4_4, B[(k) * (ldb) + 4], A_reg_4,(16));
  C_reg_4_5 = __riscv_vfmacc_vf_f16m1(C_reg_4_5, B[(k) * (ldb) + 4], A_reg_5,(16));
  C_reg_5_0 = __riscv_vfmacc_vf_f16m1(C_reg_5_0, B[(k) * (ldb) + 5], A_reg_0,(16));
  C_reg_5_1 = __riscv_vfmacc_vf_f16m1(C_reg_5_1, B[(k) * (ldb) + 5], A_reg_1,(16));
  C_reg_5_2 = __riscv_vfmacc_vf_f16m1(C_reg_5_2, B[(k) * (ldb) + 5], A_reg_2,(16));
  C_reg_5_3 = __riscv_vfmacc_vf_f16m1(C_reg_5_3, B[(k) * (ldb) + 5], A_reg_3,(16));
  C_reg_5_4 = __riscv_vfmacc_vf_f16m1(C_reg_5_4, B[(k) * (ldb) + 5], A_reg_4,(16));
  C_reg_5_5 = __riscv_vfmacc_vf_f16m1(C_reg_5_5, B[(k) * (ldb) + 5], A_reg_5,(16));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C[80], C_reg_0_5,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C[ldc + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C[ldc + 64], C_reg_1_4,(16));
__riscv_vse16_v_f16m1(&C[ldc + 80], C_reg_1_5,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_reg_2_2,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 48], C_reg_2_3,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 64], C_reg_2_4,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 80], C_reg_2_5,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 32], C_reg_3_2,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 48], C_reg_3_3,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 64], C_reg_3_4,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 80], C_reg_3_5,(16));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(16));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 16], C_reg_4_1,(16));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 32], C_reg_4_2,(16));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 48], C_reg_4_3,(16));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 64], C_reg_4_4,(16));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 80], C_reg_4_5,(16));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5_0,(16));
__riscv_vse16_v_f16m1(&C[(5) * (ldc) + 16], C_reg_5_1,(16));
__riscv_vse16_v_f16m1(&C[(5) * (ldc) + 32], C_reg_5_2,(16));
__riscv_vse16_v_f16m1(&C[(5) * (ldc) + 48], C_reg_5_3,(16));
__riscv_vse16_v_f16m1(&C[(5) * (ldc) + 64], C_reg_5_4,(16));
__riscv_vse16_v_f16m1(&C[(5) * (ldc) + 80], C_reg_5_5,(16));
}

// gemm_RVV_96x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 96] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 96] @DRAM
// )
void gemm_RVV_96x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t A_reg_0;
vfloat16m1_t A_reg_1;
vfloat16m1_t A_reg_2;
vfloat16m1_t A_reg_3;
vfloat16m1_t A_reg_4;
vfloat16m1_t A_reg_5;
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
vfloat16m1_t C_reg_2_0;
vfloat16m1_t C_reg_2_1;
vfloat16m1_t C_reg_2_2;
vfloat16m1_t C_reg_2_3;
vfloat16m1_t C_reg_2_4;
vfloat16m1_t C_reg_2_5;
vfloat16m1_t C_reg_3_0;
vfloat16m1_t C_reg_3_1;
vfloat16m1_t C_reg_3_2;
vfloat16m1_t C_reg_3_3;
vfloat16m1_t C_reg_3_4;
vfloat16m1_t C_reg_3_5;
vfloat16m1_t C_reg_4_0;
vfloat16m1_t C_reg_4_1;
vfloat16m1_t C_reg_4_2;
vfloat16m1_t C_reg_4_3;
vfloat16m1_t C_reg_4_4;
vfloat16m1_t C_reg_4_5;
vfloat16m1_t C_reg_5_0;
vfloat16m1_t C_reg_5_1;
vfloat16m1_t C_reg_5_2;
vfloat16m1_t C_reg_5_3;
vfloat16m1_t C_reg_5_4;
vfloat16m1_t C_reg_5_5;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C[48],(16));
C_reg_0_4 = __riscv_vle16_v_f16m1(&C[64],(16));
C_reg_0_5 = __riscv_vle16_v_f16m1(&C[80],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C[ldc],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C[ldc + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C[ldc + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C[ldc + 48],(16));
C_reg_1_4 = __riscv_vle16_v_f16m1(&C[ldc + 64],(16));
C_reg_1_5 = __riscv_vle16_v_f16m1(&C[ldc + 80],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 16],(16));
C_reg_2_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 32],(16));
C_reg_2_3 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 48],(16));
C_reg_2_4 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 64],(16));
C_reg_2_5 = __riscv_vle16_v_f16m1(&C[(2) * (ldc) + 80],(16));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(16));
C_reg_3_1 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 16],(16));
C_reg_3_2 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 32],(16));
C_reg_3_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 48],(16));
C_reg_3_4 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 64],(16));
C_reg_3_5 = __riscv_vle16_v_f16m1(&C[(3) * (ldc) + 80],(16));
C_reg_4_0 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(16));
C_reg_4_1 = __riscv_vle16_v_f16m1(&C[(4) * (ldc) + 16],(16));
C_reg_4_2 = __riscv_vle16_v_f16m1(&C[(4) * (ldc) + 32],(16));
C_reg_4_3 = __riscv_vle16_v_f16m1(&C[(4) * (ldc) + 48],(16));
C_reg_4_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc) + 64],(16));
C_reg_4_5 = __riscv_vle16_v_f16m1(&C[(4) * (ldc) + 80],(16));
C_reg_5_0 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(16));
C_reg_5_1 = __riscv_vle16_v_f16m1(&C[(5) * (ldc) + 16],(16));
C_reg_5_2 = __riscv_vle16_v_f16m1(&C[(5) * (ldc) + 32],(16));
C_reg_5_3 = __riscv_vle16_v_f16m1(&C[(5) * (ldc) + 48],(16));
C_reg_5_4 = __riscv_vle16_v_f16m1(&C[(5) * (ldc) + 64],(16));
C_reg_5_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc) + 80],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (lda)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (lda) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (lda) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A[(k) * (lda) + 48],(16));
  A_reg_4 = __riscv_vle16_v_f16m1(&A[(k) * (lda) + 64],(16));
  A_reg_5 = __riscv_vle16_v_f16m1(&A[(k) * (lda) + 80],(16));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (ldb)], A_reg_1,(16));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (ldb)], A_reg_2,(16));
  C_reg_0_3 = __riscv_vfmacc_vf_f16m1(C_reg_0_3, B[(k) * (ldb)], A_reg_3,(16));
  C_reg_0_4 = __riscv_vfmacc_vf_f16m1(C_reg_0_4, B[(k) * (ldb)], A_reg_4,(16));
  C_reg_0_5 = __riscv_vfmacc_vf_f16m1(C_reg_0_5, B[(k) * (ldb)], A_reg_5,(16));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (ldb) + 1], A_reg_0,(16));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (ldb) + 1], A_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (ldb) + 1], A_reg_2,(16));
  C_reg_1_3 = __riscv_vfmacc_vf_f16m1(C_reg_1_3, B[(k) * (ldb) + 1], A_reg_3,(16));
  C_reg_1_4 = __riscv_vfmacc_vf_f16m1(C_reg_1_4, B[(k) * (ldb) + 1], A_reg_4,(16));
  C_reg_1_5 = __riscv_vfmacc_vf_f16m1(C_reg_1_5, B[(k) * (ldb) + 1], A_reg_5,(16));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (ldb) + 2], A_reg_0,(16));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (ldb) + 2], A_reg_1,(16));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (ldb) + 2], A_reg_2,(16));
  C_reg_2_3 = __riscv_vfmacc_vf_f16m1(C_reg_2_3, B[(k) * (ldb) + 2], A_reg_3,(16));
  C_reg_2_4 = __riscv_vfmacc_vf_f16m1(C_reg_2_4, B[(k) * (ldb) + 2], A_reg_4,(16));
  C_reg_2_5 = __riscv_vfmacc_vf_f16m1(C_reg_2_5, B[(k) * (ldb) + 2], A_reg_5,(16));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (ldb) + 3], A_reg_0,(16));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (ldb) + 3], A_reg_1,(16));
  C_reg_3_2 = __riscv_vfmacc_vf_f16m1(C_reg_3_2, B[(k) * (ldb) + 3], A_reg_2,(16));
  C_reg_3_3 = __riscv_vfmacc_vf_f16m1(C_reg_3_3, B[(k) * (ldb) + 3], A_reg_3,(16));
  C_reg_3_4 = __riscv_vfmacc_vf_f16m1(C_reg_3_4, B[(k) * (ldb) + 3], A_reg_4,(16));
  C_reg_3_5 = __riscv_vfmacc_vf_f16m1(C_reg_3_5, B[(k) * (ldb) + 3], A_reg_5,(16));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (ldb) + 4], A_reg_0,(16));
  C_reg_4_1 = __riscv_vfmacc_vf_f16m1(C_reg_4_1, B[(k) * (ldb) + 4], A_reg_1,(16));
  C_reg_4_2 = __riscv_vfmacc_vf_f16m1(C_reg_4_2, B[(k) * (ldb) + 4], A_reg_2,(16));
  C_reg_4_3 = __riscv_vfmacc_vf_f16m1(C_reg_4_3, B[(k) * (ldb) + 4], A_reg_3,(16));
  C_reg_4_4 = __riscv_vfmacc_vf_f16m1(C_reg_4_4, B[(k) * (ldb) + 4], A_reg_4,(16));
  C_reg_4_5 = __riscv_vfmacc_vf_f16m1(C_reg_4_5, B[(k) * (ldb) + 4], A_reg_5,(16));
  C_reg_5_0 = __riscv_vfmacc_vf_f16m1(C_reg_5_0, B[(k) * (ldb) + 5], A_reg_0,(16));
  C_reg_5_1 = __riscv_vfmacc_vf_f16m1(C_reg_5_1, B[(k) * (ldb) + 5], A_reg_1,(16));
  C_reg_5_2 = __riscv_vfmacc_vf_f16m1(C_reg_5_2, B[(k) * (ldb) + 5], A_reg_2,(16));
  C_reg_5_3 = __riscv_vfmacc_vf_f16m1(C_reg_5_3, B[(k) * (ldb) + 5], A_reg_3,(16));
  C_reg_5_4 = __riscv_vfmacc_vf_f16m1(C_reg_5_4, B[(k) * (ldb) + 5], A_reg_4,(16));
  C_reg_5_5 = __riscv_vfmacc_vf_f16m1(C_reg_5_5, B[(k) * (ldb) + 5], A_reg_5,(16));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C[64], C_reg_0_4,(16));
__riscv_vse16_v_f16m1(&C[80], C_reg_0_5,(16));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C[ldc + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C[ldc + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C[ldc + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C[ldc + 64], C_reg_1_4,(16));
__riscv_vse16_v_f16m1(&C[ldc + 80], C_reg_1_5,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 32], C_reg_2_2,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 48], C_reg_2_3,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 64], C_reg_2_4,(16));
__riscv_vse16_v_f16m1(&C[(2) * (ldc) + 80], C_reg_2_5,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 32], C_reg_3_2,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 48], C_reg_3_3,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 64], C_reg_3_4,(16));
__riscv_vse16_v_f16m1(&C[(3) * (ldc) + 80], C_reg_3_5,(16));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(16));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 16], C_reg_4_1,(16));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 32], C_reg_4_2,(16));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 48], C_reg_4_3,(16));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 64], C_reg_4_4,(16));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 80], C_reg_4_5,(16));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5_0,(16));
__riscv_vse16_v_f16m1(&C[(5) * (ldc) + 16], C_reg_5_1,(16));
__riscv_vse16_v_f16m1(&C[(5) * (ldc) + 32], C_reg_5_2,(16));
__riscv_vse16_v_f16m1(&C[(5) * (ldc) + 48], C_reg_5_3,(16));
__riscv_vse16_v_f16m1(&C[(5) * (ldc) + 64], C_reg_5_4,(16));
__riscv_vse16_v_f16m1(&C[(5) * (ldc) + 80], C_reg_5_5,(16));
}


/* relying on the following instruction..."
rvv_broadcast_16xf16_0(dst,vl)
{dst_data} = __riscv_vfmv_v_f_f16m1(0.0f,{vl});
*/

/* relying on the following instruction..."
rvv_vfmacc_16xf16_1xf16(dst,lhs,rhs,vl)
{dst_data} = __riscv_vfmacc_vf_f16m1({dst_data}, {rhs_data}, {lhs_data},{vl});
*/

/* relying on the following instruction..."
rvv_vld_16xf16(dst,src,vl)
{dst_data} = __riscv_vle16_v_f16m1(&{src_data},{vl});
*/

/* relying on the following instruction..."
rvv_vst_16xf16(dst,src,vl)
__riscv_vse16_v_f16m1(&{dst_data}, {src_data},{vl});
*/
