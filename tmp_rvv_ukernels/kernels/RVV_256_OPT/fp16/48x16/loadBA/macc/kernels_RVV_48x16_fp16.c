#include "kernels_RVV_48x16_fp16.h"



#include <stdio.h>
#include <stdlib.h>

#include <riscv_vector.h>


// gemm_RVV_48x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 48] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 48] @DRAM
// )
void gemm_RVV_48x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
vfloat16m1_t C_reg_4_0;
vfloat16m1_t C_reg_4_1;
vfloat16m1_t C_reg_4_2;
vfloat16m1_t C_reg_5_0;
vfloat16m1_t C_reg_5_1;
vfloat16m1_t C_reg_5_2;
vfloat16m1_t C_reg_6_0;
vfloat16m1_t C_reg_6_1;
vfloat16m1_t C_reg_6_2;
vfloat16m1_t C_reg_7_0;
vfloat16m1_t C_reg_7_1;
vfloat16m1_t C_reg_7_2;
vfloat16m1_t C_reg_8_0;
vfloat16m1_t C_reg_8_1;
vfloat16m1_t C_reg_8_2;
vfloat16m1_t C_reg_9_0;
vfloat16m1_t C_reg_9_1;
vfloat16m1_t C_reg_9_2;
vfloat16m1_t C_reg_10_0;
vfloat16m1_t C_reg_10_1;
vfloat16m1_t C_reg_10_2;
vfloat16m1_t C_reg_11_0;
vfloat16m1_t C_reg_11_1;
vfloat16m1_t C_reg_11_2;
vfloat16m1_t C_reg_12_0;
vfloat16m1_t C_reg_12_1;
vfloat16m1_t C_reg_12_2;
vfloat16m1_t C_reg_13_0;
vfloat16m1_t C_reg_13_1;
vfloat16m1_t C_reg_13_2;
vfloat16m1_t C_reg_14_0;
vfloat16m1_t C_reg_14_1;
vfloat16m1_t C_reg_14_2;
vfloat16m1_t C_reg_15_0;
vfloat16m1_t C_reg_15_1;
vfloat16m1_t C_reg_15_2;
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
C_reg_4_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_4_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_4_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_5_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_5_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_5_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_6_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_6_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_6_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_7_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_7_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_7_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_8_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_8_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_8_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_9_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_9_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_9_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_10_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_10_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_10_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_11_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_11_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_11_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_12_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_12_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_12_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_13_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_13_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_13_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_14_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_14_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_14_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_15_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_15_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_15_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(16));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (16)], A_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (16)], A_reg_1,(16));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (16)], A_reg_2,(16));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (16) + 1], A_reg_0,(16));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (16) + 1], A_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (16) + 1], A_reg_2,(16));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (16) + 2], A_reg_0,(16));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (16) + 2], A_reg_1,(16));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (16) + 2], A_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (16) + 3], A_reg_0,(16));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (16) + 3], A_reg_1,(16));
  C_reg_3_2 = __riscv_vfmacc_vf_f16m1(C_reg_3_2, B[(k) * (16) + 3], A_reg_2,(16));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (16) + 4], A_reg_0,(16));
  C_reg_4_1 = __riscv_vfmacc_vf_f16m1(C_reg_4_1, B[(k) * (16) + 4], A_reg_1,(16));
  C_reg_4_2 = __riscv_vfmacc_vf_f16m1(C_reg_4_2, B[(k) * (16) + 4], A_reg_2,(16));
  C_reg_5_0 = __riscv_vfmacc_vf_f16m1(C_reg_5_0, B[(k) * (16) + 5], A_reg_0,(16));
  C_reg_5_1 = __riscv_vfmacc_vf_f16m1(C_reg_5_1, B[(k) * (16) + 5], A_reg_1,(16));
  C_reg_5_2 = __riscv_vfmacc_vf_f16m1(C_reg_5_2, B[(k) * (16) + 5], A_reg_2,(16));
  C_reg_6_0 = __riscv_vfmacc_vf_f16m1(C_reg_6_0, B[(k) * (16) + 6], A_reg_0,(16));
  C_reg_6_1 = __riscv_vfmacc_vf_f16m1(C_reg_6_1, B[(k) * (16) + 6], A_reg_1,(16));
  C_reg_6_2 = __riscv_vfmacc_vf_f16m1(C_reg_6_2, B[(k) * (16) + 6], A_reg_2,(16));
  C_reg_7_0 = __riscv_vfmacc_vf_f16m1(C_reg_7_0, B[(k) * (16) + 7], A_reg_0,(16));
  C_reg_7_1 = __riscv_vfmacc_vf_f16m1(C_reg_7_1, B[(k) * (16) + 7], A_reg_1,(16));
  C_reg_7_2 = __riscv_vfmacc_vf_f16m1(C_reg_7_2, B[(k) * (16) + 7], A_reg_2,(16));
  C_reg_8_0 = __riscv_vfmacc_vf_f16m1(C_reg_8_0, B[(k) * (16) + 8], A_reg_0,(16));
  C_reg_8_1 = __riscv_vfmacc_vf_f16m1(C_reg_8_1, B[(k) * (16) + 8], A_reg_1,(16));
  C_reg_8_2 = __riscv_vfmacc_vf_f16m1(C_reg_8_2, B[(k) * (16) + 8], A_reg_2,(16));
  C_reg_9_0 = __riscv_vfmacc_vf_f16m1(C_reg_9_0, B[(k) * (16) + 9], A_reg_0,(16));
  C_reg_9_1 = __riscv_vfmacc_vf_f16m1(C_reg_9_1, B[(k) * (16) + 9], A_reg_1,(16));
  C_reg_9_2 = __riscv_vfmacc_vf_f16m1(C_reg_9_2, B[(k) * (16) + 9], A_reg_2,(16));
  C_reg_10_0 = __riscv_vfmacc_vf_f16m1(C_reg_10_0, B[(k) * (16) + 10], A_reg_0,(16));
  C_reg_10_1 = __riscv_vfmacc_vf_f16m1(C_reg_10_1, B[(k) * (16) + 10], A_reg_1,(16));
  C_reg_10_2 = __riscv_vfmacc_vf_f16m1(C_reg_10_2, B[(k) * (16) + 10], A_reg_2,(16));
  C_reg_11_0 = __riscv_vfmacc_vf_f16m1(C_reg_11_0, B[(k) * (16) + 11], A_reg_0,(16));
  C_reg_11_1 = __riscv_vfmacc_vf_f16m1(C_reg_11_1, B[(k) * (16) + 11], A_reg_1,(16));
  C_reg_11_2 = __riscv_vfmacc_vf_f16m1(C_reg_11_2, B[(k) * (16) + 11], A_reg_2,(16));
  C_reg_12_0 = __riscv_vfmacc_vf_f16m1(C_reg_12_0, B[(k) * (16) + 12], A_reg_0,(16));
  C_reg_12_1 = __riscv_vfmacc_vf_f16m1(C_reg_12_1, B[(k) * (16) + 12], A_reg_1,(16));
  C_reg_12_2 = __riscv_vfmacc_vf_f16m1(C_reg_12_2, B[(k) * (16) + 12], A_reg_2,(16));
  C_reg_13_0 = __riscv_vfmacc_vf_f16m1(C_reg_13_0, B[(k) * (16) + 13], A_reg_0,(16));
  C_reg_13_1 = __riscv_vfmacc_vf_f16m1(C_reg_13_1, B[(k) * (16) + 13], A_reg_1,(16));
  C_reg_13_2 = __riscv_vfmacc_vf_f16m1(C_reg_13_2, B[(k) * (16) + 13], A_reg_2,(16));
  C_reg_14_0 = __riscv_vfmacc_vf_f16m1(C_reg_14_0, B[(k) * (16) + 14], A_reg_0,(16));
  C_reg_14_1 = __riscv_vfmacc_vf_f16m1(C_reg_14_1, B[(k) * (16) + 14], A_reg_1,(16));
  C_reg_14_2 = __riscv_vfmacc_vf_f16m1(C_reg_14_2, B[(k) * (16) + 14], A_reg_2,(16));
  C_reg_15_0 = __riscv_vfmacc_vf_f16m1(C_reg_15_0, B[(k) * (16) + 15], A_reg_0,(16));
  C_reg_15_1 = __riscv_vfmacc_vf_f16m1(C_reg_15_1, B[(k) * (16) + 15], A_reg_1,(16));
  C_reg_15_2 = __riscv_vfmacc_vf_f16m1(C_reg_15_2, B[(k) * (16) + 15], A_reg_2,(16));
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
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(16));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 16], C_reg_4_1,(16));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 32], C_reg_4_2,(16));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5_0,(16));
__riscv_vse16_v_f16m1(&C[(5) * (ldc) + 16], C_reg_5_1,(16));
__riscv_vse16_v_f16m1(&C[(5) * (ldc) + 32], C_reg_5_2,(16));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6_0,(16));
__riscv_vse16_v_f16m1(&C[(6) * (ldc) + 16], C_reg_6_1,(16));
__riscv_vse16_v_f16m1(&C[(6) * (ldc) + 32], C_reg_6_2,(16));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7_0,(16));
__riscv_vse16_v_f16m1(&C[(7) * (ldc) + 16], C_reg_7_1,(16));
__riscv_vse16_v_f16m1(&C[(7) * (ldc) + 32], C_reg_7_2,(16));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg_8_0,(16));
__riscv_vse16_v_f16m1(&C[(8) * (ldc) + 16], C_reg_8_1,(16));
__riscv_vse16_v_f16m1(&C[(8) * (ldc) + 32], C_reg_8_2,(16));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg_9_0,(16));
__riscv_vse16_v_f16m1(&C[(9) * (ldc) + 16], C_reg_9_1,(16));
__riscv_vse16_v_f16m1(&C[(9) * (ldc) + 32], C_reg_9_2,(16));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg_10_0,(16));
__riscv_vse16_v_f16m1(&C[(10) * (ldc) + 16], C_reg_10_1,(16));
__riscv_vse16_v_f16m1(&C[(10) * (ldc) + 32], C_reg_10_2,(16));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg_11_0,(16));
__riscv_vse16_v_f16m1(&C[(11) * (ldc) + 16], C_reg_11_1,(16));
__riscv_vse16_v_f16m1(&C[(11) * (ldc) + 32], C_reg_11_2,(16));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg_12_0,(16));
__riscv_vse16_v_f16m1(&C[(12) * (ldc) + 16], C_reg_12_1,(16));
__riscv_vse16_v_f16m1(&C[(12) * (ldc) + 32], C_reg_12_2,(16));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg_13_0,(16));
__riscv_vse16_v_f16m1(&C[(13) * (ldc) + 16], C_reg_13_1,(16));
__riscv_vse16_v_f16m1(&C[(13) * (ldc) + 32], C_reg_13_2,(16));
__riscv_vse16_v_f16m1(&C[(14) * (ldc)], C_reg_14_0,(16));
__riscv_vse16_v_f16m1(&C[(14) * (ldc) + 16], C_reg_14_1,(16));
__riscv_vse16_v_f16m1(&C[(14) * (ldc) + 32], C_reg_14_2,(16));
__riscv_vse16_v_f16m1(&C[(15) * (ldc)], C_reg_15_0,(16));
__riscv_vse16_v_f16m1(&C[(15) * (ldc) + 16], C_reg_15_1,(16));
__riscv_vse16_v_f16m1(&C[(15) * (ldc) + 32], C_reg_15_2,(16));
}

// gemm_RVV_48x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 48] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 48] @DRAM
// )
void gemm_RVV_48x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
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
vfloat16m1_t C_reg_4_0;
vfloat16m1_t C_reg_4_1;
vfloat16m1_t C_reg_4_2;
vfloat16m1_t C_reg_5_0;
vfloat16m1_t C_reg_5_1;
vfloat16m1_t C_reg_5_2;
vfloat16m1_t C_reg_6_0;
vfloat16m1_t C_reg_6_1;
vfloat16m1_t C_reg_6_2;
vfloat16m1_t C_reg_7_0;
vfloat16m1_t C_reg_7_1;
vfloat16m1_t C_reg_7_2;
vfloat16m1_t C_reg_8_0;
vfloat16m1_t C_reg_8_1;
vfloat16m1_t C_reg_8_2;
vfloat16m1_t C_reg_9_0;
vfloat16m1_t C_reg_9_1;
vfloat16m1_t C_reg_9_2;
vfloat16m1_t C_reg_10_0;
vfloat16m1_t C_reg_10_1;
vfloat16m1_t C_reg_10_2;
vfloat16m1_t C_reg_11_0;
vfloat16m1_t C_reg_11_1;
vfloat16m1_t C_reg_11_2;
vfloat16m1_t C_reg_12_0;
vfloat16m1_t C_reg_12_1;
vfloat16m1_t C_reg_12_2;
vfloat16m1_t C_reg_13_0;
vfloat16m1_t C_reg_13_1;
vfloat16m1_t C_reg_13_2;
vfloat16m1_t C_reg_14_0;
vfloat16m1_t C_reg_14_1;
vfloat16m1_t C_reg_14_2;
vfloat16m1_t C_reg_15_0;
vfloat16m1_t C_reg_15_1;
vfloat16m1_t C_reg_15_2;
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
C_reg_4_0 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(16));
C_reg_4_1 = __riscv_vle16_v_f16m1(&C[(4) * (ldc) + 16],(16));
C_reg_4_2 = __riscv_vle16_v_f16m1(&C[(4) * (ldc) + 32],(16));
C_reg_5_0 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(16));
C_reg_5_1 = __riscv_vle16_v_f16m1(&C[(5) * (ldc) + 16],(16));
C_reg_5_2 = __riscv_vle16_v_f16m1(&C[(5) * (ldc) + 32],(16));
C_reg_6_0 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(16));
C_reg_6_1 = __riscv_vle16_v_f16m1(&C[(6) * (ldc) + 16],(16));
C_reg_6_2 = __riscv_vle16_v_f16m1(&C[(6) * (ldc) + 32],(16));
C_reg_7_0 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(16));
C_reg_7_1 = __riscv_vle16_v_f16m1(&C[(7) * (ldc) + 16],(16));
C_reg_7_2 = __riscv_vle16_v_f16m1(&C[(7) * (ldc) + 32],(16));
C_reg_8_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(16));
C_reg_8_1 = __riscv_vle16_v_f16m1(&C[(8) * (ldc) + 16],(16));
C_reg_8_2 = __riscv_vle16_v_f16m1(&C[(8) * (ldc) + 32],(16));
C_reg_9_0 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(16));
C_reg_9_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc) + 16],(16));
C_reg_9_2 = __riscv_vle16_v_f16m1(&C[(9) * (ldc) + 32],(16));
C_reg_10_0 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(16));
C_reg_10_1 = __riscv_vle16_v_f16m1(&C[(10) * (ldc) + 16],(16));
C_reg_10_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc) + 32],(16));
C_reg_11_0 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(16));
C_reg_11_1 = __riscv_vle16_v_f16m1(&C[(11) * (ldc) + 16],(16));
C_reg_11_2 = __riscv_vle16_v_f16m1(&C[(11) * (ldc) + 32],(16));
C_reg_12_0 = __riscv_vle16_v_f16m1(&C[(12) * (ldc)],(16));
C_reg_12_1 = __riscv_vle16_v_f16m1(&C[(12) * (ldc) + 16],(16));
C_reg_12_2 = __riscv_vle16_v_f16m1(&C[(12) * (ldc) + 32],(16));
C_reg_13_0 = __riscv_vle16_v_f16m1(&C[(13) * (ldc)],(16));
C_reg_13_1 = __riscv_vle16_v_f16m1(&C[(13) * (ldc) + 16],(16));
C_reg_13_2 = __riscv_vle16_v_f16m1(&C[(13) * (ldc) + 32],(16));
C_reg_14_0 = __riscv_vle16_v_f16m1(&C[(14) * (ldc)],(16));
C_reg_14_1 = __riscv_vle16_v_f16m1(&C[(14) * (ldc) + 16],(16));
C_reg_14_2 = __riscv_vle16_v_f16m1(&C[(14) * (ldc) + 32],(16));
C_reg_15_0 = __riscv_vle16_v_f16m1(&C[(15) * (ldc)],(16));
C_reg_15_1 = __riscv_vle16_v_f16m1(&C[(15) * (ldc) + 16],(16));
C_reg_15_2 = __riscv_vle16_v_f16m1(&C[(15) * (ldc) + 32],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A[(k) * (48)],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A[(k) * (48) + 32],(16));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B[(k) * (16)], A_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B[(k) * (16)], A_reg_1,(16));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B[(k) * (16)], A_reg_2,(16));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B[(k) * (16) + 1], A_reg_0,(16));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B[(k) * (16) + 1], A_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B[(k) * (16) + 1], A_reg_2,(16));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B[(k) * (16) + 2], A_reg_0,(16));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B[(k) * (16) + 2], A_reg_1,(16));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B[(k) * (16) + 2], A_reg_2,(16));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B[(k) * (16) + 3], A_reg_0,(16));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B[(k) * (16) + 3], A_reg_1,(16));
  C_reg_3_2 = __riscv_vfmacc_vf_f16m1(C_reg_3_2, B[(k) * (16) + 3], A_reg_2,(16));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B[(k) * (16) + 4], A_reg_0,(16));
  C_reg_4_1 = __riscv_vfmacc_vf_f16m1(C_reg_4_1, B[(k) * (16) + 4], A_reg_1,(16));
  C_reg_4_2 = __riscv_vfmacc_vf_f16m1(C_reg_4_2, B[(k) * (16) + 4], A_reg_2,(16));
  C_reg_5_0 = __riscv_vfmacc_vf_f16m1(C_reg_5_0, B[(k) * (16) + 5], A_reg_0,(16));
  C_reg_5_1 = __riscv_vfmacc_vf_f16m1(C_reg_5_1, B[(k) * (16) + 5], A_reg_1,(16));
  C_reg_5_2 = __riscv_vfmacc_vf_f16m1(C_reg_5_2, B[(k) * (16) + 5], A_reg_2,(16));
  C_reg_6_0 = __riscv_vfmacc_vf_f16m1(C_reg_6_0, B[(k) * (16) + 6], A_reg_0,(16));
  C_reg_6_1 = __riscv_vfmacc_vf_f16m1(C_reg_6_1, B[(k) * (16) + 6], A_reg_1,(16));
  C_reg_6_2 = __riscv_vfmacc_vf_f16m1(C_reg_6_2, B[(k) * (16) + 6], A_reg_2,(16));
  C_reg_7_0 = __riscv_vfmacc_vf_f16m1(C_reg_7_0, B[(k) * (16) + 7], A_reg_0,(16));
  C_reg_7_1 = __riscv_vfmacc_vf_f16m1(C_reg_7_1, B[(k) * (16) + 7], A_reg_1,(16));
  C_reg_7_2 = __riscv_vfmacc_vf_f16m1(C_reg_7_2, B[(k) * (16) + 7], A_reg_2,(16));
  C_reg_8_0 = __riscv_vfmacc_vf_f16m1(C_reg_8_0, B[(k) * (16) + 8], A_reg_0,(16));
  C_reg_8_1 = __riscv_vfmacc_vf_f16m1(C_reg_8_1, B[(k) * (16) + 8], A_reg_1,(16));
  C_reg_8_2 = __riscv_vfmacc_vf_f16m1(C_reg_8_2, B[(k) * (16) + 8], A_reg_2,(16));
  C_reg_9_0 = __riscv_vfmacc_vf_f16m1(C_reg_9_0, B[(k) * (16) + 9], A_reg_0,(16));
  C_reg_9_1 = __riscv_vfmacc_vf_f16m1(C_reg_9_1, B[(k) * (16) + 9], A_reg_1,(16));
  C_reg_9_2 = __riscv_vfmacc_vf_f16m1(C_reg_9_2, B[(k) * (16) + 9], A_reg_2,(16));
  C_reg_10_0 = __riscv_vfmacc_vf_f16m1(C_reg_10_0, B[(k) * (16) + 10], A_reg_0,(16));
  C_reg_10_1 = __riscv_vfmacc_vf_f16m1(C_reg_10_1, B[(k) * (16) + 10], A_reg_1,(16));
  C_reg_10_2 = __riscv_vfmacc_vf_f16m1(C_reg_10_2, B[(k) * (16) + 10], A_reg_2,(16));
  C_reg_11_0 = __riscv_vfmacc_vf_f16m1(C_reg_11_0, B[(k) * (16) + 11], A_reg_0,(16));
  C_reg_11_1 = __riscv_vfmacc_vf_f16m1(C_reg_11_1, B[(k) * (16) + 11], A_reg_1,(16));
  C_reg_11_2 = __riscv_vfmacc_vf_f16m1(C_reg_11_2, B[(k) * (16) + 11], A_reg_2,(16));
  C_reg_12_0 = __riscv_vfmacc_vf_f16m1(C_reg_12_0, B[(k) * (16) + 12], A_reg_0,(16));
  C_reg_12_1 = __riscv_vfmacc_vf_f16m1(C_reg_12_1, B[(k) * (16) + 12], A_reg_1,(16));
  C_reg_12_2 = __riscv_vfmacc_vf_f16m1(C_reg_12_2, B[(k) * (16) + 12], A_reg_2,(16));
  C_reg_13_0 = __riscv_vfmacc_vf_f16m1(C_reg_13_0, B[(k) * (16) + 13], A_reg_0,(16));
  C_reg_13_1 = __riscv_vfmacc_vf_f16m1(C_reg_13_1, B[(k) * (16) + 13], A_reg_1,(16));
  C_reg_13_2 = __riscv_vfmacc_vf_f16m1(C_reg_13_2, B[(k) * (16) + 13], A_reg_2,(16));
  C_reg_14_0 = __riscv_vfmacc_vf_f16m1(C_reg_14_0, B[(k) * (16) + 14], A_reg_0,(16));
  C_reg_14_1 = __riscv_vfmacc_vf_f16m1(C_reg_14_1, B[(k) * (16) + 14], A_reg_1,(16));
  C_reg_14_2 = __riscv_vfmacc_vf_f16m1(C_reg_14_2, B[(k) * (16) + 14], A_reg_2,(16));
  C_reg_15_0 = __riscv_vfmacc_vf_f16m1(C_reg_15_0, B[(k) * (16) + 15], A_reg_0,(16));
  C_reg_15_1 = __riscv_vfmacc_vf_f16m1(C_reg_15_1, B[(k) * (16) + 15], A_reg_1,(16));
  C_reg_15_2 = __riscv_vfmacc_vf_f16m1(C_reg_15_2, B[(k) * (16) + 15], A_reg_2,(16));
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
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4_0,(16));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 16], C_reg_4_1,(16));
__riscv_vse16_v_f16m1(&C[(4) * (ldc) + 32], C_reg_4_2,(16));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5_0,(16));
__riscv_vse16_v_f16m1(&C[(5) * (ldc) + 16], C_reg_5_1,(16));
__riscv_vse16_v_f16m1(&C[(5) * (ldc) + 32], C_reg_5_2,(16));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6_0,(16));
__riscv_vse16_v_f16m1(&C[(6) * (ldc) + 16], C_reg_6_1,(16));
__riscv_vse16_v_f16m1(&C[(6) * (ldc) + 32], C_reg_6_2,(16));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7_0,(16));
__riscv_vse16_v_f16m1(&C[(7) * (ldc) + 16], C_reg_7_1,(16));
__riscv_vse16_v_f16m1(&C[(7) * (ldc) + 32], C_reg_7_2,(16));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg_8_0,(16));
__riscv_vse16_v_f16m1(&C[(8) * (ldc) + 16], C_reg_8_1,(16));
__riscv_vse16_v_f16m1(&C[(8) * (ldc) + 32], C_reg_8_2,(16));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg_9_0,(16));
__riscv_vse16_v_f16m1(&C[(9) * (ldc) + 16], C_reg_9_1,(16));
__riscv_vse16_v_f16m1(&C[(9) * (ldc) + 32], C_reg_9_2,(16));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg_10_0,(16));
__riscv_vse16_v_f16m1(&C[(10) * (ldc) + 16], C_reg_10_1,(16));
__riscv_vse16_v_f16m1(&C[(10) * (ldc) + 32], C_reg_10_2,(16));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg_11_0,(16));
__riscv_vse16_v_f16m1(&C[(11) * (ldc) + 16], C_reg_11_1,(16));
__riscv_vse16_v_f16m1(&C[(11) * (ldc) + 32], C_reg_11_2,(16));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg_12_0,(16));
__riscv_vse16_v_f16m1(&C[(12) * (ldc) + 16], C_reg_12_1,(16));
__riscv_vse16_v_f16m1(&C[(12) * (ldc) + 32], C_reg_12_2,(16));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg_13_0,(16));
__riscv_vse16_v_f16m1(&C[(13) * (ldc) + 16], C_reg_13_1,(16));
__riscv_vse16_v_f16m1(&C[(13) * (ldc) + 32], C_reg_13_2,(16));
__riscv_vse16_v_f16m1(&C[(14) * (ldc)], C_reg_14_0,(16));
__riscv_vse16_v_f16m1(&C[(14) * (ldc) + 16], C_reg_14_1,(16));
__riscv_vse16_v_f16m1(&C[(14) * (ldc) + 32], C_reg_14_2,(16));
__riscv_vse16_v_f16m1(&C[(15) * (ldc)], C_reg_15_0,(16));
__riscv_vse16_v_f16m1(&C[(15) * (ldc) + 16], C_reg_15_1,(16));
__riscv_vse16_v_f16m1(&C[(15) * (ldc) + 32], C_reg_15_2,(16));
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
