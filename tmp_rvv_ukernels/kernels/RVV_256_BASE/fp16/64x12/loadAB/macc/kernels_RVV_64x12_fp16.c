#include "kernels_RVV_64x12_fp16.h"



#include <stdio.h>
#include <stdlib.h>

#include <riscv_vector.h>


// gemm_RVV_64x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 64] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 64] @DRAM
// )
void gemm_RVV_64x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_4_0;
vfloat16m1_t C_reg_4_1;
vfloat16m1_t C_reg_4_2;
vfloat16m1_t C_reg_4_3;
vfloat16m1_t C_reg_5_0;
vfloat16m1_t C_reg_5_1;
vfloat16m1_t C_reg_5_2;
vfloat16m1_t C_reg_5_3;
vfloat16m1_t C_reg_6_0;
vfloat16m1_t C_reg_6_1;
vfloat16m1_t C_reg_6_2;
vfloat16m1_t C_reg_6_3;
vfloat16m1_t C_reg_7_0;
vfloat16m1_t C_reg_7_1;
vfloat16m1_t C_reg_7_2;
vfloat16m1_t C_reg_7_3;
vfloat16m1_t C_reg_8_0;
vfloat16m1_t C_reg_8_1;
vfloat16m1_t C_reg_8_2;
vfloat16m1_t C_reg_8_3;
vfloat16m1_t C_reg_9_0;
vfloat16m1_t C_reg_9_1;
vfloat16m1_t C_reg_9_2;
vfloat16m1_t C_reg_9_3;
vfloat16m1_t C_reg_10_0;
vfloat16m1_t C_reg_10_1;
vfloat16m1_t C_reg_10_2;
vfloat16m1_t C_reg_10_3;
vfloat16m1_t C_reg_11_0;
vfloat16m1_t C_reg_11_1;
vfloat16m1_t C_reg_11_2;
vfloat16m1_t C_reg_11_3;
C_reg_0_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_0_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_1_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_3_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_4_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_4_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_4_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_4_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_5_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_5_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_5_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_5_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_6_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_6_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_6_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_6_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_7_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_7_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_7_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_7_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_8_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_8_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_8_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_8_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_9_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_9_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_9_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_9_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_10_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_10_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_10_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_10_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_11_0 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_11_1 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_11_2 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
C_reg_11_3 = __riscv_vfmv_v_f_f16m1(0.0f,(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B.data[(k) * (B.strides[0])], A_reg_1,(16));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B.data[(k) * (B.strides[0])], A_reg_2,(16));
  C_reg_0_3 = __riscv_vfmacc_vf_f16m1(C_reg_0_3, B.data[(k) * (B.strides[0])], A_reg_3,(16));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B.data[(k) * (B.strides[0]) + 1], A_reg_0,(16));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B.data[(k) * (B.strides[0]) + 1], A_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B.data[(k) * (B.strides[0]) + 1], A_reg_2,(16));
  C_reg_1_3 = __riscv_vfmacc_vf_f16m1(C_reg_1_3, B.data[(k) * (B.strides[0]) + 1], A_reg_3,(16));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B.data[(k) * (B.strides[0]) + 2], A_reg_0,(16));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B.data[(k) * (B.strides[0]) + 2], A_reg_1,(16));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B.data[(k) * (B.strides[0]) + 2], A_reg_2,(16));
  C_reg_2_3 = __riscv_vfmacc_vf_f16m1(C_reg_2_3, B.data[(k) * (B.strides[0]) + 2], A_reg_3,(16));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B.data[(k) * (B.strides[0]) + 3], A_reg_0,(16));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B.data[(k) * (B.strides[0]) + 3], A_reg_1,(16));
  C_reg_3_2 = __riscv_vfmacc_vf_f16m1(C_reg_3_2, B.data[(k) * (B.strides[0]) + 3], A_reg_2,(16));
  C_reg_3_3 = __riscv_vfmacc_vf_f16m1(C_reg_3_3, B.data[(k) * (B.strides[0]) + 3], A_reg_3,(16));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B.data[(k) * (B.strides[0]) + 4], A_reg_0,(16));
  C_reg_4_1 = __riscv_vfmacc_vf_f16m1(C_reg_4_1, B.data[(k) * (B.strides[0]) + 4], A_reg_1,(16));
  C_reg_4_2 = __riscv_vfmacc_vf_f16m1(C_reg_4_2, B.data[(k) * (B.strides[0]) + 4], A_reg_2,(16));
  C_reg_4_3 = __riscv_vfmacc_vf_f16m1(C_reg_4_3, B.data[(k) * (B.strides[0]) + 4], A_reg_3,(16));
  C_reg_5_0 = __riscv_vfmacc_vf_f16m1(C_reg_5_0, B.data[(k) * (B.strides[0]) + 5], A_reg_0,(16));
  C_reg_5_1 = __riscv_vfmacc_vf_f16m1(C_reg_5_1, B.data[(k) * (B.strides[0]) + 5], A_reg_1,(16));
  C_reg_5_2 = __riscv_vfmacc_vf_f16m1(C_reg_5_2, B.data[(k) * (B.strides[0]) + 5], A_reg_2,(16));
  C_reg_5_3 = __riscv_vfmacc_vf_f16m1(C_reg_5_3, B.data[(k) * (B.strides[0]) + 5], A_reg_3,(16));
  C_reg_6_0 = __riscv_vfmacc_vf_f16m1(C_reg_6_0, B.data[(k) * (B.strides[0]) + 6], A_reg_0,(16));
  C_reg_6_1 = __riscv_vfmacc_vf_f16m1(C_reg_6_1, B.data[(k) * (B.strides[0]) + 6], A_reg_1,(16));
  C_reg_6_2 = __riscv_vfmacc_vf_f16m1(C_reg_6_2, B.data[(k) * (B.strides[0]) + 6], A_reg_2,(16));
  C_reg_6_3 = __riscv_vfmacc_vf_f16m1(C_reg_6_3, B.data[(k) * (B.strides[0]) + 6], A_reg_3,(16));
  C_reg_7_0 = __riscv_vfmacc_vf_f16m1(C_reg_7_0, B.data[(k) * (B.strides[0]) + 7], A_reg_0,(16));
  C_reg_7_1 = __riscv_vfmacc_vf_f16m1(C_reg_7_1, B.data[(k) * (B.strides[0]) + 7], A_reg_1,(16));
  C_reg_7_2 = __riscv_vfmacc_vf_f16m1(C_reg_7_2, B.data[(k) * (B.strides[0]) + 7], A_reg_2,(16));
  C_reg_7_3 = __riscv_vfmacc_vf_f16m1(C_reg_7_3, B.data[(k) * (B.strides[0]) + 7], A_reg_3,(16));
  C_reg_8_0 = __riscv_vfmacc_vf_f16m1(C_reg_8_0, B.data[(k) * (B.strides[0]) + 8], A_reg_0,(16));
  C_reg_8_1 = __riscv_vfmacc_vf_f16m1(C_reg_8_1, B.data[(k) * (B.strides[0]) + 8], A_reg_1,(16));
  C_reg_8_2 = __riscv_vfmacc_vf_f16m1(C_reg_8_2, B.data[(k) * (B.strides[0]) + 8], A_reg_2,(16));
  C_reg_8_3 = __riscv_vfmacc_vf_f16m1(C_reg_8_3, B.data[(k) * (B.strides[0]) + 8], A_reg_3,(16));
  C_reg_9_0 = __riscv_vfmacc_vf_f16m1(C_reg_9_0, B.data[(k) * (B.strides[0]) + 9], A_reg_0,(16));
  C_reg_9_1 = __riscv_vfmacc_vf_f16m1(C_reg_9_1, B.data[(k) * (B.strides[0]) + 9], A_reg_1,(16));
  C_reg_9_2 = __riscv_vfmacc_vf_f16m1(C_reg_9_2, B.data[(k) * (B.strides[0]) + 9], A_reg_2,(16));
  C_reg_9_3 = __riscv_vfmacc_vf_f16m1(C_reg_9_3, B.data[(k) * (B.strides[0]) + 9], A_reg_3,(16));
  C_reg_10_0 = __riscv_vfmacc_vf_f16m1(C_reg_10_0, B.data[(k) * (B.strides[0]) + 10], A_reg_0,(16));
  C_reg_10_1 = __riscv_vfmacc_vf_f16m1(C_reg_10_1, B.data[(k) * (B.strides[0]) + 10], A_reg_1,(16));
  C_reg_10_2 = __riscv_vfmacc_vf_f16m1(C_reg_10_2, B.data[(k) * (B.strides[0]) + 10], A_reg_2,(16));
  C_reg_10_3 = __riscv_vfmacc_vf_f16m1(C_reg_10_3, B.data[(k) * (B.strides[0]) + 10], A_reg_3,(16));
  C_reg_11_0 = __riscv_vfmacc_vf_f16m1(C_reg_11_0, B.data[(k) * (B.strides[0]) + 11], A_reg_0,(16));
  C_reg_11_1 = __riscv_vfmacc_vf_f16m1(C_reg_11_1, B.data[(k) * (B.strides[0]) + 11], A_reg_1,(16));
  C_reg_11_2 = __riscv_vfmacc_vf_f16m1(C_reg_11_2, B.data[(k) * (B.strides[0]) + 11], A_reg_2,(16));
  C_reg_11_3 = __riscv_vfmacc_vf_f16m1(C_reg_11_3, B.data[(k) * (B.strides[0]) + 11], A_reg_3,(16));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0]) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0]) + 32], C_reg_2_2,(16));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0]) + 48], C_reg_2_3,(16));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0]) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0]) + 32], C_reg_3_2,(16));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0]) + 48], C_reg_3_3,(16));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(16));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0]) + 16], C_reg_4_1,(16));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0]) + 32], C_reg_4_2,(16));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0]) + 48], C_reg_4_3,(16));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5_0,(16));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0]) + 16], C_reg_5_1,(16));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0]) + 32], C_reg_5_2,(16));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0]) + 48], C_reg_5_3,(16));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6_0,(16));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0]) + 16], C_reg_6_1,(16));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0]) + 32], C_reg_6_2,(16));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0]) + 48], C_reg_6_3,(16));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7_0,(16));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0]) + 16], C_reg_7_1,(16));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0]) + 32], C_reg_7_2,(16));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0]) + 48], C_reg_7_3,(16));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8_0,(16));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0]) + 16], C_reg_8_1,(16));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0]) + 32], C_reg_8_2,(16));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0]) + 48], C_reg_8_3,(16));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9_0,(16));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0]) + 16], C_reg_9_1,(16));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0]) + 32], C_reg_9_2,(16));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0]) + 48], C_reg_9_3,(16));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10_0,(16));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0]) + 16], C_reg_10_1,(16));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0]) + 32], C_reg_10_2,(16));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0]) + 48], C_reg_10_3,(16));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11_0,(16));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0]) + 16], C_reg_11_1,(16));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0]) + 32], C_reg_11_2,(16));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0]) + 48], C_reg_11_3,(16));
}

// gemm_RVV_64x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 64] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 64] @DRAM
// )
void gemm_RVV_64x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
vfloat16m1_t C_reg_4_0;
vfloat16m1_t C_reg_4_1;
vfloat16m1_t C_reg_4_2;
vfloat16m1_t C_reg_4_3;
vfloat16m1_t C_reg_5_0;
vfloat16m1_t C_reg_5_1;
vfloat16m1_t C_reg_5_2;
vfloat16m1_t C_reg_5_3;
vfloat16m1_t C_reg_6_0;
vfloat16m1_t C_reg_6_1;
vfloat16m1_t C_reg_6_2;
vfloat16m1_t C_reg_6_3;
vfloat16m1_t C_reg_7_0;
vfloat16m1_t C_reg_7_1;
vfloat16m1_t C_reg_7_2;
vfloat16m1_t C_reg_7_3;
vfloat16m1_t C_reg_8_0;
vfloat16m1_t C_reg_8_1;
vfloat16m1_t C_reg_8_2;
vfloat16m1_t C_reg_8_3;
vfloat16m1_t C_reg_9_0;
vfloat16m1_t C_reg_9_1;
vfloat16m1_t C_reg_9_2;
vfloat16m1_t C_reg_9_3;
vfloat16m1_t C_reg_10_0;
vfloat16m1_t C_reg_10_1;
vfloat16m1_t C_reg_10_2;
vfloat16m1_t C_reg_10_3;
vfloat16m1_t C_reg_11_0;
vfloat16m1_t C_reg_11_1;
vfloat16m1_t C_reg_11_2;
vfloat16m1_t C_reg_11_3;
C_reg_0_0 = __riscv_vle16_v_f16m1(&C.data[0],(16));
C_reg_0_1 = __riscv_vle16_v_f16m1(&C.data[16],(16));
C_reg_0_2 = __riscv_vle16_v_f16m1(&C.data[32],(16));
C_reg_0_3 = __riscv_vle16_v_f16m1(&C.data[48],(16));
C_reg_1_0 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(16));
C_reg_1_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 16],(16));
C_reg_1_2 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 32],(16));
C_reg_1_3 = __riscv_vle16_v_f16m1(&C.data[C.strides[0] + 48],(16));
C_reg_2_0 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(16));
C_reg_2_1 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0]) + 16],(16));
C_reg_2_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0]) + 32],(16));
C_reg_2_3 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0]) + 48],(16));
C_reg_3_0 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(16));
C_reg_3_1 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0]) + 16],(16));
C_reg_3_2 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0]) + 32],(16));
C_reg_3_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0]) + 48],(16));
C_reg_4_0 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0])],(16));
C_reg_4_1 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0]) + 16],(16));
C_reg_4_2 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0]) + 32],(16));
C_reg_4_3 = __riscv_vle16_v_f16m1(&C.data[(4) * (C.strides[0]) + 48],(16));
C_reg_5_0 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0])],(16));
C_reg_5_1 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0]) + 16],(16));
C_reg_5_2 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0]) + 32],(16));
C_reg_5_3 = __riscv_vle16_v_f16m1(&C.data[(5) * (C.strides[0]) + 48],(16));
C_reg_6_0 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0])],(16));
C_reg_6_1 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0]) + 16],(16));
C_reg_6_2 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0]) + 32],(16));
C_reg_6_3 = __riscv_vle16_v_f16m1(&C.data[(6) * (C.strides[0]) + 48],(16));
C_reg_7_0 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0])],(16));
C_reg_7_1 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0]) + 16],(16));
C_reg_7_2 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0]) + 32],(16));
C_reg_7_3 = __riscv_vle16_v_f16m1(&C.data[(7) * (C.strides[0]) + 48],(16));
C_reg_8_0 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0])],(16));
C_reg_8_1 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0]) + 16],(16));
C_reg_8_2 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0]) + 32],(16));
C_reg_8_3 = __riscv_vle16_v_f16m1(&C.data[(8) * (C.strides[0]) + 48],(16));
C_reg_9_0 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0])],(16));
C_reg_9_1 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0]) + 16],(16));
C_reg_9_2 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0]) + 32],(16));
C_reg_9_3 = __riscv_vle16_v_f16m1(&C.data[(9) * (C.strides[0]) + 48],(16));
C_reg_10_0 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0])],(16));
C_reg_10_1 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0]) + 16],(16));
C_reg_10_2 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0]) + 32],(16));
C_reg_10_3 = __riscv_vle16_v_f16m1(&C.data[(10) * (C.strides[0]) + 48],(16));
C_reg_11_0 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0])],(16));
C_reg_11_1 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0]) + 16],(16));
C_reg_11_2 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0]) + 32],(16));
C_reg_11_3 = __riscv_vle16_v_f16m1(&C.data[(11) * (C.strides[0]) + 48],(16));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(16));
  A_reg_1 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 16],(16));
  A_reg_2 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 32],(16));
  A_reg_3 = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0]) + 48],(16));
  C_reg_0_0 = __riscv_vfmacc_vf_f16m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(16));
  C_reg_0_1 = __riscv_vfmacc_vf_f16m1(C_reg_0_1, B.data[(k) * (B.strides[0])], A_reg_1,(16));
  C_reg_0_2 = __riscv_vfmacc_vf_f16m1(C_reg_0_2, B.data[(k) * (B.strides[0])], A_reg_2,(16));
  C_reg_0_3 = __riscv_vfmacc_vf_f16m1(C_reg_0_3, B.data[(k) * (B.strides[0])], A_reg_3,(16));
  C_reg_1_0 = __riscv_vfmacc_vf_f16m1(C_reg_1_0, B.data[(k) * (B.strides[0]) + 1], A_reg_0,(16));
  C_reg_1_1 = __riscv_vfmacc_vf_f16m1(C_reg_1_1, B.data[(k) * (B.strides[0]) + 1], A_reg_1,(16));
  C_reg_1_2 = __riscv_vfmacc_vf_f16m1(C_reg_1_2, B.data[(k) * (B.strides[0]) + 1], A_reg_2,(16));
  C_reg_1_3 = __riscv_vfmacc_vf_f16m1(C_reg_1_3, B.data[(k) * (B.strides[0]) + 1], A_reg_3,(16));
  C_reg_2_0 = __riscv_vfmacc_vf_f16m1(C_reg_2_0, B.data[(k) * (B.strides[0]) + 2], A_reg_0,(16));
  C_reg_2_1 = __riscv_vfmacc_vf_f16m1(C_reg_2_1, B.data[(k) * (B.strides[0]) + 2], A_reg_1,(16));
  C_reg_2_2 = __riscv_vfmacc_vf_f16m1(C_reg_2_2, B.data[(k) * (B.strides[0]) + 2], A_reg_2,(16));
  C_reg_2_3 = __riscv_vfmacc_vf_f16m1(C_reg_2_3, B.data[(k) * (B.strides[0]) + 2], A_reg_3,(16));
  C_reg_3_0 = __riscv_vfmacc_vf_f16m1(C_reg_3_0, B.data[(k) * (B.strides[0]) + 3], A_reg_0,(16));
  C_reg_3_1 = __riscv_vfmacc_vf_f16m1(C_reg_3_1, B.data[(k) * (B.strides[0]) + 3], A_reg_1,(16));
  C_reg_3_2 = __riscv_vfmacc_vf_f16m1(C_reg_3_2, B.data[(k) * (B.strides[0]) + 3], A_reg_2,(16));
  C_reg_3_3 = __riscv_vfmacc_vf_f16m1(C_reg_3_3, B.data[(k) * (B.strides[0]) + 3], A_reg_3,(16));
  C_reg_4_0 = __riscv_vfmacc_vf_f16m1(C_reg_4_0, B.data[(k) * (B.strides[0]) + 4], A_reg_0,(16));
  C_reg_4_1 = __riscv_vfmacc_vf_f16m1(C_reg_4_1, B.data[(k) * (B.strides[0]) + 4], A_reg_1,(16));
  C_reg_4_2 = __riscv_vfmacc_vf_f16m1(C_reg_4_2, B.data[(k) * (B.strides[0]) + 4], A_reg_2,(16));
  C_reg_4_3 = __riscv_vfmacc_vf_f16m1(C_reg_4_3, B.data[(k) * (B.strides[0]) + 4], A_reg_3,(16));
  C_reg_5_0 = __riscv_vfmacc_vf_f16m1(C_reg_5_0, B.data[(k) * (B.strides[0]) + 5], A_reg_0,(16));
  C_reg_5_1 = __riscv_vfmacc_vf_f16m1(C_reg_5_1, B.data[(k) * (B.strides[0]) + 5], A_reg_1,(16));
  C_reg_5_2 = __riscv_vfmacc_vf_f16m1(C_reg_5_2, B.data[(k) * (B.strides[0]) + 5], A_reg_2,(16));
  C_reg_5_3 = __riscv_vfmacc_vf_f16m1(C_reg_5_3, B.data[(k) * (B.strides[0]) + 5], A_reg_3,(16));
  C_reg_6_0 = __riscv_vfmacc_vf_f16m1(C_reg_6_0, B.data[(k) * (B.strides[0]) + 6], A_reg_0,(16));
  C_reg_6_1 = __riscv_vfmacc_vf_f16m1(C_reg_6_1, B.data[(k) * (B.strides[0]) + 6], A_reg_1,(16));
  C_reg_6_2 = __riscv_vfmacc_vf_f16m1(C_reg_6_2, B.data[(k) * (B.strides[0]) + 6], A_reg_2,(16));
  C_reg_6_3 = __riscv_vfmacc_vf_f16m1(C_reg_6_3, B.data[(k) * (B.strides[0]) + 6], A_reg_3,(16));
  C_reg_7_0 = __riscv_vfmacc_vf_f16m1(C_reg_7_0, B.data[(k) * (B.strides[0]) + 7], A_reg_0,(16));
  C_reg_7_1 = __riscv_vfmacc_vf_f16m1(C_reg_7_1, B.data[(k) * (B.strides[0]) + 7], A_reg_1,(16));
  C_reg_7_2 = __riscv_vfmacc_vf_f16m1(C_reg_7_2, B.data[(k) * (B.strides[0]) + 7], A_reg_2,(16));
  C_reg_7_3 = __riscv_vfmacc_vf_f16m1(C_reg_7_3, B.data[(k) * (B.strides[0]) + 7], A_reg_3,(16));
  C_reg_8_0 = __riscv_vfmacc_vf_f16m1(C_reg_8_0, B.data[(k) * (B.strides[0]) + 8], A_reg_0,(16));
  C_reg_8_1 = __riscv_vfmacc_vf_f16m1(C_reg_8_1, B.data[(k) * (B.strides[0]) + 8], A_reg_1,(16));
  C_reg_8_2 = __riscv_vfmacc_vf_f16m1(C_reg_8_2, B.data[(k) * (B.strides[0]) + 8], A_reg_2,(16));
  C_reg_8_3 = __riscv_vfmacc_vf_f16m1(C_reg_8_3, B.data[(k) * (B.strides[0]) + 8], A_reg_3,(16));
  C_reg_9_0 = __riscv_vfmacc_vf_f16m1(C_reg_9_0, B.data[(k) * (B.strides[0]) + 9], A_reg_0,(16));
  C_reg_9_1 = __riscv_vfmacc_vf_f16m1(C_reg_9_1, B.data[(k) * (B.strides[0]) + 9], A_reg_1,(16));
  C_reg_9_2 = __riscv_vfmacc_vf_f16m1(C_reg_9_2, B.data[(k) * (B.strides[0]) + 9], A_reg_2,(16));
  C_reg_9_3 = __riscv_vfmacc_vf_f16m1(C_reg_9_3, B.data[(k) * (B.strides[0]) + 9], A_reg_3,(16));
  C_reg_10_0 = __riscv_vfmacc_vf_f16m1(C_reg_10_0, B.data[(k) * (B.strides[0]) + 10], A_reg_0,(16));
  C_reg_10_1 = __riscv_vfmacc_vf_f16m1(C_reg_10_1, B.data[(k) * (B.strides[0]) + 10], A_reg_1,(16));
  C_reg_10_2 = __riscv_vfmacc_vf_f16m1(C_reg_10_2, B.data[(k) * (B.strides[0]) + 10], A_reg_2,(16));
  C_reg_10_3 = __riscv_vfmacc_vf_f16m1(C_reg_10_3, B.data[(k) * (B.strides[0]) + 10], A_reg_3,(16));
  C_reg_11_0 = __riscv_vfmacc_vf_f16m1(C_reg_11_0, B.data[(k) * (B.strides[0]) + 11], A_reg_0,(16));
  C_reg_11_1 = __riscv_vfmacc_vf_f16m1(C_reg_11_1, B.data[(k) * (B.strides[0]) + 11], A_reg_1,(16));
  C_reg_11_2 = __riscv_vfmacc_vf_f16m1(C_reg_11_2, B.data[(k) * (B.strides[0]) + 11], A_reg_2,(16));
  C_reg_11_3 = __riscv_vfmacc_vf_f16m1(C_reg_11_3, B.data[(k) * (B.strides[0]) + 11], A_reg_3,(16));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0_0,(16));
__riscv_vse16_v_f16m1(&C.data[16], C_reg_0_1,(16));
__riscv_vse16_v_f16m1(&C.data[32], C_reg_0_2,(16));
__riscv_vse16_v_f16m1(&C.data[48], C_reg_0_3,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1_0,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 16], C_reg_1_1,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 32], C_reg_1_2,(16));
__riscv_vse16_v_f16m1(&C.data[C.strides[0] + 48], C_reg_1_3,(16));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2_0,(16));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0]) + 16], C_reg_2_1,(16));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0]) + 32], C_reg_2_2,(16));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0]) + 48], C_reg_2_3,(16));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3_0,(16));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0]) + 16], C_reg_3_1,(16));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0]) + 32], C_reg_3_2,(16));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0]) + 48], C_reg_3_3,(16));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0])], C_reg_4_0,(16));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0]) + 16], C_reg_4_1,(16));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0]) + 32], C_reg_4_2,(16));
__riscv_vse16_v_f16m1(&C.data[(4) * (C.strides[0]) + 48], C_reg_4_3,(16));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0])], C_reg_5_0,(16));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0]) + 16], C_reg_5_1,(16));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0]) + 32], C_reg_5_2,(16));
__riscv_vse16_v_f16m1(&C.data[(5) * (C.strides[0]) + 48], C_reg_5_3,(16));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0])], C_reg_6_0,(16));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0]) + 16], C_reg_6_1,(16));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0]) + 32], C_reg_6_2,(16));
__riscv_vse16_v_f16m1(&C.data[(6) * (C.strides[0]) + 48], C_reg_6_3,(16));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0])], C_reg_7_0,(16));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0]) + 16], C_reg_7_1,(16));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0]) + 32], C_reg_7_2,(16));
__riscv_vse16_v_f16m1(&C.data[(7) * (C.strides[0]) + 48], C_reg_7_3,(16));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0])], C_reg_8_0,(16));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0]) + 16], C_reg_8_1,(16));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0]) + 32], C_reg_8_2,(16));
__riscv_vse16_v_f16m1(&C.data[(8) * (C.strides[0]) + 48], C_reg_8_3,(16));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0])], C_reg_9_0,(16));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0]) + 16], C_reg_9_1,(16));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0]) + 32], C_reg_9_2,(16));
__riscv_vse16_v_f16m1(&C.data[(9) * (C.strides[0]) + 48], C_reg_9_3,(16));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0])], C_reg_10_0,(16));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0]) + 16], C_reg_10_1,(16));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0]) + 32], C_reg_10_2,(16));
__riscv_vse16_v_f16m1(&C.data[(10) * (C.strides[0]) + 48], C_reg_10_3,(16));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0])], C_reg_11_0,(16));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0]) + 16], C_reg_11_1,(16));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0]) + 32], C_reg_11_2,(16));
__riscv_vse16_v_f16m1(&C.data[(11) * (C.strides[0]) + 48], C_reg_11_3,(16));
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
